"""Schema indexing, compact M-Schema extraction, and BM25 value indexing for Assay.

Extracts:
  - Compact M-Schema metadata: backtick columns, types, PK/FK constraints,
    column descriptions from BIRD CSVs, 3-5 distinct non-null sample values,
    and explicit NULL presence flags.
  - Distinct text values for lexical cell retrieval.
  - BM25 indexes over tables, columns, and text cell values.
  - Precomputed dense embeddings for table and column representations.
  - Foreign key graph for shortest-path join closure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.llm import default_client


def tokenize(text: Any) -> List[str]:
    """Acronym- and camelCase-aware tokenizer for schema and query text."""
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    # Split acronyms followed by words: CDSCode -> CDS Code
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Split camelCase: satScores -> sat Scores
    s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
    # Replace non-alphanumeric with spaces
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)
    return [w.lower() for w in s.split() if w]


def safe_read_csv(path: Path) -> List[List[str]]:
    """Read a CSV file safely across multiple encodings (UTF-8-SIG, CP1252, Latin-1)."""
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return list(csv.reader(f))
        except (UnicodeDecodeError, Exception):
            continue
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return list(csv.reader(f))
    except Exception:
        return []


def load_bird_descriptions(db_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load table and column descriptions from database_description/*.csv.
    
    Returns:
        Dict mapping table_name_lower -> col_name_lower -> {
            "original_column_name": ...,
            "column_name": ...,
            "column_description": ...,
            "value_description": ...
        }
    """
    desc_dir = db_dir / "database_description"
    if not desc_dir.exists() or not desc_dir.is_dir():
        return {}

    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    for csv_file in desc_dir.glob("*.csv"):
        tbl_name = csv_file.stem.strip()
        rows = safe_read_csv(csv_file)
        if not rows or len(rows) < 2:
            continue

        header = [h.strip().lower() for h in rows[0]]
        col_map: Dict[str, Dict[str, str]] = {}
        for row in rows[1:]:
            if not row:
                continue
            item: Dict[str, str] = {}
            for idx, val in enumerate(row):
                if idx < len(header):
                    item[header[idx]] = val.strip()

            orig_col = item.get("original_column_name", "") or item.get("column_name", "")
            if orig_col:
                col_map[orig_col.strip().lower()] = {
                    "original_column_name": orig_col.strip(),
                    "column_name": item.get("column_name", "").strip(),
                    "column_description": item.get("column_description", "").strip(),
                    "value_description": item.get("value_description", "").strip(),
                }

        result[tbl_name.lower()] = col_map
    return result


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_pk: bool = False
    foreign_keys: List[Tuple[str, str]] = field(default_factory=list)  # [(target_table, target_col), ...]
    description: str = ""
    samples: List[Any] = field(default_factory=list)
    has_null: bool = False
    distinct_text_values: List[str] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    description: str = ""
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Tuple[str, str, str]] = field(default_factory=list)  # [(from_col, to_table, to_col), ...]


@dataclass
class DatabaseIndex:
    db_id: str
    tables: Dict[str, TableInfo]
    # FK Graph: table -> list of (target_table, from_col, to_col)
    fk_graph: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)
    # BM25 over tables
    bm25_tables: Optional[BM25Okapi] = None
    table_names_list: List[str] = field(default_factory=list)
    # BM25 over columns
    bm25_columns: Optional[BM25Okapi] = None
    column_keys_list: List[Tuple[str, str]] = field(default_factory=list)  # (table, column)
    # BM25 over text cell values
    bm25_values: Optional[BM25Okapi] = None
    value_entries: List[Tuple[str, str, str]] = field(default_factory=list)  # (table, column, val_str)
    # Dense embeddings: (table, col) -> vector
    column_embeddings: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    # Dense embeddings: table -> vector
    table_embeddings: Dict[str, List[float]] = field(default_factory=dict)


def extract_database_index(
    db_path: Path,
    db_id: str,
    tables_json_path: Optional[Path] = None,
    compute_embeddings: bool = True,
    max_distinct_values: int = 300,
) -> DatabaseIndex:
    """Extract schema, sample values, build BM25 and dense indices for a single database."""
    abs_db_path = Path(db_path).resolve()
    if not abs_db_path.exists():
        raise FileNotFoundError(f"Database file not found: {abs_db_path}")

    db_dir = abs_db_path.parent
    csv_descriptions = load_bird_descriptions(db_dir)

    # 1. Parse dev_tables.json if available
    tables_json_entry = None
    if tables_json_path and Path(tables_json_path).exists():
        with open(tables_json_path, "r", encoding="utf-8") as f:
            all_tables = json.load(f)
            for entry in all_tables:
                if entry.get("db_id") == db_id:
                    tables_json_entry = entry
                    break

    # 2. Connect to SQLite in read-only mode
    conn = sqlite3.connect(f"file:{abs_db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    # Get all user tables and views from sqlite_master
    cur.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    table_rows = cur.fetchall()

    tables: Dict[str, TableInfo] = {}

    # Extract tables from sqlite_master
    for tbl_name, tbl_type in table_rows:
        t_info = TableInfo(name=tbl_name)
        tables[tbl_name] = t_info

    # Map tables and columns from tables_json_entry for PKs, FKs, normalized names
    fk_list_all: List[Tuple[str, str, str, str]] = []  # (from_table, from_col, to_table, to_col)
    pk_map: Dict[str, Set[str]] = {}  # table -> set of pk columns

    if tables_json_entry:
        tbl_names_orig = tables_json_entry.get("table_names_original", [])
        tbl_names_clean = tables_json_entry.get("table_names", [])
        col_names_orig = tables_json_entry.get("column_names_original", [])
        col_names_clean = tables_json_entry.get("column_names", [])
        col_types = tables_json_entry.get("column_types", [])
        primary_keys = tables_json_entry.get("primary_keys", [])
        foreign_keys = tables_json_entry.get("foreign_keys", [])

        # Assign table descriptions from cleaned names
        for idx, t_orig in enumerate(tbl_names_orig):
            for t_key, t_obj in tables.items():
                if t_key.lower() == t_orig.lower():
                    clean_name = tbl_names_clean[idx] if idx < len(tbl_names_clean) else ""
                    if clean_name and clean_name.lower() != t_orig.lower():
                        t_obj.description = clean_name
                    break

        # Extract PKs
        for pk in primary_keys:
            if isinstance(pk, int) and 0 <= pk < len(col_names_orig):
                t_idx, c_name = col_names_orig[pk]
                if 0 <= t_idx < len(tbl_names_orig):
                    t_name = tbl_names_orig[t_idx]
                    pk_map.setdefault(t_name.lower(), set()).add(c_name.lower())
            elif isinstance(pk, list):
                for p in pk:
                    if isinstance(p, int) and 0 <= p < len(col_names_orig):
                        t_idx, c_name = col_names_orig[p]
                        if 0 <= t_idx < len(tbl_names_orig):
                            t_name = tbl_names_orig[t_idx]
                            pk_map.setdefault(t_name.lower(), set()).add(c_name.lower())

        # Extract FKs: foreign_keys is list of [col_idx_from, col_idx_to]
        for fk in foreign_keys:
            if len(fk) == 2:
                idx_from, idx_to = fk[0], fk[1]
                if 0 <= idx_from < len(col_names_orig) and 0 <= idx_to < len(col_names_orig):
                    t_idx_from, c_from = col_names_orig[idx_from]
                    t_idx_to, c_to = col_names_orig[idx_to]
                    if 0 <= t_idx_from < len(tbl_names_orig) and 0 <= t_idx_to < len(tbl_names_orig):
                        t_from = tbl_names_orig[t_idx_from]
                        t_to = tbl_names_orig[t_idx_to]
                        fk_list_all.append((t_from, c_from, t_to, c_to))

    # Inspect each table in SQLite
    for tbl_name, t_info in tables.items():
        tbl_csv_desc = csv_descriptions.get(tbl_name.lower(), {})

        # Query PRAGMA table_info
        try:
            cur.execute(f'PRAGMA table_info("{tbl_name}");')
            col_rows = cur.fetchall()
        except Exception:
            col_rows = []

        for cid, col_name, col_type, notnull, dflt_value, pk_flag in col_rows:
            is_pk = bool(pk_flag) or (col_name.lower() in pk_map.get(tbl_name.lower(), set()))
            if is_pk and col_name not in t_info.primary_keys:
                t_info.primary_keys.append(col_name)

            col_meta = ColumnInfo(
                name=col_name,
                data_type=col_type.upper() if col_type else "TEXT",
                is_pk=is_pk,
            )

            # Check description from CSV
            if col_name.lower() in tbl_csv_desc:
                info_csv = tbl_csv_desc[col_name.lower()]
                desc_parts = []
                c_clean = info_csv.get("column_name", "")
                c_desc = info_csv.get("column_description", "")
                c_val_desc = info_csv.get("value_description", "")
                if c_clean and c_clean.lower() != col_name.lower():
                    desc_parts.append(c_clean)
                if c_desc and c_desc.lower() not in (col_name.lower(), c_clean.lower()):
                    desc_parts.append(c_desc)
                if c_val_desc:
                    desc_parts.append(f"Notes: {c_val_desc}")
                col_meta.description = " | ".join(desc_parts)

            # Query samples and nulls from SQLite
            try:
                # Check has_null
                cur.execute(f'SELECT 1 FROM "{tbl_name}" WHERE "{col_name}" IS NULL LIMIT 1;')
                col_meta.has_null = cur.fetchone() is not None

                # Fetch 3-5 distinct non-null values
                cur.execute(
                    f'SELECT DISTINCT "{col_name}" FROM "{tbl_name}" '
                    f'WHERE "{col_name}" IS NOT NULL AND "{col_name}" != \'\' '
                    f'LIMIT 5;'
                )
                samples = [row[0] for row in cur.fetchall()]
                col_meta.samples = samples

                # If text-like or samples are strings, sample distinct values for cell retrieval
                is_text = (
                    "text" in col_meta.data_type.lower()
                    or "char" in col_meta.data_type.lower()
                    or "clob" in col_meta.data_type.lower()
                    or (samples and isinstance(samples[0], str))
                )
                if is_text:
                    cur.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "{tbl_name}" '
                        f'WHERE "{col_name}" IS NOT NULL AND length(CAST("{col_name}" AS TEXT)) BETWEEN 1 AND 80 '
                        f'LIMIT {max_distinct_values};'
                    )
                    distinct_vals = [
                        str(r[0]).strip() for r in cur.fetchall() if str(r[0]).strip()
                    ]
                    col_meta.distinct_text_values = distinct_vals

            except Exception:
                pass

            t_info.columns[col_name] = col_meta

    # Wire foreign keys into TableInfo and ColumnInfo
    for t_from, c_from, t_to, c_to in fk_list_all:
        # Match case-insensitively to registered tables
        t_from_match = next((t for t in tables if t.lower() == t_from.lower()), None)
        t_to_match = next((t for t in tables if t.lower() == t_to.lower()), None)
        if t_from_match and t_to_match:
            c_from_match = next((c for c in tables[t_from_match].columns if c.lower() == c_from.lower()), c_from)
            c_to_match = next((c for c in tables[t_to_match].columns if c.lower() == c_to.lower()), c_to)

            tables[t_from_match].foreign_keys.append((c_from_match, t_to_match, c_to_match))
            if c_from_match in tables[t_from_match].columns:
                tables[t_from_match].columns[c_from_match].foreign_keys.append((t_to_match, c_to_match))

    conn.close()

    # 3. Build Foreign Key Graph (undirected adjacency list for shortest paths)
    fk_graph: Dict[str, List[Tuple[str, str, str]]] = {}
    for t_name, t_info in tables.items():
        fk_graph.setdefault(t_name, [])
        for c_from, t_target, c_target in t_info.foreign_keys:
            fk_graph.setdefault(t_name, []).append((t_target, c_from, c_target))
            fk_graph.setdefault(t_target, []).append((t_name, c_target, c_from))

    # 4. Build BM25 Index for Tables
    table_names_list: List[str] = list(tables.keys())
    table_corpus: List[List[str]] = []
    for t_name in table_names_list:
        t_obj = tables[t_name]
        tokens = tokenize(t_name)
        if t_obj.description:
            tokens.extend(tokenize(t_obj.description))
        for c_name, c_obj in t_obj.columns.items():
            tokens.extend(tokenize(c_name))
            if c_obj.description:
                tokens.extend(tokenize(c_obj.description))
        table_corpus.append(tokens)
    bm25_tables = BM25Okapi(table_corpus) if table_corpus else None

    # 5. Build BM25 Index for Columns
    column_keys_list: List[Tuple[str, str]] = []
    column_corpus: List[List[str]] = []
    for t_name, t_obj in tables.items():
        for c_name, c_obj in t_obj.columns.items():
            column_keys_list.append((t_name, c_name))
            tokens = tokenize(t_name) + tokenize(c_name)
            if c_obj.description:
                tokens.extend(tokenize(c_obj.description))
            if c_obj.samples:
                for s in c_obj.samples:
                    tokens.extend(tokenize(s))
            column_corpus.append(tokens)
    bm25_columns = BM25Okapi(column_corpus) if column_corpus else None

    # 6. Build BM25 Index for Distinct Text Values (Cell Retrieval)
    value_entries: List[Tuple[str, str, str]] = []
    value_corpus: List[List[str]] = []
    for t_name, t_obj in tables.items():
        for c_name, c_obj in t_obj.columns.items():
            for val in c_obj.distinct_text_values:
                val_toks = tokenize(val)
                if val_toks:
                    value_entries.append((t_name, c_name, val))
                    value_corpus.append(val_toks)
    bm25_values = BM25Okapi(value_corpus) if value_corpus else None

    # 7. Dense Embeddings for Columns & Tables
    column_embeddings: Dict[Tuple[str, str], List[float]] = {}
    table_embeddings: Dict[str, List[float]] = {}

    if compute_embeddings and column_keys_list:
        try:
            texts_to_embed: List[str] = []
            keys_to_embed: List[Tuple[str, str]] = []

            for t_name, c_name in column_keys_list:
                c_obj = tables[t_name].columns[c_name]
                desc = f": {c_obj.description}" if c_obj.description else ""
                sample_str = f" | Samples: {c_obj.samples[:3]}" if c_obj.samples else ""
                text = f"Table `{t_name}`, Column `{c_name}` ({c_obj.data_type}){desc}{sample_str}"
                texts_to_embed.append(text)
                keys_to_embed.append((t_name, c_name))

            # Batch embed in chunks of 50
            batch_size = 50
            all_vecs: List[List[float]] = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i : i + batch_size]
                vecs = default_client.get_embedding(batch_texts)
                all_vecs.extend(vecs)

            for key, vec in zip(keys_to_embed, all_vecs):
                column_embeddings[key] = vec

            # Embed table representations
            tbl_texts = []
            for t_name in table_names_list:
                t_obj = tables[t_name]
                col_list_str = ", ".join([f"`{c}`" for c in list(t_obj.columns.keys())[:10]])
                t_desc = f": {t_obj.description}" if t_obj.description else ""
                tbl_texts.append(f"Table `{t_name}`{t_desc} with columns {col_list_str}")
            tbl_vecs = default_client.get_embedding(tbl_texts)
            for t_name, vec in zip(table_names_list, tbl_vecs):
                table_embeddings[t_name] = vec

        except Exception as e:
            print(f"Warning: Failed to compute embeddings for {db_id}: {e}")

    return DatabaseIndex(
        db_id=db_id,
        tables=tables,
        fk_graph=fk_graph,
        bm25_tables=bm25_tables,
        table_names_list=table_names_list,
        bm25_columns=bm25_columns,
        column_keys_list=column_keys_list,
        bm25_values=bm25_values,
        value_entries=value_entries,
        column_embeddings=column_embeddings,
        table_embeddings=table_embeddings,
    )


def save_index(index: DatabaseIndex, output_dir: Union[str, Path]) -> Path:
    """Serialize DatabaseIndex to pickle file."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index.db_id}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


class IndexUnpickler(pickle.Unpickler):
    """Custom unpickler to handle classes serialized under __main__."""
    def find_class(self, module, name):
        if module == "__main__" and name in ("DatabaseIndex", "TableInfo", "ColumnInfo"):
            module = "src.schema.indexer"
        return super().find_class(module, name)


def load_index(db_id: str, index_dir: Union[str, Path]) -> DatabaseIndex:
    """Load serialized DatabaseIndex from pickle file."""
    p = Path(index_dir).resolve() / f"{db_id}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Index not found for database '{db_id}' at {p}")
    with open(p, "rb") as f:
        return IndexUnpickler(f).load()


def build_all_indices(
    db_base_dir: str = "data/benchmarks/bird/dev_databases",
    tables_json: str = "data/benchmarks/bird/dev_tables.json",
    output_dir: str = "data/benchmarks/indices",
    compute_embeddings: bool = True,
    target_db_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Index all databases in db_base_dir."""
    base_dir = Path(db_base_dir).resolve()
    tables_path = Path(tables_json).resolve() if tables_json else None
    out_dir = Path(output_dir).resolve()

    if not base_dir.exists():
        raise FileNotFoundError(f"Database root not found: {base_dir}")

    db_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if target_db_id:
        db_dirs = [d for d in db_dirs if d.name == target_db_id]

    saved_indices: Dict[str, Path] = {}
    print(f"\n========================================================")
    print(f"Indexing {len(db_dirs)} Databases (Output: {out_dir})")
    print(f"Dense Embeddings: {'Enabled' if compute_embeddings else 'Disabled'}")
    print(f"========================================================\n")

    for idx, d in enumerate(sorted(db_dirs, key=lambda p: p.name), 1):
        db_id = d.name
        sqlite_file = d / f"{db_id}.sqlite"
        if not sqlite_file.exists():
            # Check for any .sqlite file in folder
            sqls = list(d.glob("*.sqlite"))
            if sqls:
                sqlite_file = sqls[0]
            else:
                print(f"[{idx}/{len(db_dirs)}] Skipping {db_id}: no .sqlite file found.")
                continue

        t0 = time.time()
        print(f"[{idx:2d}/{len(db_dirs):2d}] Indexing {db_id:26s} ...", end="", flush=True)
        try:
            db_index = extract_database_index(
                db_path=sqlite_file,
                db_id=db_id,
                tables_json_path=tables_path,
                compute_embeddings=compute_embeddings,
            )
            out_file = save_index(db_index, out_dir)
            saved_indices[db_id] = out_file
            print(f" Done ({time.time() - t0:.2f}s) -> {out_file.name}")
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\nSuccessfully indexed {len(saved_indices)} databases into {out_dir}\n")
    return saved_indices


def main():
    parser = argparse.ArgumentParser(description="Assay Schema & BM25 Indexer.")
    parser.add_argument("--benchmark", default="bird", help="Benchmark name (bird / spider)")
    parser.add_argument("--split", default="dev", help="Dataset split (dev / train)")
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/bird/dev_databases",
        help="Path to database root directory containing db folders.",
    )
    parser.add_argument(
        "--tables-json",
        default="data/benchmarks/bird/dev_tables.json",
        help="Path to tables.json containing schema metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/indices",
        help="Directory to save serialized indices.",
    )
    parser.add_argument(
        "--db-id",
        default=None,
        help="Optional single database ID to index.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing dense nomic embeddings.",
    )

    args = parser.parse_args()

    import src.schema.indexer as idx_mod
    idx_mod.build_all_indices(
        db_base_dir=args.db_dir,
        tables_json=args.tables_json,
        output_dir=args.output_dir,
        compute_embeddings=not args.skip_embeddings,
        target_db_id=args.db_id,
    )


if __name__ == "__main__":
    main()
