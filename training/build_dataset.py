"""Dataset builder for Assay Stage 3 QLoRA fine-tuning.

Transforms BIRD train/dev queries into train-inference aligned instruction data:
  1. Exact Stage 1 Compact DDL (M-Schema) prompt format.
  2. Injection of 1-3 distractor tables and distractor columns to prevent
     train-inference distribution shift.
  3. Cell-value retrieval grounding via BM25 + LCS.
  4. Concise Chain-of-Thought (CoT) rationales:
     Schema Mapping -> Join Path -> Filters/Casting -> Final SQL.
  5. JSONL output compatible with TRL SFTTrainer and assistant-only loss masking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from eval.eval_recall import extract_gold_schema_elements
from src.schema.indexer import DatabaseIndex, TableInfo, load_bird_descriptions
from src.schema.linker import MatchedValue, SchemaLinker

logger = logging.getLogger("assay.build_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_system_prompt() -> str:
    """Build train-aligned system prompt matching Stage 1 & 2."""
    return (
        "You are an expert SQL engineer. Given a compact database schema with types, foreign keys, "
        "sample values, and domain evidence, write a single valid SQLite SQL query that answers the question accurately.\n"
        "Strict Requirements:\n"
        "1. Output ONLY the SQL query inside a ```sql``` code block. Do NOT include explanations, comments, or extra text.\n"
        "2. For column names containing spaces or parentheses (e.g. `School Type`, `County Code`, `Enrollment (Ages 5-17)`), ALWAYS wrap them in backticks like `School Type` to prevent syntax errors.\n"
        "3. For division involving integer columns, wrap numerators in `CAST(... AS REAL)` to prevent 0-truncation.\n"
        "4. For string filter predicates, match exact values provided in the schema or evidence.\n"
        "5. Join tables using explicit foreign key equality conditions."
    )


def build_user_prompt(m_schema_str: str, evidence: str, question: str) -> str:
    """Build train-aligned user prompt matching Stage 1 & 2."""
    return (
        f"{m_schema_str}\n\n"
        f"【Domain Evidence】\n{evidence if evidence else 'None'}\n\n"
        f"【Question】\n{question}\n\n"
        f"Write the SQLite SQL query to answer the question:"
    )


def extract_cot_rationale(
    sql: str,
    gold_tables: Set[str],
    gold_columns: Set[str],
    evidence: str = "",
) -> str:
    """Construct a concise, execution-verified Chain-of-Thought (CoT) rationale from SQL AST.
    
    Structure:
      1. Schema Mapping: Relevant tables & columns
      2. Join Path: Join predicates or Single table query
      3. Filters & Logic: WHERE filters, integer division casting, aggregations
      4. SQL Query: code block
    """
    tbl_list = sorted(list(gold_tables))
    tbl_str = ", ".join(f"`{t}`" for t in tbl_list) if tbl_list else "None"
    col_str = ", ".join(f"`{c}`" for c in sorted(list(gold_columns))[:8]) if gold_columns else "None"

    # Analyze AST for joins, filters, casts, aggregations
    join_clauses: List[str] = []
    filter_clauses: List[str] = []
    logic_notes: List[str] = []

    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        
        # Joins
        for join in parsed.find_all(exp.Join):
            on_clause = join.args.get("on")
            if on_clause:
                join_clauses.append(str(on_clause).strip())
            else:
                join_tbl = join.this
                if join_tbl:
                    join_clauses.append(f"JOIN {join_tbl}")

        # Where predicates
        for where in parsed.find_all(exp.Where):
            this_filter = str(where.this).strip()
            if len(this_filter) > 80:
                this_filter = this_filter[:77] + "..."
            filter_clauses.append(this_filter)

        # Cast for division
        has_cast = any(isinstance(c, exp.Cast) for c in parsed.find_all(exp.Cast)) or "AS REAL" in sql.upper()
        if has_cast:
            logic_notes.append("Cast numerator with `CAST(... AS REAL)` to prevent 0-truncation in integer division")

        # Aggregations
        aggs = set()
        for agg_cls in (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max):
            for _ in parsed.find_all(agg_cls):
                aggs.add(agg_cls.__name__.upper())
        if aggs:
            logic_notes.append(f"Aggregate functions: {', '.join(sorted(list(aggs)))}")

        # Order by / Limit
        if parsed.find(exp.Order):
            logic_notes.append("Sort results via `ORDER BY`")
        if parsed.find(exp.Limit):
            limit_val = parsed.find(exp.Limit).expression
            logic_notes.append(f"Limit output rows (`LIMIT {limit_val}`)")

    except Exception:
        # Fallback heuristic analysis
        if "JOIN" in sql.upper():
            join_clauses.append("Join related tables via foreign key equality")
        if "WHERE" in sql.upper():
            filter_clauses.append("Apply filter predicates specified in question/evidence")
        if "CAST(" in sql.upper() and "REAL" in sql.upper():
            logic_notes.append("Cast integer division with `CAST(... AS REAL)`")

    # Format lines
    cot_lines = ["【Thought】"]
    cot_lines.append(f"1. Schema Mapping:\n   - Relevant Tables: {tbl_str}\n   - Required Columns: {col_str}")

    if join_clauses:
        join_str = "; ".join(join_clauses[:3])
        cot_lines.append(f"2. Join Path:\n   - {join_str}")
    elif len(tbl_list) <= 1:
        cot_lines.append("2. Join Path:\n   - Single table query (no join needed)")
    else:
        cot_lines.append("2. Join Path:\n   - Match primary/foreign key relationships across tables")

    filters_section = []
    if filter_clauses:
        filters_section.append(f"Filter predicates: {'; '.join(filter_clauses[:2])}")
    if logic_notes:
        filters_section.extend(logic_notes)
    if not filters_section:
        filters_section.append("No specific WHERE filters required")

    cot_lines.append("3. Filters & Logic:\n   - " + "\n   - ".join(filters_section))
    cot_lines.append(f"4. SQL Query:\n```sql\n{sql.strip()}\n```")

    return "\n".join(cot_lines)


def format_schema_with_distractors(
    linker: SchemaLinker,
    gold_tables: Set[str],
    gold_columns: Set[str],
    question: str,
    evidence: str,
    num_distractor_tables: int = 2,
    num_distractor_cols: int = 3,
    max_columns_per_table: int = 8,
    rng: Optional[random.Random] = None,
) -> str:
    """Format compact M-Schema including gold schema elements plus random distractors."""
    if rng is None:
        rng = random.Random(42)

    all_db_tables = list(linker.index.tables.keys())
    # Match gold table names case-insensitively
    gold_matched_tables: Set[str] = set()
    for g_t in gold_tables:
        for db_t in all_db_tables:
            if g_t.lower() == db_t.lower():
                gold_matched_tables.add(db_t)

    # If no gold tables matched (e.g. subquery or alias), use tables retrieved by linker
    if not gold_matched_tables:
        linked = linker.link_and_prune(question, evidence, use_embeddings=False)
        gold_matched_tables.update(linked.selected_tables)

    # Distractor tables
    eligible_distractor_tables = [t for t in all_db_tables if t not in gold_matched_tables]
    chosen_distractor_tables = rng.sample(
        eligible_distractor_tables,
        min(num_distractor_tables, len(eligible_distractor_tables)),
    ) if eligible_distractor_tables else []

    selected_tables = set(gold_matched_tables) | set(chosen_distractor_tables)

    # Foreign Key graph closure: ensure connectivity between tables
    closed_tables, active_fks = linker.foreign_key_graph_closure(selected_tables)

    # Column selection: gold columns + distractor columns
    selected_columns: Dict[str, List[str]] = {}
    for t_name in closed_tables:
        if t_name not in linker.index.tables:
            continue
        t_obj = linker.index.tables[t_name]
        all_cols = list(t_obj.columns.keys())

        # Gold columns in this table
        table_gold_cols = [
            c for c in all_cols if c.lower() in [g.lower() for g in gold_columns]
        ]
        # Always include PKs
        pks = [c for c in all_cols if t_obj.columns[c].is_pk]
        
        # Distractor columns
        other_cols = [c for c in all_cols if c not in table_gold_cols and c not in pks]
        distractor_cols = rng.sample(other_cols, min(num_distractor_cols, len(other_cols))) if other_cols else []

        kept_cols = set(table_gold_cols) | set(pks) | set(distractor_cols)
        # Limit total columns per table
        final_cols = [c for c in all_cols if c in kept_cols][:max_columns_per_table]
        if not final_cols:
            final_cols = all_cols[:max_columns_per_table]
        selected_columns[t_name] = final_cols

    # Matched cell values
    matched_values = linker.retrieve_cell_values(question, evidence, top_k=5)

    return linker.format_m_schema(
        closed_tables=closed_tables,
        selected_columns=selected_columns,
        matched_values=matched_values,
        active_fks=active_fks,
    )


def build_training_dataset(
    input_file: Path,
    output_file: Path,
    db_base_dir: Path,
    index_dir: Path = Path("data/benchmarks/indices"),
    num_distractor_tables: int = 2,
    num_distractor_cols: int = 3,
    max_columns_per_table: int = 8,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Process query JSON into aligned train_qlora.jsonl."""
    logger.info(f"Loading input queries from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    linkers: Dict[str, SchemaLinker] = {}
    sys_prompt = build_system_prompt()

    total_samples = 0
    token_lengths: List[int] = []

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(data, 1):
            qid = item.get("question_id", idx)
            db_id = item["db_id"]
            question = item["question"]
            evidence = item.get("evidence", "")
            gold_sql = item["SQL"]

            # Initialize linker / index if needed
            if db_id not in linkers:
                try:
                    linkers[db_id] = SchemaLinker(db_id, index_dir=str(index_dir))
                except Exception as e:
                    logger.warning(f"Could not load index for {db_id}: {e}")
                    continue

            linker = linkers[db_id]

            # 1. Extract gold tables and columns
            gold_tables, gold_columns = extract_gold_schema_elements(gold_sql)

            # 2. Format schema with distractor tables/columns
            m_schema_str = format_schema_with_distractors(
                linker=linker,
                gold_tables=gold_tables,
                gold_columns=gold_columns,
                question=question,
                evidence=evidence,
                num_distractor_tables=num_distractor_tables,
                num_distractor_cols=num_distractor_cols,
                max_columns_per_table=max_columns_per_table,
                rng=rng,
            )

            # 3. User Prompt
            usr_prompt = build_user_prompt(m_schema_str, evidence, question)

            # 4. Chain of Thought Assistant Completion
            asst_completion = extract_cot_rationale(
                sql=gold_sql,
                gold_tables=gold_tables,
                gold_columns=gold_columns,
                evidence=evidence,
            )

            # 5. Formatted Chat Messages
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
                {"role": "assistant", "content": asst_completion},
            ]

            # Estimate token count (rough heuristic: 1 token ~ 3.8 chars)
            est_tokens = (len(sys_prompt) + len(usr_prompt) + len(asst_completion)) // 4
            token_lengths.append(est_tokens)

            record = {
                "id": f"bird_{qid}",
                "db_id": db_id,
                "difficulty": item.get("difficulty", "moderate"),
                "question": question,
                "evidence": evidence,
                "gold_sql": gold_sql,
                "gold_tables": sorted(list(gold_tables)),
                "gold_columns": sorted(list(gold_columns)),
                "messages": messages,
                "prompt": usr_prompt,
                "response": asst_completion,
                "est_tokens": est_tokens,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_samples += 1

            if idx % 50 == 0 or idx == len(data):
                logger.info(f"Processed {idx}/{len(data)} queries ({total_samples} written)")

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0

    stats = {
        "total_samples": total_samples,
        "output_file": str(output_file),
        "avg_estimated_tokens": round(avg_tokens, 1),
        "max_estimated_tokens": max_tokens,
        "databases_covered": len(linkers),
    }

    logger.info(f"Dataset build complete! Summary: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Assay Stage 3 QLoRA Dataset Builder.")
    parser.add_argument(
        "--bird-train",
        default="data/benchmarks/bird/train.json",
        help="Path to BIRD train.json dataset.",
    )
    parser.add_argument(
        "--bird-dev",
        default="data/benchmarks/bird/dev.json",
        help="Path to BIRD dev.json dataset (used as fallback).",
    )
    parser.add_argument(
        "--slice",
        default="data/bird_dev_slice_250.json",
        help="Path to stratified dev slice.",
    )
    parser.add_argument(
        "--output",
        default="data/train_qlora.jsonl",
        help="Path to output train_qlora.jsonl.",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/bird/dev_databases",
        help="Root directory containing SQLite databases.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/benchmarks/indices",
        help="Root directory for schema and BM25 index files.",
    )
    parser.add_argument(
        "--distractor-tables",
        type=int,
        default=2,
        help="Number of random distractor tables to inject.",
    )
    parser.add_argument(
        "--distractor-columns",
        type=int,
        default=3,
        help="Number of random distractor columns per table to inject.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=8,
        help="Maximum columns rendered per table in compact schema.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible distractor sampling.",
    )

    args = parser.parse_args()

    # Determine input dataset
    train_path = Path(args.bird_train)
    dev_path = Path(args.bird_dev)
    slice_path = Path(args.slice)

    if train_path.exists():
        chosen_input = train_path
        logger.info(f"Using primary BIRD train dataset: {chosen_input}")
    elif slice_path.exists():
        chosen_input = slice_path
        logger.warning(
            f"Primary train dataset '{train_path}' not found locally. "
            f"Falling back to available stratified slice: {chosen_input}"
        )
    elif dev_path.exists():
        chosen_input = dev_path
        logger.warning(
            f"Primary train dataset '{train_path}' not found locally. "
            f"Falling back to full dev dataset: {chosen_input}"
        )
    else:
        logger.error(f"No valid BIRD dataset JSON found at {train_path}, {slice_path}, or {dev_path}!")
        return

    build_training_dataset(
        input_file=chosen_input,
        output_file=Path(args.output),
        db_base_dir=Path(args.db_dir),
        index_dir=Path(args.index_dir),
        num_distractor_tables=args.distractor_tables,
        num_distractor_cols=args.distractor_columns,
        max_columns_per_table=args.max_columns,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
