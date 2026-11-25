"""Evaluate Schema Linking Recall (SLR) on BIRD dev slice.

Measures:
  - Table Recall: % of tables in gold SQL retained by SchemaLinker.
  - Column Recall: % of columns in gold SQL retained by SchemaLinker.
Target: >= 97% Table Recall.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlglot
from sqlglot import exp

from src.schema.linker import SchemaLinker


def extract_gold_schema_elements(sql: str) -> Tuple[Set[str], Set[str]]:
    """Extract table names and column names from gold SQL query using sqlglot."""
    tables: Set[str] = set()
    columns: Set[str] = set()
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        for tbl in parsed.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name.strip().strip("`\"'[]").lower())
        for col in parsed.find_all(exp.Column):
            if col.name and col.name != "*":
                columns.add(col.name.strip().strip("`\"'[]").lower())
    except Exception:
        # Fallback regex extraction if parsing fails
        import re
        tbl_matches = re.findall(r"\bFROM\s+([A-Za-z0-9_]+)|\bJOIN\s+([A-Za-z0-9_]+)", sql, re.IGNORECASE)
        for m in tbl_matches:
            name = m[0] or m[1]
            if name:
                tables.add(name.lower())
    return tables, columns


def evaluate_schema_recall(
    slice_path: str = "data/bird_dev_slice_250.json",
    index_dir: str = "data/benchmarks/indices",
    limit: Optional[int] = None,
    use_embeddings: bool = True,
) -> Dict[str, Any]:
    with open(slice_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    total_queries = len(data)
    table_recalls: List[float] = []
    column_recalls: List[float] = []
    perfect_table_matches = 0
    perfect_column_matches = 0

    linkers: Dict[str, SchemaLinker] = {}

    print(f"\nEvaluating Schema Linking Recall across {total_queries} queries...")

    for idx, item in enumerate(data, 1):
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence", "")
        gold_sql = item["SQL"]

        if db_id not in linkers:
            linkers[db_id] = SchemaLinker(db_id, index_dir=index_dir)
        linker = linkers[db_id]

        gold_tables, gold_cols = extract_gold_schema_elements(gold_sql)

        # Run linking & pruning
        linked = linker.link_and_prune(question, evidence, use_embeddings=use_embeddings)

        pruned_tables = {t.lower() for t in linked.selected_tables}
        pruned_cols: Set[str] = set()
        for t, cols in linked.selected_columns.items():
            for c in cols:
                pruned_cols.add(c.lower())

        # Table Recall
        if gold_tables:
            matched_tables = gold_tables.intersection(pruned_tables)
            t_recall = len(matched_tables) / len(gold_tables)
        else:
            t_recall = 1.0
        table_recalls.append(t_recall)
        if t_recall >= 1.0:
            perfect_table_matches += 1

        # Column Recall
        if gold_cols:
            matched_cols = gold_cols.intersection(pruned_cols)
            c_recall = len(matched_cols) / len(gold_cols)
        else:
            c_recall = 1.0
        column_recalls.append(c_recall)
        if c_recall >= 1.0:
            perfect_column_matches += 1

        if idx % 50 == 0 or idx == total_queries:
            avg_tr = sum(table_recalls) / len(table_recalls) * 100.0
            avg_cr = sum(column_recalls) / len(column_recalls) * 100.0
            print(f"[{idx:3d}/{total_queries:3d}] Running Table Recall: {avg_tr:.2f}% | Column Recall: {avg_cr:.2f}%")

    avg_table_recall = (sum(table_recalls) / total_queries) * 100.0
    avg_col_recall = (sum(column_recalls) / total_queries) * 100.0
    perfect_t_pct = (perfect_table_matches / total_queries) * 100.0

    print("\n" + "=" * 60)
    print("SCHEMA LINKING RECALL RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Queries:   {total_queries}")
    print(f"Average Table Recall:      {avg_table_recall:.2f}% (Target: >= 97.0%)")
    print(f"Average Column Recall:     {avg_col_recall:.2f}%")
    print(f"Queries with 100% Tables:  {perfect_table_matches}/{total_queries} ({perfect_t_pct:.1f}%)")
    print("=" * 60 + "\n")

    return {
        "total_queries": total_queries,
        "avg_table_recall": round(avg_table_recall, 2),
        "avg_col_recall": round(avg_col_recall, 2),
        "target_table_recall_pct": 97.0,
        "perfect_table_matches": perfect_table_matches,
        "perfect_table_matches_pct": round(perfect_t_pct, 2),
        "use_embeddings": use_embeddings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assay Schema Linking Recall evaluator.")
    parser.add_argument(
        "--slice",
        default="data/bird_dev_slice_250.json",
        help="Path to the evaluation slice JSON.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/benchmarks/indices",
        help="Directory containing serialized schema indices.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of queries.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable dense retrieval (useful when Ollama is offline).",
    )
    parser.add_argument(
        "--output",
        default="eval/results/stage1_recall.json",
        help="Path to persist the recall summary JSON.",
    )
    args = parser.parse_args()

    result = evaluate_schema_recall(
        slice_path=args.slice,
        index_dir=args.index_dir,
        limit=args.limit,
        use_embeddings=not args.no_embeddings,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved recall summary to: {output_path}")
