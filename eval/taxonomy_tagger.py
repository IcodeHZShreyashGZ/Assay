"""Taxonomy Tagger CLI for Assay.

Classifies failed Text-to-SQL evaluation traces into 6 standardized failure buckets:
1. Schema Linking Errors (schema_missing): Missing tables/columns, bridge table omission.
2. Business Logic & Evidence (logic_mismatch): Semantic mismatch, inverted boolean filters.
3. SQL Functions & Casting (type_or_div_error): Truncation from integer division, aggregate errors.
4. Relational Join Paths (join_error): Incorrect joins, Cartesian products.
5. Dirty Values & Predicates (empty_result): Case/whitespace mismatches, missing IS NOT NULL.
6. Benchmark Annotation Noise / Syntax (syntax_error / timeout): Ambiguity or syntax violations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.taxonomy import classify_error


TAXONOMY_DISPLAY_NAMES = {
    "schema_missing": "1. Schema Linking Errors",
    "logic_mismatch": "2. Business Logic & Evidence",
    "type_or_div_error": "3. SQL Functions & Casting",
    "join_error": "4. Relational Join Paths",
    "empty_result": "5. Dirty Values & Predicates",
    "syntax_error": "6. Syntax / Benchmark Annotation",
    "timeout": "6. Timeout / Resource Cap",
}


def analyze_log_file(
    log_path: Path,
    sample_size: int = 100,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse JSONL log traces and compute failure taxonomy statistics."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log trace file not found: {log_path}")

    traces: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_queries = len(traces)
    correct_queries = sum(1 for t in traces if t.get("ex", 0.0) == 1.0)
    failed_traces = [t for t in traces if t.get("ex", 0.0) < 1.0]
    total_failures = len(failed_traces)

    category_counts: Counter = Counter()
    for t in failed_traces:
        cat = t.get("error_category")
        if not cat:
            cat = classify_error(
                success=t.get("pred_success", False),
                error_message=t.get("pred_error"),
                pred_rows=None,
                gold_rows=None,
                pred_sql=t.get("pred_sql", ""),
                gold_sql=t.get("gold_sql", ""),
            )
        category_counts[cat] += 1

    # Sample failures for qualitative audit
    sampled_failures = failed_traces[:sample_size]
    sample_cat_counts = Counter(t.get("error_category", "logic_mismatch") for t in sampled_failures)

    results = {
        "log_path": str(log_path),
        "total_queries": total_queries,
        "correct_queries": correct_queries,
        "failed_queries": total_failures,
        "execution_accuracy_pct": round((correct_queries / total_queries * 100.0), 2) if total_queries else 0.0,
        "category_counts": dict(category_counts),
        "sampled_size": len(sampled_failures),
        "sample_breakdown_pct": {
            TAXONOMY_DISPLAY_NAMES.get(k, k): round((v / len(sampled_failures) * 100.0), 1)
            for k, v in sample_cat_counts.items()
        } if sampled_failures else {},
    }

    print("\n" + "=" * 70)
    print(f"ASSAY FAILURE TAXONOMY REPORT: {log_path.name}")
    print(f"Total Queries: {total_queries} | Failures: {total_failures} | EX: {results['execution_accuracy_pct']}%")
    print("=" * 70)
    print("\n[6-Bucket Failure Distribution]")
    for cat, count in category_counts.most_common():
        pct_of_fails = (count / total_failures * 100.0) if total_failures else 0.0
        disp_name = TAXONOMY_DISPLAY_NAMES.get(cat, cat)
        print(f"  {disp_name:36s}: {count:3d} ({pct_of_fails:5.1f}% of failures)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved taxonomy analysis to: {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Assay 6-Bucket Failure Taxonomy Classifier & Tagger")
    parser.add_argument(
        "--log",
        "-l",
        dest="log_path",
        default="eval/logs/stage2_agent.jsonl",
        help="Path to JSONL log trace file (default: eval/logs/stage2_agent.jsonl)",
    )
    parser.add_argument(
        "--sample",
        "-s",
        type=int,
        default=100,
        help="Number of failure samples to evaluate for percentage share (default: 100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        default=None,
        help="Optional path to output JSON summary",
    )
    args = parser.parse_args()

    analyze_log_file(
        log_path=Path(args.log_path),
        sample_size=args.sample,
        output_path=Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()
