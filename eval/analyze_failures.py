"""Detailed failure analysis of Stage 0 baseline evaluation traces.

Categorizes and analyzes the 127 failures to inform Stage 1 indexer and linker weights:
- empty_result: ungrounded literals in WHERE clauses
- schema_missing: hallucinated or misnamed tables/columns
- syntax_error / type_or_div_error: SQLite syntax quirks and integer division
- logic_mismatch: join path errors, incorrect aggregates, missed evidence
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_stage0_failures(log_path: str = "eval/logs/stage0_baseline.jsonl"):
    with open(log_path, "r", encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]

    failures = [t for t in traces if t["ex"] == 0.0]
    total_traces = len(traces)
    total_failures = len(failures)

    print("=" * 72)
    print(f"STAGE 0 FAILURE TAXONOMY AUDIT (Total Misses: {total_failures} / {total_traces})")
    print("=" * 72)

    cat_counts = Counter(t["error_category"] for t in failures)
    print("\n[Failure Category Distribution]")
    for cat, cnt in cat_counts.most_common():
        pct_of_fails = (cnt / total_failures) * 100
        pct_of_total = (cnt / total_traces) * 100
        print(f"  {cat:20s}: {cnt:3d} ({pct_of_fails:5.1f}% of misses | {pct_of_total:4.1f}% of all queries)")

    # Failure rate per database
    db_fails = Counter(t["db_id"] for t in failures)
    db_totals = Counter(t["db_id"] for t in traces)
    print("\n[Failure Distribution by Database]")
    for db, fail_cnt in db_fails.most_common():
        tot = db_totals[db]
        fail_pct = (fail_cnt / tot) * 100
        print(f"  {db:26s}: {fail_cnt:2d}/{tot:2d} failed ({fail_pct:5.1f}%)")

    # 1. EMPTY RESULT ANALYSIS (22 queries)
    print("\n" + "=" * 72)
    print(f"1. DEEP DIVE: EMPTY RESULT ({cat_counts['empty_result']} queries)")
    print("   Root Cause: Ungrounded WHERE literals, casing/spacing mismatches,")
    print("   missing distinct non-null sample values or LCS cell retrieval.")
    print("=" * 72)
    empty_samples = [t for t in failures if t["error_category"] == "empty_result"]
    for i, t in enumerate(empty_samples, 1):
        qid = t["question_id"]
        db = t["db_id"]
        q = t["question"]
        ev = t.get("evidence", "")
        pred_sql = t["pred_sql"].replace("\n", " ")
        gold_sql = t["gold_sql"].replace("\n", " ")
        print(f"\n({i:2d}) QID {qid} [{db}]")
        print(f"    Question: {q}")
        if ev:
            print(f"    Evidence: {ev}")
        print(f"    Pred SQL: {pred_sql}")
        print(f"    Gold SQL: {gold_sql}")

    # 2. SCHEMA MISSING ANALYSIS (24 queries)
    print("\n" + "=" * 72)
    print(f"2. DEEP DIVE: SCHEMA MISSING ({cat_counts['schema_missing']} queries)")
    print("   Root Cause: Hallucinated column/table names due to attention dilution")
    print("   over raw DDL dumps; lack of compact M-Schema representation.")
    print("=" * 72)
    schema_samples = [t for t in failures if t["error_category"] == "schema_missing"]
    for i, t in enumerate(schema_samples, 1):
        qid = t["question_id"]
        db = t["db_id"]
        err = t.get("pred_error", "")
        pred_sql = t["pred_sql"].replace("\n", " ")
        gold_sql = t["gold_sql"].replace("\n", " ")
        print(f"\n({i:2d}) QID {qid} [{db}]")
        print(f"    Error   : {err}")
        print(f"    Pred SQL: {pred_sql}")
        print(f"    Gold SQL: {gold_sql}")

    # 3. SYNTAX & TYPE/DIV ERRORS (3 queries)
    print("\n" + "=" * 72)
    print("3. DEEP DIVE: SYNTAX & TYPE/DIV ERRORS (3 queries)")
    print("=" * 72)
    syntax_samples = [t for t in failures if t["error_category"] in ("syntax_error", "type_or_div_error")]
    for i, t in enumerate(syntax_samples, 1):
        qid = t["question_id"]
        db = t["db_id"]
        cat = t["error_category"]
        err = t.get("pred_error", "")
        pred_sql = t["pred_sql"].replace("\n", " ")
        gold_sql = t["gold_sql"].replace("\n", " ")
        print(f"\n({i:2d}) QID {qid} [{db}] - [{cat}]")
        print(f"    Error   : {err}")
        print(f"    Pred SQL: {pred_sql}")
        print(f"    Gold SQL: {gold_sql}")

    # 4. LOGIC MISMATCH SAMPLES (78 queries)
    print("\n" + "=" * 72)
    print(f"4. OVERVIEW: LOGIC MISMATCH ({cat_counts['logic_mismatch']} queries)")
    print("   Root Cause: Join path selection (missing FK bridge tables),")
    print("   inverted boolean logic, and wrong aggregation/grouping.")
    print("=" * 72)
    logic_samples = [t for t in failures if t["error_category"] == "logic_mismatch"]
    for i, t in enumerate(logic_samples[:10], 1):
        qid = t["question_id"]
        db = t["db_id"]
        diff = t["difficulty"]
        q = t["question"]
        pred_sql = t["pred_sql"].replace("\n", " ")
        gold_sql = t["gold_sql"].replace("\n", " ")
        print(f"\n({i:2d}) QID {qid} [{db} | {diff}]")
        print(f"    Question: {q}")
        print(f"    Pred SQL: {pred_sql}")
        print(f"    Gold SQL: {gold_sql}")


if __name__ == "__main__":
    analyze_stage0_failures()
