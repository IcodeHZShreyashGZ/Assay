"""Staged evaluation runner for Assay.

Supports executing ablation stages across BIRD and Spider benchmarks:
  Stage 0: Zero-shot baseline over raw DDL + domain evidence
  Stage 1: Compact schema representation + BM25/LCS value RAG (pending)
  Stage 2: LangGraph execution-guided self-repair (pending)
  Stage 3: QLoRA fine-tuned model evaluation (pending)
  Stage 4: AST safety guard & semantic abstention (pending)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.metrics import compute_ex, compute_r_ves, compute_soft_f1, execute_query
from eval.taxonomy import classify_error
from src.llm import default_client


def extract_raw_ddl(db_path: Union[str, Path]) -> str:
    """Extract raw CREATE TABLE / CREATE VIEW DDLs from sqlite_master."""
    abs_path = Path(db_path).resolve()
    if not abs_path.exists():
        return ""
    try:
        conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND sql IS NOT NULL ORDER BY name;"
        )
        statements = [row[0].strip() for row in cursor.fetchall() if row[0]]
        conn.close()
        return ";\n\n".join(statements) + (";" if statements else "")
    except Exception as e:
        return f"-- Error extracting DDL: {e}"


def extract_sql_from_response(raw_text: str) -> str:
    """Extract clean SQL string from LLM completion text."""
    if not raw_text:
        return ""

    # Look for ```sql ... ``` or ``` ... ```
    pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, raw_text, re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip()
    else:
        # Fallback: take lines starting with SELECT or WITH
        candidate = raw_text.strip()
        select_match = re.search(r"(SELECT|WITH)\b[\s\S]+", candidate, re.IGNORECASE)
        if select_match:
            candidate = select_match.group(0).strip()

    # Clean trailing semicolon or unwanted formatting
    candidate = re.sub(r";+\s*$", "", candidate)
    return candidate.strip()


def build_stage0_prompt(ddl: str, evidence: str, question: str) -> Tuple[str, str]:
    """Build system and user prompts for Stage 0 zero-shot baseline."""
    system_prompt = (
        "You are an expert SQL engineer. Given a database schema DDL and natural language question with domain evidence, "
        "write a single valid SQLite SQL query that answers the question accurately.\n"
        "Output ONLY the SQL query inside a ```sql``` code block. Do not provide explanations, comments, or extra text."
    )
    user_prompt = (
        f"【Database Schema DDL】\n{ddl}\n\n"
        f"【Domain Evidence】\n{evidence if evidence else 'None'}\n\n"
        f"【Question】\n{question}\n\n"
        f"Write the SQLite SQL query to answer the question:"
    )
    return system_prompt, user_prompt


def run_stage_0(
    slice_data: List[Dict[str, Any]],
    db_base_dir: str,
    log_path: str = "eval/logs/stage0_baseline.jsonl",
    result_path: str = "eval/results/stage0_baseline.json",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Stage 0 Zero-Shot Baseline evaluation."""
    total_items = len(slice_data) if limit is None else min(limit, len(slice_data))
    items_to_run = slice_data[:total_items]

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    res_file = Path(result_path)
    res_file.parent.mkdir(parents=True, exist_ok=True)

    traces: List[Dict[str, Any]] = []
    latencies: List[float] = []
    prompt_tokens_list: List[int] = []
    completion_tokens_list: List[int] = []

    diff_counts: Dict[str, Dict[str, float]] = {
        "simple": {"total": 0, "correct": 0},
        "moderate": {"total": 0, "correct": 0},
        "challenging": {"total": 0, "correct": 0},
    }
    taxonomy_counts: Counter = Counter()

    print(f"\n========================================================")
    print(f"Starting Stage 0 Baseline Evaluation ({total_items} queries)")
    print(f"Model:  {default_client.generation_model}")
    print(f"Target: {log_path}")
    print(f"========================================================\n")

    # Clear previous log file if restarting run
    with open(log_file, "w", encoding="utf-8") as f:
        pass

    start_time_all = time.time()

    for idx, item in enumerate(items_to_run, start=1):
        qid = item["question_id"]
        db_id = item["db_id"]
        question = item["question"]
        evidence = item.get("evidence", "")
        gold_sql = item["SQL"]
        difficulty = item.get("difficulty", "moderate")

        db_path = Path(db_base_dir) / db_id / f"{db_id}.sqlite"
        ddl = extract_raw_ddl(db_path)

        # 1. LLM Generation
        sys_p, usr_p = build_stage0_prompt(ddl, evidence, question)
        t_llm_start = time.perf_counter()
        raw_output, usage = default_client.generate(
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            temperature=0.0,
            num_ctx=8192,
        )
        llm_elapsed = time.perf_counter() - t_llm_start

        # 2. Extract SQL
        pred_sql = extract_sql_from_response(raw_output)

        # 3. Execute Candidate SQL (isolated worker, 5s timeout, ?mode=ro)
        pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, pred_sql, timeout_sec=5.0)

        # 4. Execute Gold SQL
        gold_ok, gold_rows, gold_elapsed, gold_err = execute_query(db_path, gold_sql, timeout_sec=5.0)

        # 5. Compute Metrics
        ex = compute_ex(pred_rows, gold_rows, gold_sql=gold_sql)
        r_ves = compute_r_ves(ex, pred_elapsed, gold_elapsed)
        soft_f1 = compute_soft_f1(pred_rows, gold_rows)

        # 6. Failure Taxonomy
        error_category = None
        if ex <= 0.0:
            error_category = classify_error(
                success=pred_ok,
                error_message=pred_err,
                pred_rows=pred_rows,
                gold_rows=gold_rows,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
            )
            taxonomy_counts[error_category] += 1

        total_query_latency = llm_elapsed + pred_elapsed
        latencies.append(total_query_latency)
        prompt_tokens_list.append(usage["prompt_tokens"])
        completion_tokens_list.append(usage["completion_tokens"])

        if difficulty in diff_counts:
            diff_counts[difficulty]["total"] += 1
            if ex == 1.0:
                diff_counts[difficulty]["correct"] += 1

        trace = {
            "question_id": qid,
            "db_id": db_id,
            "difficulty": difficulty,
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "raw_output": raw_output,
            "ex": ex,
            "r_ves": r_ves,
            "soft_f1": soft_f1,
            "error_category": error_category,
            "pred_success": pred_ok,
            "pred_error": pred_err,
            "pred_time_sec": pred_elapsed,
            "gold_time_sec": gold_elapsed,
            "llm_time_sec": llm_elapsed,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
        traces.append(trace)

        # Stream trace to JSONL
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        status_tag = "[PASS]" if ex == 1.0 else f"[FAIL: {error_category}]"
        print(f"[{idx:3d}/{total_items:3d}] QID {qid:4d} ({db_id:18s}) | EX: {ex:.0f} | Soft F1: {soft_f1:.2f} | Latency: {total_query_latency:.2f}s | {status_tag}")

    total_time_all = time.time() - start_time_all

    # Compute Summary
    num_ex = sum(t["ex"] for t in traces)
    overall_ex = (num_ex / total_items) * 100.0 if total_items > 0 else 0.0
    overall_r_ves = (sum(t["r_ves"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    overall_soft_f1 = (sum(t["soft_f1"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    avg_prompt_tokens = sum(prompt_tokens_list) / total_items if total_items > 0 else 0.0
    avg_comp_tokens = sum(completion_tokens_list) / total_items if total_items > 0 else 0.0

    sorted_lat = sorted(latencies)
    p50_lat = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0.0
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

    diff_breakdown = {}
    for d, counts in diff_counts.items():
        tot = counts["total"]
        cor = counts["correct"]
        diff_breakdown[d] = {
            "total": tot,
            "correct": cor,
            "ex_pct": (cor / tot * 100.0) if tot > 0 else 0.0,
        }

    summary = {
        "stage": 0,
        "description": "Zero-Shot Baseline (Raw DDL)",
        "model": default_client.generation_model,
        "total_queries": total_items,
        "execution_accuracy_pct": round(overall_ex, 2),
        "r_ves_pct": round(overall_r_ves, 2),
        "soft_f1_pct": round(overall_soft_f1, 2),
        "avg_prompt_tokens": round(avg_prompt_tokens, 1),
        "avg_completion_tokens": round(avg_comp_tokens, 1),
        "p50_latency_sec": round(p50_lat, 3),
        "p95_latency_sec": round(p95_lat, 3),
        "wall_time_sec": round(total_time_all, 2),
        "difficulty_breakdown": diff_breakdown,
        "taxonomy_distribution": dict(taxonomy_counts),
    }

    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 64)
    print("STAGE 0 BASELINE RESULTS SUMMARY")
    print("=" * 64)
    print(f"Total Evaluated Queries:   {total_items}")
    print(f"Model:                     {default_client.generation_model}")
    print(f"Execution Accuracy (EX):   {overall_ex:.2f}%")
    print(f"Reward-Based VES (R-VES):  {overall_r_ves:.2f}%")
    print(f"Soft F1-Score:             {overall_soft_f1:.2f}%")
    print(f"Avg Prompt Tokens:         {avg_prompt_tokens:.1f}")
    print(f"Avg Completion Tokens:     {avg_comp_tokens:.1f}")
    print(f"p50 / p95 Latency:         {p50_lat:.2f}s / {p95_lat:.2f}s")
    print(f"Total Run Duration:        {total_time_all:.1f}s")
    print("-" * 64)
    print("Difficulty Band Breakdown:")
    for d, stats in diff_breakdown.items():
        print(f"  {d.capitalize():12s}: {stats['correct']:3d} / {stats['total']:3d} ({stats['ex_pct']:.2f}%)")
    print("-" * 64)
    print("Failure Taxonomy:")
    for cat, count in taxonomy_counts.most_common():
        pct = (count / total_items) * 100.0
        print(f"  {cat:20s}: {count:3d} ({pct:.1f}%)")
    print("=" * 64 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Assay staged evaluation runner.")
    parser.add_argument(
        "--stage",
        default="0",
        help="Evaluation stage: 0 (or baseline), 1 (rag), 2 (agent), 3 (qlora), 4 (safety)",
    )
    parser.add_argument(
        "--slice",
        "--dataset",
        dest="slice_path",
        default="data/bird_dev_slice_250.json",
        help="Path to slice or dataset JSON.",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/bird/dev_databases",
        help="Path to database root directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of queries to evaluate.",
    )
    parser.add_argument(
        "--full",
        "--full-dev",
        action="store_true",
        help="Run against the full dev dataset instead of slice.",
    )

    args = parser.parse_args()

    slice_path = args.slice_path
    if args.full:
        slice_path = "data/benchmarks/bird/dev.json"

    with open(slice_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage_str = str(args.stage).lower()
    if stage_str in ["0", "baseline", "stage0"]:
        run_stage_0(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    else:
        print(f"Stage '{args.stage}' not yet implemented or unknown.")
        sys.exit(1)


if __name__ == "__main__":
    main()
