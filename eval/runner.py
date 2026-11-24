"""Staged evaluation runner for Assay.

Supports executing ablation stages across BIRD and Spider benchmarks:
  Stage 0: Zero-shot baseline over raw DDL + domain evidence
  Stage 1: Compact schema representation + BM25/LCS value RAG
  Stage 2: LangGraph execution-guided self-repair
  Stage 3: QLoRA fine-tuned model evaluation
  Stage 4: AST safety guard & semantic abstention
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
from typing import Any, Dict, List, Optional, Tuple, Union

from eval.metrics import compute_ex, compute_r_ves, compute_soft_f1, execute_query
from eval.taxonomy import classify_error
from src.agent.graph import run_agent, validate_sql_ast
from src.api.abstention import SemanticAbstentionGate
from src.llm import default_client
from src.safety.guard import SafetyGuard, execute_safe_sandbox
from src.schema.linker import SchemaLinker


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


def build_stage1_prompt(m_schema_str: str, evidence: str, question: str) -> Tuple[str, str]:
    """Build system and user prompts for Stage 1 Schema Linking & Value RAG."""
    system_prompt = (
        "You are an expert SQL engineer. Given a compact database schema with types, foreign keys, "
        "sample values, and domain evidence, write a single valid SQLite SQL query that answers the question accurately.\n"
        "Strict Requirements:\n"
        "1. Output ONLY the SQL query inside a ```sql``` code block. Do NOT include explanations, comments, or extra text.\n"
        "2. For column names containing spaces or parentheses (e.g. `School Type`, `County Code`, `Enrollment (Ages 5-17)`), ALWAYS wrap them in backticks like `School Type` to prevent syntax errors.\n"
        "3. For division involving integer columns, wrap numerators in `CAST(... AS REAL)` to prevent 0-truncation.\n"
        "4. For string filter predicates, match exact values provided in the schema or evidence.\n"
        "5. Join tables using explicit foreign key equality conditions."
    )
    user_prompt = (
        f"{m_schema_str}\n\n"
        f"【Domain Evidence】\n{evidence if evidence else 'None'}\n\n"
        f"【Question】\n{question}\n\n"
        f"Write the SQLite SQL query to answer the question:"
    )
    return system_prompt, user_prompt


def run_stage_1(
    slice_data: List[Dict[str, Any]],
    db_base_dir: str,
    index_dir: str = "data/benchmarks/indices",
    log_path: str = "eval/logs/stage1_rag.jsonl",
    result_path: str = "eval/results/stage1_rag.json",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Stage 1 Schema Linking & Value RAG evaluation."""
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
    linkers: Dict[str, SchemaLinker] = {}

    print(f"\n========================================================")
    print(f"Starting Stage 1 Schema Linking & Value RAG ({total_items} queries)")
    print(f"Model:   {default_client.generation_model}")
    print(f"Indices: {index_dir}")
    print(f"Target:  {log_path}")
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

        # 1. Schema Linking & Value RAG
        if db_id not in linkers:
            linkers[db_id] = SchemaLinker(db_id, index_dir=index_dir)
        linker = linkers[db_id]

        t_rag_start = time.perf_counter()
        linked = linker.link_and_prune(question, evidence, use_embeddings=True)
        rag_elapsed = time.perf_counter() - t_rag_start

        # 2. LLM Generation
        sys_p, usr_p = build_stage1_prompt(linked.m_schema_str, evidence, question)
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

        # 3. Extract SQL
        pred_sql = extract_sql_from_response(raw_output)

        # 4. Execute Candidate SQL (isolated worker, 5s timeout, ?mode=ro)
        pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, pred_sql, timeout_sec=5.0)

        # 5. Execute Gold SQL
        gold_ok, gold_rows, gold_elapsed, gold_err = execute_query(db_path, gold_sql, timeout_sec=5.0)

        # 6. Compute Metrics
        ex = compute_ex(pred_rows, gold_rows, gold_sql=gold_sql)
        r_ves = compute_r_ves(ex, pred_elapsed, gold_elapsed)
        soft_f1 = compute_soft_f1(pred_rows, gold_rows)

        # 7. Failure Taxonomy
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

        total_query_latency = rag_elapsed + llm_elapsed + pred_elapsed
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
            "selected_tables": linked.selected_tables,
            "matched_values_count": len(linked.matched_values),
            "ex": ex,
            "r_ves": r_ves,
            "soft_f1": soft_f1,
            "error_category": error_category,
            "pred_success": pred_ok,
            "pred_error": pred_err,
            "rag_time_sec": rag_elapsed,
            "llm_time_sec": llm_elapsed,
            "pred_time_sec": pred_elapsed,
            "gold_time_sec": gold_elapsed,
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
        "stage": 1,
        "description": "Schema Linking & Value RAG (M-Schema)",
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
    print("STAGE 1 SCHEMA LINKING & VALUE RAG RESULTS SUMMARY")
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


def run_stage_2(
    slice_data: List[Dict[str, Any]],
    db_base_dir: str,
    index_dir: str = "data/benchmarks/indices",
    log_path: str = "eval/logs/stage2_agent.jsonl",
    result_path: str = "eval/results/stage2_agent.json",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Stage 2 LangGraph Execution-Guided Self-Repair evaluation."""
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
    attempts_list: List[int] = []

    diff_counts: Dict[str, Dict[str, float]] = {
        "simple": {"total": 0, "correct": 0},
        "moderate": {"total": 0, "correct": 0},
        "challenging": {"total": 0, "correct": 0},
    }
    taxonomy_counts: Counter = Counter()
    linkers: Dict[str, SchemaLinker] = {}

    total_queries_needing_repair = 0
    repaired_success_count = 0

    print(f"\n========================================================")
    print(f"Starting Stage 2 LangGraph Self-Repair Agent ({total_items} queries)")
    print(f"Model:   {default_client.generation_model}")
    print(f"Max Retries (k): 3 (with attempt 2 CTE fallback)")
    print(f"Target:  {log_path}")
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

        # 1. Run Schema Linking
        if db_id not in linkers:
            linkers[db_id] = SchemaLinker(db_id, index_dir=index_dir)
        linker = linkers[db_id]

        t_rag_start = time.perf_counter()
        linked = linker.link_and_prune(question, evidence, use_embeddings=True)
        rag_elapsed = time.perf_counter() - t_rag_start

        # 2. Run LangGraph Self-Repair Agent
        t_agent_start = time.perf_counter()
        final_state = run_agent(
            question=question,
            evidence=evidence,
            db_id=db_id,
            db_path=str(db_path),
            dialect="sqlite",
            pruned_schema=linked.m_schema_str,
            max_retries=3,
        )
        agent_elapsed = time.perf_counter() - t_agent_start

        pred_sql = final_state.get("candidate_sql", "")
        attempts_count = len(final_state.get("attempts", []))
        retries_used = final_state.get("retry_count", 0)
        attempts_list.append(attempts_count)

        # 3. Execute Candidate SQL against DB
        pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, pred_sql, timeout_sec=5.0)

        # 4. Execute Gold SQL
        gold_ok, gold_rows, gold_elapsed, gold_err = execute_query(db_path, gold_sql, timeout_sec=5.0)

        # 5. Compute Metrics
        ex = compute_ex(pred_rows, gold_rows, gold_sql=gold_sql)
        r_ves = compute_r_ves(ex, pred_elapsed, gold_elapsed)
        soft_f1 = compute_soft_f1(pred_rows, gold_rows)

        # 6. Repair Recovery Tracking
        needed_repair = (attempts_count > 1)
        repaired = False
        if needed_repair:
            total_queries_needing_repair += 1
            if ex == 1.0:
                repaired = True
                repaired_success_count += 1

        # 7. Failure Taxonomy
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

        total_query_latency = rag_elapsed + agent_elapsed
        latencies.append(total_query_latency)
        prompt_tokens_list.append(final_state.get("total_prompt_tokens", 0))
        completion_tokens_list.append(final_state.get("total_completion_tokens", 0))

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
            "selected_tables": linked.selected_tables,
            "attempts_count": attempts_count,
            "retries_used": retries_used,
            "needed_repair": needed_repair,
            "repaired_success": repaired,
            "ex": ex,
            "r_ves": r_ves,
            "soft_f1": soft_f1,
            "error_category": error_category,
            "pred_success": pred_ok,
            "pred_error": pred_err,
            "rag_time_sec": rag_elapsed,
            "agent_time_sec": agent_elapsed,
            "pred_time_sec": pred_elapsed,
            "gold_time_sec": gold_elapsed,
            "prompt_tokens": final_state.get("total_prompt_tokens", 0),
            "completion_tokens": final_state.get("total_completion_tokens", 0),
            "total_tokens": final_state.get("total_prompt_tokens", 0) + final_state.get("total_completion_tokens", 0),
            "attempts_history": final_state.get("attempts", []),
        }
        traces.append(trace)

        # Stream trace to JSONL
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        status_tag = f"[PASS ({attempts_count} att)]" if ex == 1.0 else f"[FAIL ({attempts_count} att): {error_category}]"
        print(f"[{idx:3d}/{total_items:3d}] QID {qid:4d} ({db_id:18s}) | EX: {ex:.0f} | Retries: {retries_used} | Latency: {total_query_latency:.2f}s | {status_tag}")

    total_time_all = time.time() - start_time_all

    # Compute Summary
    num_ex = sum(t["ex"] for t in traces)
    overall_ex = (num_ex / total_items) * 100.0 if total_items > 0 else 0.0
    overall_r_ves = (sum(t["r_ves"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    overall_soft_f1 = (sum(t["soft_f1"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    avg_prompt_tokens = sum(prompt_tokens_list) / total_items if total_items > 0 else 0.0
    avg_comp_tokens = sum(completion_tokens_list) / total_items if total_items > 0 else 0.0
    avg_attempts = sum(attempts_list) / total_items if total_items > 0 else 1.0
    repair_recovery_rate = (repaired_success_count / total_queries_needing_repair * 100.0) if total_queries_needing_repair > 0 else 0.0

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
        "stage": 2,
        "description": "LangGraph Execution-Guided Self-Repair (k <= 3, CTE Fallback)",
        "model": default_client.generation_model,
        "total_queries": total_items,
        "execution_accuracy_pct": round(overall_ex, 2),
        "r_ves_pct": round(overall_r_ves, 2),
        "soft_f1_pct": round(overall_soft_f1, 2),
        "repair_recovery_rate_pct": round(repair_recovery_rate, 2),
        "queries_needing_repair": total_queries_needing_repair,
        "queries_repaired_successfully": repaired_success_count,
        "avg_attempts": round(avg_attempts, 2),
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
    print("STAGE 2 LANGGRAPH SELF-REPAIR AGENT RESULTS SUMMARY")
    print("=" * 64)
    print(f"Total Evaluated Queries:   {total_items}")
    print(f"Model:                     {default_client.generation_model}")
    print(f"Execution Accuracy (EX):   {overall_ex:.2f}%")
    print(f"Reward-Based VES (R-VES):  {overall_r_ves:.2f}%")
    print(f"Soft F1-Score:             {overall_soft_f1:.2f}%")
    print(f"Repair Recovery Rate:      {repair_recovery_rate:.2f}% ({repaired_success_count}/{total_queries_needing_repair})")
    print(f"Average Attempts:          {avg_attempts:.2f}")
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


def run_stage_3(
    slice_data: List[Dict[str, Any]],
    db_base_dir: str,
    index_dir: str = "data/benchmarks/indices",
    log_path: str = "eval/logs/stage3_qlora.jsonl",
    result_path: str = "eval/results/stage3_qlora.json",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Stage 3 QLoRA Fine-Tuned Model evaluation using LangGraph Self-Repair."""
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
    attempts_list: List[int] = []

    diff_counts: Dict[str, Dict[str, float]] = {
        "simple": {"total": 0, "correct": 0},
        "moderate": {"total": 0, "correct": 0},
        "challenging": {"total": 0, "correct": 0},
    }
    taxonomy_counts: Counter = Counter()
    linkers: Dict[str, SchemaLinker] = {}

    total_queries_needing_repair = 0
    repaired_success_count = 0

    print(f"\n========================================================")
    print(f"Starting Stage 3 QLoRA Fine-Tuned Evaluation ({total_items} queries)")
    print(f"Model:   {default_client.generation_model}")
    print(f"Agent:   LangGraph Execution-Guided Self-Repair (k <= 3, CTE Fallback)")
    print(f"Target:  {log_path}")
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

        # 1. Run Schema Linking
        if db_id not in linkers:
            linkers[db_id] = SchemaLinker(db_id, index_dir=index_dir)
        linker = linkers[db_id]

        t_rag_start = time.perf_counter()
        linked = linker.link_and_prune(question, evidence, use_embeddings=True)
        rag_elapsed = time.perf_counter() - t_rag_start

        # 2. Run LangGraph Self-Repair Agent with Fine-Tuned Model
        t_agent_start = time.perf_counter()
        final_state = run_agent(
            question=question,
            evidence=evidence,
            db_id=db_id,
            db_path=str(db_path),
            dialect="sqlite",
            pruned_schema=linked.m_schema_str,
            max_retries=3,
        )
        agent_elapsed = time.perf_counter() - t_agent_start

        pred_sql = final_state.get("candidate_sql", "")
        attempts_count = len(final_state.get("attempts", []))
        retries_used = final_state.get("retry_count", 0)
        attempts_list.append(attempts_count)

        # 3. Execute Candidate SQL against DB
        pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, pred_sql, timeout_sec=5.0)

        # 4. Execute Gold SQL
        gold_ok, gold_rows, gold_elapsed, gold_err = execute_query(db_path, gold_sql, timeout_sec=5.0)

        # 5. Compute Metrics
        ex = compute_ex(pred_rows, gold_rows, gold_sql=gold_sql)
        r_ves = compute_r_ves(ex, pred_elapsed, gold_elapsed)
        soft_f1 = compute_soft_f1(pred_rows, gold_rows)

        # 6. Repair Recovery Tracking
        needed_repair = (attempts_count > 1)
        repaired = False
        if needed_repair:
            total_queries_needing_repair += 1
            if ex == 1.0:
                repaired = True
                repaired_success_count += 1

        # 7. Failure Taxonomy
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

        total_query_latency = rag_elapsed + agent_elapsed
        latencies.append(total_query_latency)
        prompt_tokens_list.append(final_state.get("total_prompt_tokens", 0))
        completion_tokens_list.append(final_state.get("total_completion_tokens", 0))

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
            "selected_tables": linked.selected_tables,
            "attempts_count": attempts_count,
            "retries_used": retries_used,
            "needed_repair": needed_repair,
            "repaired_success": repaired,
            "ex": ex,
            "r_ves": r_ves,
            "soft_f1": soft_f1,
            "error_category": error_category,
            "pred_success": pred_ok,
            "pred_error": pred_err,
            "rag_time_sec": rag_elapsed,
            "agent_time_sec": agent_elapsed,
            "pred_time_sec": pred_elapsed,
            "gold_time_sec": gold_elapsed,
            "prompt_tokens": final_state.get("total_prompt_tokens", 0),
            "completion_tokens": final_state.get("total_completion_tokens", 0),
            "total_tokens": final_state.get("total_prompt_tokens", 0) + final_state.get("total_completion_tokens", 0),
            "attempts_history": final_state.get("attempts", []),
        }
        traces.append(trace)

        # Stream trace to JSONL
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        status_tag = f"[PASS ({attempts_count} att)]" if ex == 1.0 else f"[FAIL ({attempts_count} att): {error_category}]"
        print(f"[{idx:3d}/{total_items:3d}] QID {qid:4d} ({db_id:18s}) | EX: {ex:.0f} | Retries: {retries_used} | Latency: {total_query_latency:.2f}s | {status_tag}")

    total_time_all = time.time() - start_time_all

    # Compute Summary
    num_ex = sum(t["ex"] for t in traces)
    overall_ex = (num_ex / total_items) * 100.0 if total_items > 0 else 0.0
    overall_r_ves = (sum(t["r_ves"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    overall_soft_f1 = (sum(t["soft_f1"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    avg_prompt_tokens = sum(prompt_tokens_list) / total_items if total_items > 0 else 0.0
    avg_comp_tokens = sum(completion_tokens_list) / total_items if total_items > 0 else 0.0
    avg_attempts = sum(attempts_list) / total_items if total_items > 0 else 1.0
    repair_recovery_rate = (repaired_success_count / total_queries_needing_repair * 100.0) if total_queries_needing_repair > 0 else 0.0

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
        "stage": 3,
        "description": "QLoRA Fine-Tuned Model + LangGraph Self-Repair",
        "model": default_client.generation_model,
        "total_queries": total_items,
        "execution_accuracy_pct": round(overall_ex, 2),
        "r_ves_pct": round(overall_r_ves, 2),
        "soft_f1_pct": round(overall_soft_f1, 2),
        "repair_recovery_rate_pct": round(repair_recovery_rate, 2),
        "queries_needing_repair": total_queries_needing_repair,
        "queries_repaired_successfully": repaired_success_count,
        "avg_attempts": round(avg_attempts, 2),
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
    print("STAGE 3 QLORA FINE-TUNED AGENT RESULTS SUMMARY")
    print("=" * 64)
    print(f"Total Evaluated Queries:   {total_items}")
    print(f"Model:                     {default_client.generation_model}")
    print(f"Execution Accuracy (EX):   {overall_ex:.2f}%")
    print(f"Reward-Based VES (R-VES):  {overall_r_ves:.2f}%")
    print(f"Soft F1-Score:             {overall_soft_f1:.2f}%")
    print(f"Repair Recovery Rate:      {repair_recovery_rate:.2f}% ({repaired_success_count}/{total_queries_needing_repair})")
    print(f"Average Attempts:          {avg_attempts:.2f}")
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


def run_stage_4_adversarial(
    dataset_path: str = "data/adversarial_50.json",
    log_path: str = "eval/logs/stage4_safety.jsonl",
    result_path: str = "eval/results/stage4_safety.json",
) -> Dict[str, Any]:
    """Execute Stage 4 Adversarial Safety Audit across 50 attack & abstention cases."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    res_file = Path(result_path)
    res_file.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total_cases = len(cases)
    guard = SafetyGuard(dialect="sqlite", enforce_limit=True)
    gate = SemanticAbstentionGate()

    category_stats: Dict[str, Dict[str, int]] = {
        "direct_dml": {"total": 0, "defended": 0, "passed": 0},
        "stacked_injection": {"total": 0, "defended": 0, "passed": 0},
        "metadata_attach": {"total": 0, "defended": 0, "passed": 0},
        "abstention_metric": {"total": 0, "defended": 0, "passed": 0},
        "abstention_ambiguity": {"total": 0, "defended": 0, "passed": 0},
    }

    traces = []
    print("\n========================================================")
    print(f"Starting Stage 4 Adversarial Safety Audit ({total_cases} cases)")
    print(f"Dataset: {dataset_path}")
    print(f"Target:  {log_path}")
    print("========================================================\n")

    # Clear previous log
    with open(log_file, "w", encoding="utf-8") as f:
        pass

    start_time = time.time()
    defended_count = 0

    for idx, item in enumerate(cases, 1):
        cid = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        desc = item.get("description", "")
        exp_action = item.get("expected_action", "block_ast")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "defended": 0, "passed": 0}

        category_stats[cat]["total"] += 1
        defended = False
        action_taken = ""
        detail = ""

        if "abstain" in cat or "abstention" in cat or exp_action == "abstain":
            decision = gate.evaluate_query(prompt)
            if decision.should_abstain:
                defended = True
                category_stats[cat]["defended"] += 1
                action_taken = f"ABSTAINED ({decision.reason})"
                detail = decision.clarification_message or ""
            else:
                category_stats[cat]["passed"] += 1
                action_taken = "PASSED_UNABSTAINED"
        else:
            guard_res = guard.validate_ast(prompt)
            if not guard_res.is_safe:
                defended = True
                category_stats[cat]["defended"] += 1
                action_taken = f"BLOCKED_AST ({guard_res.rule_triggered})"
                detail = guard_res.rejection_reason or ""
            else:
                category_stats[cat]["passed"] += 1
                action_taken = "PASSED_AST"

        if defended:
            defended_count += 1
            status_tag = f"[DEFENDED: {action_taken}]"
        else:
            status_tag = f"[FAILED_DEFENSE: {action_taken}]"

        trace = {
            "id": cid,
            "category": cat,
            "prompt": prompt,
            "description": desc,
            "expected_action": exp_action,
            "defended": defended,
            "action_taken": action_taken,
            "detail": detail,
        }
        traces.append(trace)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        print(f"[{idx:2d}/{total_cases:2d}] Case #{cid:2d} ({cat:20s}) | {status_tag}")

    total_time = time.time() - start_time
    overall_block_rate = (defended_count / total_cases * 100.0) if total_cases > 0 else 0.0

    summary = {
        "stage": 4,
        "description": "Adversarial Safety & Abstention Audit",
        "total_test_cases": total_cases,
        "overall_block_rate_pct": round(overall_block_rate, 2),
        "defended_count": defended_count,
        "failed_count": total_cases - defended_count,
        "wall_time_sec": round(total_time, 2),
        "category_breakdown": category_stats,
    }

    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Formatted Table matching README.md Table 4
    print("\n" + "=" * 74)
    print("STAGE 4 ADVERSARIAL SAFETY AUDIT RESULTS")
    print("=" * 74)
    print(f"Total Test Cases:            {total_cases}")
    print(f"Overall Defense Rate:        {overall_block_rate:.2f}% ({defended_count}/{total_cases})")
    print(f"Duration:                    {total_time:.2f}s")
    print("-" * 74)
    print(f"{'Attack Category':<35} | {'Cases':<6} | {'Blocked / Abstained':<20} | {'Rate (%)':<8}")
    print("-" * 74)

    cat_labels = {
        "direct_dml": "Direct & Comment-Obfuscated DDL/DML",
        "stacked_injection": "Stacked Queries & Injection",
        "metadata_attach": "Engine Metadata / PRAGMA / ATTACH",
        "abstention_metric": "Undefined Corporate Metrics",
        "abstention_ambiguity": "Ambiguous Business Terms",
    }

    for c_key, c_label in cat_labels.items():
        if c_key in category_stats:
            stats = category_stats[c_key]
            tot = stats["total"]
            defended_cat = stats["defended"]
            rate = (defended_cat / tot * 100.0) if tot > 0 else 0.0
            print(f"{c_label:<35} | {tot:<6d} | {defended_cat:<20d} | {rate:6.1f}%")

    print("=" * 74 + "\n")
    return summary


def run_stage_4(
    slice_data: List[Dict[str, Any]],
    db_base_dir: str,
    index_dir: str = "data/benchmarks/indices",
    log_path: str = "eval/logs/stage4_regression.jsonl",
    result_path: str = "eval/results/stage4_regression.json",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Stage 4 Regression with AST Guard, Abstention, and LIMIT 1000."""
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
    attempts_list: List[int] = []

    diff_counts: Dict[str, Dict[str, float]] = {
        "simple": {"total": 0, "correct": 0},
        "moderate": {"total": 0, "correct": 0},
        "challenging": {"total": 0, "correct": 0},
    }
    taxonomy_counts: Counter = Counter()
    linkers: Dict[str, SchemaLinker] = {}
    guard = SafetyGuard(dialect="sqlite", default_limit=1000, enforce_limit=True)
    gate = SemanticAbstentionGate()

    total_queries_needing_repair = 0
    repaired_success_count = 0
    injected_limit_count = 0

    print(f"\n========================================================")
    print(f"Starting Stage 4 Safety Regression Evaluation ({total_items} queries)")
    print(f"Model:   {default_client.generation_model}")
    print(f"Safety:  AST Guard + LIMIT 1000 + Semantic Abstention Gate")
    print(f"Target:  {log_path}")
    print(f"========================================================\n")

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

        # 1. Semantic Abstention Gate
        abstention_dec = gate.evaluate_query(question, evidence)
        if abstention_dec.injected_evidence:
            evidence = f"{evidence}\n{abstention_dec.injected_evidence}".strip()

        # 2. Schema Linking
        if db_id not in linkers:
            linkers[db_id] = SchemaLinker(db_id, index_dir=index_dir)
        linker = linkers[db_id]

        t_rag_start = time.perf_counter()
        linked = linker.link_and_prune(question, evidence, use_embeddings=True)
        rag_elapsed = time.perf_counter() - t_rag_start

        # 3. LangGraph Agent
        t_agent_start = time.perf_counter()
        final_state = run_agent(
            question=question,
            evidence=evidence,
            db_id=db_id,
            db_path=str(db_path),
            dialect="sqlite",
            pruned_schema=linked.m_schema_str,
            max_retries=3,
        )
        agent_elapsed = time.perf_counter() - t_agent_start

        raw_pred_sql = final_state.get("candidate_sql", "")
        attempts_count = len(final_state.get("attempts", []))
        retries_used = final_state.get("retry_count", 0)
        attempts_list.append(attempts_count)

        # 4. AST Guard & LIMIT 1000 Enforcement
        guard_res = guard.validate_ast(raw_pred_sql)
        if guard_res.injected_limit:
            injected_limit_count += 1
        pred_sql = guard_res.sanitized_sql if guard_res.is_safe else raw_pred_sql

        # 5. DB Sandbox Execution
        if guard_res.is_safe:
            pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, pred_sql, timeout_sec=5.0)
        else:
            pred_ok, pred_rows, pred_elapsed, pred_err = False, [], 0.0, f"Blocked by AST Guard: {guard_res.rejection_reason}"

        # 6. Execute Gold SQL
        gold_ok, gold_rows, gold_elapsed, gold_err = execute_query(db_path, gold_sql, timeout_sec=5.0)

        # 7. Compute Metrics
        ex = compute_ex(pred_rows, gold_rows, gold_sql=gold_sql)
        r_ves = compute_r_ves(ex, pred_elapsed, gold_elapsed)
        soft_f1 = compute_soft_f1(pred_rows, gold_rows)

        needed_repair = (attempts_count > 1)
        repaired = False
        if needed_repair:
            total_queries_needing_repair += 1
            if ex == 1.0:
                repaired = True
                repaired_success_count += 1

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

        total_query_latency = rag_elapsed + agent_elapsed
        latencies.append(total_query_latency)
        prompt_tokens_list.append(final_state.get("total_prompt_tokens", 0))
        completion_tokens_list.append(final_state.get("total_completion_tokens", 0))

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
            "guard_safe": guard_res.is_safe,
            "injected_limit": guard_res.injected_limit,
            "attempts_count": attempts_count,
            "retries_used": retries_used,
            "ex": ex,
            "r_ves": r_ves,
            "soft_f1": soft_f1,
            "error_category": error_category,
            "pred_success": pred_ok,
            "pred_error": pred_err,
            "latency_sec": total_query_latency,
            "prompt_tokens": final_state.get("total_prompt_tokens", 0),
            "completion_tokens": final_state.get("total_completion_tokens", 0),
        }
        traces.append(trace)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        status_tag = f"[PASS ({attempts_count} att)]" if ex == 1.0 else f"[FAIL ({attempts_count} att): {error_category}]"
        print(f"[{idx:3d}/{total_items:3d}] QID {qid:4d} ({db_id:18s}) | EX: {ex:.0f} | Retries: {retries_used} | Latency: {total_query_latency:.2f}s | {status_tag}")

    total_time_all = time.time() - start_time_all

    num_ex = sum(t["ex"] for t in traces)
    overall_ex = (num_ex / total_items) * 100.0 if total_items > 0 else 0.0
    overall_r_ves = (sum(t["r_ves"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    overall_soft_f1 = (sum(t["soft_f1"] for t in traces) / total_items) * 100.0 if total_items > 0 else 0.0
    avg_prompt_tokens = sum(prompt_tokens_list) / total_items if total_items > 0 else 0.0
    avg_comp_tokens = sum(completion_tokens_list) / total_items if total_items > 0 else 0.0
    avg_attempts = sum(attempts_list) / total_items if total_items > 0 else 1.0
    repair_recovery_rate = (repaired_success_count / total_queries_needing_repair * 100.0) if total_queries_needing_repair > 0 else 0.0

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
        "stage": 4,
        "description": "Deterministic AST Safety, Abstention & LIMIT 1000 Guard",
        "model": default_client.generation_model,
        "total_queries": total_items,
        "execution_accuracy_pct": round(overall_ex, 2),
        "r_ves_pct": round(overall_r_ves, 2),
        "soft_f1_pct": round(overall_soft_f1, 2),
        "repair_recovery_rate_pct": round(repair_recovery_rate, 2),
        "injected_limit_count": injected_limit_count,
        "avg_attempts": round(avg_attempts, 2),
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

    print("\n" + "=" * 64)
    print("STAGE 4 SAFETY REGRESSION RESULTS SUMMARY")
    print("=" * 64)
    print(f"Total Evaluated Queries:   {total_items}")
    print(f"Model:                     {default_client.generation_model}")
    print(f"Execution Accuracy (EX):   {overall_ex:.2f}%")
    print(f"Reward-Based VES (R-VES):  {overall_r_ves:.2f}%")
    print(f"Soft F1-Score:             {overall_soft_f1:.2f}%")
    print(f"Queries with LIMIT 1000:   {injected_limit_count}")
    print(f"Repair Recovery Rate:      {repair_recovery_rate:.2f}%")
    print(f"p50 / p95 Latency:         {p50_lat:.2f}s / {p95_lat:.2f}s")
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
        "--adversarial",
        dest="adversarial_path",
        default=None,
        help="Path to adversarial safety suite JSON.",
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

    stage_str = str(args.stage).lower()

    if stage_str in ["4", "safety", "stage4"] and (args.adversarial_path or "adversarial" in args.slice_path.lower()):
        target_adv = args.adversarial_path or args.slice_path
        run_stage_4_adversarial(dataset_path=target_adv)
        return

    slice_path = args.slice_path
    if args.full:
        slice_path = "data/benchmarks/bird/dev.json"

    with open(slice_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if stage_str in ["0", "baseline", "stage0"]:
        run_stage_0(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    elif stage_str in ["1", "rag", "stage1"]:
        run_stage_1(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    elif stage_str in ["2", "agent", "stage2"]:
        run_stage_2(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    elif stage_str in ["3", "qlora", "stage3"]:
        run_stage_3(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    elif stage_str in ["4", "safety", "stage4"]:
        run_stage_4(
            slice_data=data,
            db_base_dir=args.db_dir,
            limit=args.limit,
        )
    else:
        print(f"Stage '{args.stage}' not yet implemented or unknown.")
        sys.exit(1)


if __name__ == "__main__":
    main()
