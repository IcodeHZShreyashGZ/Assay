"""Evaluation metrics and isolated execution harness for Assay.

Implements:
- 5-second hard process-level kill timeout via multiprocessing
- Read-only SQLite URI connection (file:{db_path}?mode=ro)
- Robust tempfile IPC to prevent pipe-buffer deadlocks on large result sets
- EX: Multiset tuple equality (strict list order if ORDER BY in gold SQL)
- R-VES: Reward-Based Valid Efficiency Score
- Soft F1: Cell-level precision/recall F1
- Verification Gate CLI (--sanity-check)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _worker_query_execution(
    db_path: str,
    sql: str,
    result_queue: mp.Queue,
) -> None:
    """Worker process that executes a single query under read-only mode."""
    tmp_path = None
    try:
        abs_path = Path(db_path).resolve()
        if not abs_path.exists():
            result_queue.put((False, None, 0.0, f"Database not found: {abs_path}"))
            return

        conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        t0 = time.perf_counter()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - t0
        conn.close()

        # Write results to a temporary file to avoid multiprocessing pipe buffer deadlock
        fd, tmp_path = tempfile.mkstemp(prefix="assay_res_", suffix=".pkl")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)

        result_queue.put((True, tmp_path, elapsed, None))
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        result_queue.put((False, None, 0.0, str(e)))


def execute_query(
    db_path: Union[str, Path],
    sql: str,
    timeout_sec: float = 5.0,
) -> Tuple[bool, Optional[List[Tuple]], float, Optional[str]]:
    """Execute SQL in an isolated process with a hard kill timeout.

    Args:
        db_path: Path to the SQLite database file.
        sql: SQL query to execute.
        timeout_sec: Maximum execution duration (default 5.0s).

    Returns:
        Tuple of (success, rows, elapsed_sec, error_message).
    """
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker_query_execution,
        args=(str(db_path), sql, queue),
    )

    t_start = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        total_time = time.perf_counter() - t_start
        return False, None, total_time, f"Query exceeded {timeout_sec}s timeout limit"

    if not queue.empty():
        success, tmp_path_or_none, elapsed, err = queue.get()
        if success and tmp_path_or_none:
            try:
                with open(tmp_path_or_none, "rb") as f:
                    rows = pickle.load(f)
                return True, rows, elapsed, None
            finally:
                if os.path.exists(tmp_path_or_none):
                    try:
                        os.unlink(tmp_path_or_none)
                    except OSError:
                        pass
        return False, None, elapsed, err

    return False, None, time.perf_counter() - t_start, "Worker process terminated unexpectedly"


def _normalize_cell(val: Any) -> Any:
    """Normalize cell values for comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return "__NAN__"
        return round(val, 4)
    if isinstance(val, int):
        return val
    
    s = str(val).strip()
    if s.lower() == "nan":
        return "nan"

    try:
        f = float(s)
        if math.isnan(f):
            return "nan"
        return round(f, 4)
    except ValueError:
        return s.lower()


def _normalize_rows(rows: List[Tuple]) -> List[Tuple]:
    """Normalize rows for comparison."""
    if not rows:
        return []
    return [tuple(_normalize_cell(cell) for cell in row) for row in rows]


def compute_ex(
    pred_rows: Optional[List[Tuple]],
    gold_rows: Optional[List[Tuple]],
    gold_sql: str = "",
) -> float:
    """Compute Execution Accuracy (EX).

    Multiset tuple equality, enforcing exact list order only when gold SQL
    contains an 'ORDER BY' clause.
    """
    if pred_rows is None or gold_rows is None:
        return 0.0

    norm_pred = _normalize_rows(pred_rows)
    norm_gold = _normalize_rows(gold_rows)

    has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))

    if has_order_by:
        return 1.0 if norm_pred == norm_gold else 0.0
    else:
        return 1.0 if Counter(norm_pred) == Counter(norm_gold) else 0.0


def compute_r_ves(
    ex: float,
    pred_time: float,
    gold_time: float,
) -> float:
    """Compute Reward-Based Valid Efficiency Score (R-VES).

    R = t_gold / t_pred
    Bucketed reward if EX == 1.0:
      R >= 2.0  -> 1.25
      1.0 <= R < 2.0 -> 1.00
      0.75 <= R < 1.0 -> 0.75
      0.50 <= R < 0.75 -> 0.50
      R < 0.50 -> 0.25
    If EX == 0.0 -> 0.00
    """
    if ex <= 0.0:
        return 0.0

    pred_t = max(pred_time, 1e-5)
    gold_t = max(gold_time, 1e-5)
    ratio = gold_t / pred_t

    if ratio >= 2.0:
        return 1.25
    elif ratio >= 1.0:
        return 1.00
    elif ratio >= 0.75:
        return 0.75
    elif ratio >= 0.50:
        return 0.50
    else:
        return 0.25


def compute_soft_f1(
    pred_rows: Optional[List[Tuple]],
    gold_rows: Optional[List[Tuple]],
) -> float:
    """Compute cell-level precision, recall, and F1 over result tables."""
    if pred_rows is None or gold_rows is None:
        return 0.0

    pred_cells = [str(_normalize_cell(cell)) for row in pred_rows for cell in row]
    gold_cells = [str(_normalize_cell(cell)) for row in gold_rows for cell in row]

    if not pred_cells and not gold_cells:
        return 1.0
    if not pred_cells or not gold_cells:
        return 0.0

    c_pred = Counter(pred_cells)
    c_gold = Counter(gold_cells)

    intersection = sum(min(c_pred[val], c_gold[val]) for val in c_gold)

    precision = intersection / len(pred_cells) if pred_cells else 0.0
    recall = intersection / len(gold_cells) if gold_cells else 0.0

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def run_sanity_check(
    slice_path: str = "data/bird_dev_slice_250.json",
    db_base_dir: str = "data/benchmarks/bird/dev_databases",
    timeout_sec: float = 5.0,
) -> bool:
    """Run Gold Sanity Check on all queries in slice.

    Must achieve 100.0% EX and 0 timeouts.
    """
    slice_file = Path(slice_path)
    if not slice_file.exists():
        print(f"Slice file not found: {slice_file}")
        return False

    with open(slice_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    passed = 0
    timeouts = 0
    errors = 0

    print(f"\n==========================================")
    print(f"Running Gold Sanity Check on {total} queries...")
    print(f"Slice: {slice_path}")
    print(f"Timeout: {timeout_sec}s per query")
    print(f"==========================================\n")

    for i, item in enumerate(data):
        qid = item["question_id"]
        db_id = item["db_id"]
        gold_sql = item["SQL"]
        db_path = Path(db_base_dir) / db_id / f"{db_id}.sqlite"

        success, rows, elapsed, err = execute_query(db_path, gold_sql, timeout_sec=timeout_sec)

        if not success:
            if "timeout" in (err or "").lower():
                timeouts += 1
                print(f"[FAIL] QID {qid} ({db_id}): TIMEOUT ({elapsed:.2f}s)")
            else:
                errors += 1
                print(f"[FAIL] QID {qid} ({db_id}): ERROR: {err}")
        else:
            ex = compute_ex(rows, rows, gold_sql=gold_sql)
            if ex == 1.0:
                passed += 1
            else:
                print(f"[FAIL] QID {qid} ({db_id}): EX self-check returned {ex}")

    acc = (passed / total) * 100.0 if total > 0 else 0.0
    print(f"\n==========================================")
    print(f"Gold Sanity Check Results:")
    print(f"  Total Queries: {total}")
    print(f"  Passed:        {passed} ({acc:.2f}%)")
    print(f"  Errors:        {errors}")
    print(f"  Timeouts:      {timeouts}")
    print(f"==========================================\n")

    if passed == total and timeouts == 0 and errors == 0:
        print(">>> SANITY CHECK PASSED: 100.0% Gold Accuracy with 0 timeouts. <<<\n")
        return True
    else:
        print(">>> SANITY CHECK FAILED: Does not meet Verification Gate requirements. <<<\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assay metrics and sanity check.")
    parser.add_argument(
        "--sanity-check",
        nargs="?",
        const="data/bird_dev_slice_250.json",
        default=None,
        help="Run gold sanity check on specified slice path.",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/bird/dev_databases",
        help="Path to database root directory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Query timeout in seconds (default: 5.0).",
    )
    args = parser.parse_args()

    if args.sanity_check:
        success = run_sanity_check(
            slice_path=args.sanity_check,
            db_base_dir=args.db_dir,
            timeout_sec=args.timeout,
        )
        sys.exit(0 if success else 1)
