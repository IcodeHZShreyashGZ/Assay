"""LangGraph execution-guided self-repair state machine for Assay.

Architecture:
  link_schema -> generate_sql -> execute_and_check -> (valid or k>=3 ? END : refine_sql -> execute_and_check)

Features:
  - Dual Error Triggering: Runtime/syntax exceptions AND empty ([]) or all-NULL results.
  - Bounded Retries: Strict hard cap k <= 3 attempts.
  - CTE Decomposition Fallback: Triggered on attempt 2 (retry_count == 2) to break queries into modular CTEs.
  - Layer 1 Static AST Guard (sqlglot): Rejects DDL/DML, multi-statement/stacked queries before execution.
  - Layer 2 Read-Only DB Sandbox: Isolated 5s execution timeout over ?mode=ro SQLite connection.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sqlglot
from sqlglot import exp
from langgraph.graph import END, START, StateGraph

from eval.metrics import execute_query
from eval.taxonomy import classify_error
from src.agent.prompts import build_generation_prompt, build_repair_prompt, extract_sql_from_response
from src.agent.state import AssayState, QueryAttempt
from src.llm import default_client
from src.schema.linker import SchemaLinker


def validate_sql_ast(sql: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    """Static AST validation using sqlglot (Layer 1 Guard).
    
    Ensures:
      - Valid parseable SQL syntax
      - Single statement (no stacked queries or comment-injections)
      - Root statement is strictly SELECT or UNION (including CTEs)
      - No DDL/DML operations (DROP, DELETE, UPDATE, INSERT, ALTER, ATTACH, PRAGMA)
    """
    if not sql or not sql.strip():
        return False, "Candidate SQL is empty"

    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        return False, f"AST Parse Error: {e}"

    if len(parsed) != 1:
        return False, f"Multiple or stacked statements rejected ({len(parsed)} statements found)"

    stmt = parsed[0]
    if stmt is None:
        return False, "AST Root Statement is None"

    if not isinstance(stmt, (exp.Select, exp.Union)):
        return False, f"Root statement must be SELECT or UNION, got {type(stmt).__name__}"

    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Pragma,
        exp.Command,
    )
    for node in stmt.walk():
        if isinstance(node, forbidden):
            return False, f"Forbidden statement type detected: {type(node).__name__}"

    return True, None


def is_all_null(rows: List[Tuple[Any, ...]]) -> bool:
    """Check if all values in returned result rows are NULL / None."""
    if not rows:
        return True
    for row in rows:
        for val in row:
            if val is not None and str(val).strip().lower() not in ("none", "null", ""):
                return False
    return True


# ---------------------------------------------------------------------------
# Graph Node Implementations
# ---------------------------------------------------------------------------

def node_link_schema(state: AssayState) -> Dict[str, Any]:
    """Node 1: Link and compress database schema using Bidirectional RAG."""
    # If schema is already provided, keep it
    if state.get("pruned_schema"):
        return {}

    db_id = state["db_id"]
    question = state["question"]
    evidence = state.get("evidence", "")

    try:
        linker = SchemaLinker(db_id)
        linked = linker.link_and_prune(
            question=question,
            evidence=evidence,
            max_tables=4,
            max_columns_per_table=10,
            use_embeddings=True,
        )
        return {
            "pruned_schema": linked.m_schema_str,
            "selected_tables": linked.selected_tables,
        }
    except Exception as e:
        # Fallback if linker fails: minimal schema indicator
        return {
            "pruned_schema": f"-- Error retrieving schema for {db_id}: {e}",
            "selected_tables": [],
        }


def node_generate_sql(state: AssayState) -> Dict[str, Any]:
    """Node 2: Generate initial candidate SQL from linked schema and query."""
    sys_p, usr_p = build_generation_prompt(
        schema_str=state["pruned_schema"],
        question=state["question"],
        evidence=state.get("evidence", ""),
        dialect=state.get("dialect", "sqlite"),
    )

    t0 = time.perf_counter()
    raw_output, usage = default_client.generate(
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        temperature=0.0,
        num_ctx=8192,
    )
    llm_elapsed = time.perf_counter() - t0

    candidate_sql = extract_sql_from_response(raw_output)

    return {
        "candidate_sql": candidate_sql,
        "total_llm_time": state.get("total_llm_time", 0.0) + llm_elapsed,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + usage.get("prompt_tokens", 0),
        "total_completion_tokens": state.get("total_completion_tokens", 0) + usage.get("completion_tokens", 0),
    }


def node_execute_and_check(state: AssayState) -> Dict[str, Any]:
    """Node 3: Validate AST (Layer 1) and execute in read-only sandbox (Layer 2)."""
    candidate_sql = state["candidate_sql"]
    db_path = state["db_path"]
    dialect = state.get("dialect", "sqlite")
    retry_count = state.get("retry_count", 0)

    # Layer 1: Static AST Guard
    ast_valid, ast_err = validate_sql_ast(candidate_sql, dialect=dialect)
    if not ast_valid:
        attempt: QueryAttempt = {
            "attempt": retry_count,
            "sql": candidate_sql,
            "is_valid": False,
            "error_message": ast_err,
            "error_category": "syntax_error",
            "rows": None,
            "llm_time_sec": 0.0,
            "exec_time_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        attempts = list(state.get("attempts", [])) + [attempt]
        return {
            "is_valid": False,
            "error_message": ast_err,
            "error_category": "syntax_error",
            "execution_result": None,
            "attempts": attempts,
        }

    # Layer 2: Read-Only DB Sandbox (5s timeout, ?mode=ro)
    pred_ok, pred_rows, pred_elapsed, pred_err = execute_query(db_path, candidate_sql, timeout_sec=5.0)

    is_valid = False
    error_cat = None
    error_msg = None

    if not pred_ok:
        # Runtime / Syntax Exception or Timeout
        is_valid = False
        error_msg = pred_err
        error_cat = classify_error(
            success=False,
            error_message=pred_err,
            pred_rows=None,
            gold_rows=None,
            pred_sql=candidate_sql,
        )
    elif len(pred_rows) == 0:
        # Empty result set trigger
        is_valid = False
        error_msg = "Query returned 0 rows (empty result set [])."
        error_cat = "empty_result"
    elif is_all_null(pred_rows):
        # All-NULL result set trigger
        is_valid = False
        error_msg = "Query returned all NULL values."
        error_cat = "empty_result"
    else:
        # Success with valid non-empty rows
        is_valid = True
        error_msg = None
        error_cat = None

    attempt: QueryAttempt = {
        "attempt": retry_count,
        "sql": candidate_sql,
        "is_valid": is_valid,
        "error_message": error_msg,
        "error_category": error_cat,
        "rows": pred_rows if is_valid else None,
        "llm_time_sec": 0.0,
        "exec_time_sec": pred_elapsed,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    attempts = list(state.get("attempts", [])) + [attempt]

    return {
        "is_valid": is_valid,
        "error_message": error_msg,
        "error_category": error_cat,
        "execution_result": pred_rows if is_valid else None,
        "attempts": attempts,
        "total_exec_time": state.get("total_exec_time", 0.0) + pred_elapsed,
    }


def node_refine_sql(state: AssayState) -> Dict[str, Any]:
    """Node 4: Refine failed SQL with error feedback or CTE decomposition fallback."""
    current_retry = state.get("retry_count", 0) + 1

    sys_p, usr_p = build_repair_prompt(
        schema_str=state["pruned_schema"],
        question=state["question"],
        evidence=state.get("evidence", ""),
        failed_sql=state["candidate_sql"],
        error_category=state.get("error_category", "syntax_error") or "syntax_error",
        error_message=state.get("error_message", "") or "",
        retry_count=current_retry,
        dialect=state.get("dialect", "sqlite"),
    )

    t0 = time.perf_counter()
    raw_output, usage = default_client.generate(
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        temperature=0.0,
        num_ctx=8192,
    )
    llm_elapsed = time.perf_counter() - t0

    repaired_sql = extract_sql_from_response(raw_output)

    return {
        "candidate_sql": repaired_sql,
        "retry_count": current_retry,
        "total_llm_time": state.get("total_llm_time", 0.0) + llm_elapsed,
        "total_prompt_tokens": state.get("total_prompt_tokens", 0) + usage.get("prompt_tokens", 0),
        "total_completion_tokens": state.get("total_completion_tokens", 0) + usage.get("completion_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Conditional Routing Edge
# ---------------------------------------------------------------------------

def route_after_execution(state: AssayState) -> str:
    """Decide whether to exit (valid or retries exhausted) or refine SQL."""
    # Exit condition 1: Query execution succeeded with non-empty results
    if state.get("is_valid") is True:
        return "end"

    # Exit condition 2: Reached bounded retry cap (k <= 3 attempts total)
    if state.get("retry_count", 0) >= 3:
        return "end"

    # Otherwise route to self-repair node
    return "refine_sql"


# ---------------------------------------------------------------------------
# Graph Construction & Compilation
# ---------------------------------------------------------------------------

def build_sql_agent():
    """Build and compile the 4-node LangGraph execution-guided repair state machine."""
    workflow = StateGraph(AssayState)

    # Add Nodes
    workflow.add_node("link_schema", node_link_schema)
    workflow.add_node("generate_sql", node_generate_sql)
    workflow.add_node("execute_and_check", node_execute_and_check)
    workflow.add_node("refine_sql", node_refine_sql)

    # Add Edges
    workflow.add_edge(START, "link_schema")
    workflow.add_edge("link_schema", "generate_sql")
    workflow.add_edge("generate_sql", "execute_and_check")

    # Conditional Branch after execution
    workflow.add_conditional_edges(
        "execute_and_check",
        route_after_execution,
        {
            "end": END,
            "refine_sql": "refine_sql",
        },
    )

    # Loop back from refine_sql to execute_and_check
    workflow.add_edge("refine_sql", "execute_and_check")

    return workflow.compile()


# Global shared compiled agent instance
sql_agent = build_sql_agent()


def run_agent(
    question: str,
    evidence: str,
    db_id: str,
    db_path: str,
    dialect: str = "sqlite",
    pruned_schema: Optional[str] = None,
    max_retries: int = 3,
) -> AssayState:
    """Execute the closed-loop LangGraph agent for a single query."""
    initial_state: AssayState = {
        "question": question,
        "evidence": evidence,
        "db_id": db_id,
        "db_path": str(db_path),
        "dialect": dialect,
        "pruned_schema": pruned_schema or "",
        "selected_tables": [],
        "candidate_sql": "",
        "execution_result": None,
        "is_valid": False,
        "error_message": None,
        "error_category": None,
        "retry_count": 0,
        "attempts": [],
        "total_llm_time": 0.0,
        "total_exec_time": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }

    final_state = sql_agent.invoke(initial_state)
    return final_state
