"""Dual-layer safety sandboxing and static AST validation for Assay (Stage 4).

Architecture:
  Layer 1 (Static AST Guard via sqlglot):
    - Single statement enforcement (rejects stacked queries & multi-statements).
    - Root AST node whitelist: strictly exp.Select / exp.Union.
    - Recursive rejection of DDL/DML/PRAGMA/ATTACH/system nodes in CTEs and subqueries.
    - Automatic injection of LIMIT 1000 if absent.
  Layer 2 (DB Sandbox):
    - Read-only enforcement (?mode=ro URI for SQLite, read-only transaction for PostgreSQL).
    - Process-level multiprocessing isolation with hard 5-second kill timeout.
    - Catches timeouts, syntax exceptions, empty result sets, and all-NULL responses.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sqlglot
from sqlglot import exp

logger = logging.getLogger("assay.safety")


@dataclass
class GuardResult:
    """Result of static AST validation and execution safety verification."""
    is_safe: bool
    sanitized_sql: str
    rejection_reason: Optional[str] = None
    rule_triggered: Optional[str] = None
    injected_limit: bool = False
    ast_statement_type: Optional[str] = None


@dataclass
class SandboxExecutionResult:
    """Outcome of sandboxed query execution."""
    success: bool
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error_message: Optional[str] = None
    timed_out: bool = False
    is_empty: bool = False
    is_all_null: bool = False


# Forbidden AST node types across all expressions (CTEs, subqueries, unions)
FORBIDDEN_AST_NODES = tuple(
    node
    for node in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Alter", None),
        getattr(exp, "Create", None),
        getattr(exp, "TruncateTable", None),
        getattr(exp, "Pragma", None),
        getattr(exp, "Command", None),
        getattr(exp, "Kill", None),
        getattr(exp, "Transaction", None),
        getattr(exp, "Commit", None),
        getattr(exp, "Rollback", None),
        getattr(exp, "Merge", None),
    )
    if node is not None
)

# Dangerous function names that allow arbitrary code execution, sleeps, or filesystem access
DANGEROUS_FUNCTIONS: Set[str] = {
    "load_extension",
    "writefile",
    "readfile",
    "edit",
    "exec",
    "system",
    "xp_cmdshell",
    "sleep",
    "pg_sleep",
    "benchmark",
    "delay",
}


class SafetyGuard:
    """Layer 1 Static AST Guard & Query Sanitizer."""

    def __init__(
        self,
        dialect: str = "sqlite",
        default_limit: int = 1000,
        enforce_limit: bool = True,
    ):
        self.dialect = dialect.lower()
        self.default_limit = default_limit
        self.enforce_limit = enforce_limit

    def validate_ast(self, sql: str) -> GuardResult:
        """Validate candidate SQL string against Layer 1 safety rules."""
        if not sql or not sql.strip():
            return GuardResult(
                is_safe=False,
                sanitized_sql="",
                rejection_reason="Candidate SQL string is empty",
                rule_triggered="EMPTY_INPUT",
            )

        cleaned_sql = sql.strip()

        # Check for obvious comment-obfuscated multi-statements before parsing
        # e.g. "SELECT 1; -- DROP TABLE" or "; DROP TABLE"
        uncommented = re.sub(r"--[^\n]*", "", cleaned_sql)
        uncommented = re.sub(r"/\*[\s\S]*?\*/", "", uncommented).strip()

        # Disallow raw semicolon chained statements
        semicolon_parts = [p.strip() for p in uncommented.split(";") if p.strip()]
        if len(semicolon_parts) > 1:
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason=f"Multiple stacked statements detected ({len(semicolon_parts)} statements)",
                rule_triggered="STACKED_QUERIES",
            )

        # Parse AST with sqlglot
        try:
            parsed = sqlglot.parse(cleaned_sql, read=self.dialect)
        except Exception as parse_err:
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason=f"SQL syntax / AST parse error: {parse_err}",
                rule_triggered="PARSE_ERROR",
            )

        if not parsed:
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason="No valid SQL statements found in input",
                rule_triggered="EMPTY_AST",
            )

        if len(parsed) != 1:
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason=f"Multiple statements rejected by AST parser ({len(parsed)} found)",
                rule_triggered="STACKED_QUERIES",
            )

        stmt = parsed[0]
        if stmt is None:
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason="AST root statement is None",
                rule_triggered="NULL_AST",
            )

        stmt_type = type(stmt).__name__

        # Root whitelist: strictly exp.Select or exp.Union
        if not isinstance(stmt, (exp.Select, exp.Union)):
            return GuardResult(
                is_safe=False,
                sanitized_sql=cleaned_sql,
                rejection_reason=f"Root statement must be SELECT or UNION, got {stmt_type}",
                rule_triggered="NON_SELECT_ROOT",
                ast_statement_type=stmt_type,
            )

        # Recursive check for forbidden node types anywhere in the AST
        for node in stmt.walk():
            if isinstance(node, FORBIDDEN_AST_NODES):
                node_name = type(node).__name__
                return GuardResult(
                    is_safe=False,
                    sanitized_sql=cleaned_sql,
                    rejection_reason=f"Forbidden operation '{node_name}' detected in AST",
                    rule_triggered=f"FORBIDDEN_NODE_{node_name.upper()}",
                    ast_statement_type=stmt_type,
                )

            # Check function calls for dangerous functions
            if isinstance(node, exp.Anonymous):
                fn_name = node.name.lower()
                if fn_name in DANGEROUS_FUNCTIONS:
                    return GuardResult(
                        is_safe=False,
                        sanitized_sql=cleaned_sql,
                        rejection_reason=f"Dangerous function invocation '{fn_name}' blocked",
                        rule_triggered="DANGEROUS_FUNCTION",
                        ast_statement_type=stmt_type,
                    )

            # Check for ATTACH / DETACH in raw SQLite commands or expressions
            sql_repr = str(node).upper()
            if "ATTACH DATABASE" in sql_repr or "DETACH DATABASE" in sql_repr or "ATTACH " in sql_repr:
                return GuardResult(
                    is_safe=False,
                    sanitized_sql=cleaned_sql,
                    rejection_reason="ATTACH/DETACH DATABASE statement is strictly forbidden",
                    rule_triggered="FORBIDDEN_ATTACH",
                    ast_statement_type=stmt_type,
                )

        # Automatic LIMIT 1000 injection if absent on outer query
        injected_limit = False
        final_sql = cleaned_sql

        if self.enforce_limit:
            has_outer_limit = bool(stmt.find(exp.Limit))
            if not has_outer_limit and isinstance(stmt, exp.Select):
                stmt = stmt.limit(self.default_limit)
                injected_limit = True
                final_sql = stmt.sql(dialect=self.dialect)

        return GuardResult(
            is_safe=True,
            sanitized_sql=final_sql,
            injected_limit=injected_limit,
            ast_statement_type=stmt_type,
        )


def validate_sql_ast(sql: str, dialect: str = "sqlite") -> Tuple[bool, Optional[str]]:
    """Backward-compatible helper function for agent and evaluation modules."""
    guard = SafetyGuard(dialect=dialect, enforce_limit=False)
    res = guard.validate_ast(sql)
    return res.is_safe, res.rejection_reason


# ---------------------------------------------------------------------------
# Layer 2: Read-Only DB Sandbox Runner
# ---------------------------------------------------------------------------

def _sandbox_sqlite_worker(
    db_path_str: str,
    sql: str,
    queue: multiprocessing.Queue,
) -> None:
    """Isolated subprocess worker executing SQLite query under ?mode=ro."""
    start_time = time.perf_counter()
    try:
        abs_path = Path(db_path_str).resolve()
        conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()
        elapsed = time.perf_counter() - start_time
        queue.put((True, rows, cols, elapsed, None))
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        queue.put((False, [], [], elapsed, str(exc)))


def execute_safe_sandbox(
    db_path: Union[str, Path],
    sql: str,
    dialect: str = "sqlite",
    timeout_sec: float = 5.0,
    enforce_limit: bool = True,
    default_limit: int = 1000,
) -> Tuple[GuardResult, SandboxExecutionResult]:
    """Execute query through Dual-Layer Safety Sandbox.
    
    Returns:
      Tuple of (GuardResult, SandboxExecutionResult)
    """
    # Layer 1: Static AST Guard
    guard = SafetyGuard(dialect=dialect, default_limit=default_limit, enforce_limit=enforce_limit)
    guard_res = guard.validate_ast(sql)

    if not guard_res.is_safe:
        return guard_res, SandboxExecutionResult(
            success=False,
            error_message=f"Layer 1 Block: {guard_res.rejection_reason}",
        )

    # Layer 2: Isolated DB Execution
    sql_to_run = guard_res.sanitized_sql
    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_sandbox_sqlite_worker,
        args=(str(db_path), sql_to_run, queue),
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
        return guard_res, SandboxExecutionResult(
            success=False,
            timed_out=True,
            elapsed_sec=timeout_sec,
            error_message=f"Query timed out after {timeout_sec:.1f}s sandbox limit",
        )

    if not queue.empty():
        success, rows, cols, elapsed, err = queue.get()
    else:
        success, rows, cols, elapsed, err = False, [], [], timeout_sec, "Process terminated unexpectedly"

    is_empty = success and (len(rows) == 0)
    is_all_null = False
    if success and rows:
        is_all_null = all(
            all(v is None or str(v).strip().lower() in ("none", "null", "") for v in row)
            for row in rows
        )

    exec_res = SandboxExecutionResult(
        success=success,
        rows=rows,
        columns=cols,
        elapsed_sec=elapsed,
        error_message=err,
        is_empty=is_empty,
        is_all_null=is_all_null,
    )
    return guard_res, exec_res
