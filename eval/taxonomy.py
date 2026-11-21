"""Failure taxonomy classifier for Text-to-SQL errors.

Classifies SQL generation and execution failures into 6 standard categories:
1. timeout: Query execution exceeded the hard 5.0-second limit.
2. schema_missing: Missing table or column referenced in SQL.
3. syntax_error: SQLite syntax error or invalid SQL construct.
4. empty_result: Query returned empty result set while gold returned rows.
5. type_or_div_error: Division by zero, type mismatch, or integer division truncation.
6. logic_mismatch: Query succeeded and returned rows, but result did not match gold.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple


def classify_error(
    success: bool,
    error_message: Optional[str],
    pred_rows: Optional[List[Tuple]],
    gold_rows: Optional[List[Tuple]],
    pred_sql: str = "",
    gold_sql: str = "",
) -> str:
    """Classify the failure into one of 6 taxonomy buckets."""
    err = (error_message or "").lower()

    # 1. Timeout
    if "timeout" in err or "time out" in err or "exceeded" in err:
        return "timeout"

    # 2. Schema missing
    if any(phrase in err for phrase in [
        "no such column",
        "no such table",
        "has no column named",
        "ambiguous column name",
    ]):
        return "schema_missing"

    # 3. Syntax error / runtime SQL exception
    if not success:
        if any(phrase in err for phrase in [
            "division by zero",
            "datatype mismatch",
            "conversion error",
        ]):
            return "type_or_div_error"
        return "syntax_error"

    # Query executed successfully from here on (success == True)
    # 4. Empty result
    if pred_rows is not None and len(pred_rows) == 0:
        if gold_rows is not None and len(gold_rows) > 0:
            return "empty_result"

    # 5. Type or division truncation error
    # Check for integer division truncation: e.g. pred returned 0 while gold returned a non-zero float
    if pred_rows and gold_rows and len(pred_rows) == len(gold_rows) == 1:
        p_val = pred_rows[0][0] if len(pred_rows[0]) > 0 else None
        g_val = gold_rows[0][0] if len(gold_rows[0]) > 0 else None
        if isinstance(p_val, int) and isinstance(g_val, float):
            if p_val == 0 and abs(g_val) > 0:
                return "type_or_div_error"
        # Also check if pred SQL divides without CAST(... AS REAL)
        if "/" in pred_sql and "cast" not in pred_sql.lower():
            if any(isinstance(cell, float) for row in gold_rows for cell in row):
                return "type_or_div_error"

    # 6. Default execution mismatch
    return "logic_mismatch"
