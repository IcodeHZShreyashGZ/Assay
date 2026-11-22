"""Dialect-aware prompt templates, error-tailored repair prompts, and CTE fallback for Assay."""

from __future__ import annotations

import re
from typing import Tuple


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


def build_generation_prompt(
    schema_str: str,
    question: str,
    evidence: str = "",
    dialect: str = "sqlite",
) -> Tuple[str, str]:
    """Build system and user prompts for initial SQL generation."""
    if dialect.lower() == "postgres":
        dialect_rules = (
            "- PostgreSQL syntax: Quote mixed-case or space-containing identifiers in double quotes (e.g. \"Col Name\").\n"
            "- For float division, use `CAST(col AS FLOAT)` or `col::numeric`.\n"
            "- Use explicit JOIN ... ON syntax."
        )
    else:
        dialect_rules = (
            "- SQLite syntax: Quote columns containing spaces or parentheses with backticks (e.g. `School Type`, `Enrollment (Ages 5-17)`).\n"
            "- For integer division, always wrap numerators in `CAST(col AS REAL)` to prevent integer truncation to 0.\n"
            "- Join tables using explicit foreign key equality (e.g. `JOIN t2 ON t1.fk = t2.pk`).\n"
            "- Use exact casing and spelling for string literals as given in the schema or evidence."
        )

    system_prompt = (
        "You are an expert SQL engineer. Given a compact database schema with types, foreign keys, "
        "sample values, and domain evidence, write a single valid SQL query that accurately answers the question.\n"
        "Strict Requirements:\n"
        "1. Output ONLY the SQL query inside a ```sql``` code block. Do NOT include explanations, comments, or extra text.\n"
        f"{dialect_rules}\n"
        "5. Do NOT invent columns or tables. Use ONLY elements defined in the provided schema."
    )

    user_prompt = (
        f"{schema_str}\n\n"
        f"【Domain Evidence】\n{evidence if evidence else 'None'}\n\n"
        f"【Question】\n{question}\n\n"
        f"Write the {dialect.upper()} SQL query to answer the question:"
    )

    return system_prompt, user_prompt


def build_repair_prompt(
    schema_str: str,
    question: str,
    evidence: str,
    failed_sql: str,
    error_category: str,
    error_message: str,
    retry_count: int,
    dialect: str = "sqlite",
) -> Tuple[str, str]:
    """Build error-tailored self-repair prompt or CTE decomposition fallback prompt.
    
    Attempt retry_count == 2 triggers the mandatory CTE Decomposition Fallback.
    """
    is_cte_fallback = (retry_count == 2)

    system_prompt = (
        "You are an expert SQL repair engineer. Your task is to diagnose and correct a failed SQL query.\n"
        "Output ONLY the corrected SQL query inside a ```sql``` code block. Do NOT include explanations, markdown before/after, or apologies."
    )

    # Error-specific diagnosis
    diagnosis_lines = [
        f"【Failed SQL Query】\n```sql\n{failed_sql}\n```",
        f"【Error Classification】: {error_category}",
    ]
    if error_message:
        diagnosis_lines.append(f"【Error Message】: {error_message}")

    if is_cte_fallback:
        # Attempt 2: Mandatory Common Table Expression (CTE) Decomposition
        repair_instruction = (
            "【MANDATORY REPAIR STRATEGY: CTE DECOMPOSITION】\n"
            "The query has failed previous attempts. You MUST decompose the logic into modular Common Table Expressions (`WITH ...`):\n"
            "1. `WITH step_1 AS (...)`: Isolate table joins and primary filter conditions.\n"
            "2. `step_2 AS (...)`: Perform necessary aggregations, groupings, or mathematical expressions.\n"
            "3. Final `SELECT`: Project the requested answer from the CTE with appropriate ordering / limit.\n"
            "Ensure all column identifiers with spaces or symbols are enclosed in backticks."
        )
    else:
        # Error-specific repair instructions
        if error_category == "syntax_error":
            repair_instruction = (
                "【DIAGNOSIS: SYNTAX ERROR】\n"
                "- The SQL parser rejected the query.\n"
                "- Common cause: Column or table names containing spaces, hyphens, or parentheses (e.g. `Free Meal Count (K-12)`) were not quoted.\n"
                "- Action: Wrap EVERY column and table identifier containing spaces or symbols in backticks (e.g. `col name`). Check commas and clauses."
            )
        elif error_category == "schema_missing":
            repair_instruction = (
                "【DIAGNOSIS: UNKNOWN COLUMN OR TABLE】\n"
                "- The query referenced a table or column that does NOT exist in the database.\n"
                "- Action: Review the 【Compact Database Schema】 below carefully. Replace hallucinated column/table names with the exact identifiers present in the schema."
            )
        elif error_category == "empty_result":
            repair_instruction = (
                "【DIAGNOSIS: EMPTY RESULT SET [] OR ALL-NULLS】\n"
                "- The query executed successfully but returned ZERO rows (or all NULL values).\n"
                "- Common causes: Overly restrictive WHERE filter, wrong casing in string comparison, inverted boolean logic (AND instead of OR), or incorrect join keys.\n"
                "- Action: Check 【Matched Cell Values】 for exact string casing. Remove unnecessary filter predicates and verify foreign key join conditions."
            )
        elif error_category == "timeout":
            repair_instruction = (
                "【DIAGNOSIS: EXECUTION TIMEOUT (>5s)】\n"
                "- The query exceeded the 5-second sandbox limit.\n"
                "- Common cause: Accidental Cartesian product (cross join) caused by a missing or flawed ON condition in a JOIN.\n"
                "- Action: Ensure every JOIN has an explicit, indexed equality predicate (e.g. `JOIN t2 ON t1.fk = t2.pk`)."
            )
        elif error_category == "type_or_div_error":
            repair_instruction = (
                "【DIAGNOSIS: INTEGER DIVISION TRUNCATION / TYPE MISMATCH】\n"
                "- In SQLite, dividing two integer columns results in 0 (e.g. `10 / 20 = 0`).\n"
                "- Action: Wrap the numerator in `CAST(numerator AS REAL)` before division (e.g. `CAST(a AS REAL) / b`)."
            )
        else:
            repair_instruction = (
                "【DIAGNOSIS: LOGIC OR AGGREGATE MISMATCH】\n"
                "- The query produced incorrect output compared to the ground truth expectation.\n"
                "- Action: Carefully re-read 【Question】 and 【Domain Evidence】. Verify aggregate functions (SUM, AVG, COUNT, MAX), grouping columns, and ORDER BY direction."
            )

    user_prompt = (
        f"{schema_str}\n\n"
        f"【Domain Evidence】\n{evidence if evidence else 'None'}\n\n"
        f"【Question】\n{question}\n\n"
        f"{chr(10).join(diagnosis_lines)}\n\n"
        f"{repair_instruction}\n\n"
        f"Write the corrected {dialect.upper()} SQL query:"
    )

    return system_prompt, user_prompt
