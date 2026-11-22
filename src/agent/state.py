"""State schema definitions for Assay's LangGraph execution-guided repair agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict


class QueryAttempt(TypedDict):
    attempt: int
    sql: str
    is_valid: bool
    error_message: Optional[str]
    error_category: Optional[str]
    rows: Optional[List[Tuple[Any, ...]]]
    llm_time_sec: float
    exec_time_sec: float
    prompt_tokens: int
    completion_tokens: int


class AssayState(TypedDict):
    # Inputs
    question: str
    evidence: str
    db_id: str
    db_path: str
    dialect: str  # "sqlite" or "postgres"

    # Schema & RAG
    pruned_schema: str
    selected_tables: List[str]

    # Agent Loop State
    candidate_sql: str
    execution_result: Optional[List[Tuple[Any, ...]]]
    is_valid: bool
    error_message: Optional[str]
    error_category: Optional[str]
    retry_count: int  # 0, 1, 2, 3 (max k=3 attempts)

    # Telemetry & History
    attempts: List[QueryAttempt]
    total_llm_time: float
    total_exec_time: float
    total_prompt_tokens: int
    total_completion_tokens: int
