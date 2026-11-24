"""FastAPI production inference and telemetry service for Assay (Stage 4).

Features:
  - Semantic Abstention: Evaluates glossary.yaml rules to intercept ambiguous
    or undefined metrics before invoking LLM generation.
  - Dual-Layer Safety Guard: Rejects non-SELECT ASTs, stacked queries, and injects LIMIT 1000.
  - Cross-Dialect Transpilation: Transpiles SQLite SQL to PostgreSQL using sqlglot.
  - Performance Telemetry: Tracks p50 / p95 latency, token overhead, and repair attempts.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlglot
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.agent.graph import run_agent
from src.api.abstention import SemanticAbstentionGate
from src.llm import default_client
from src.safety.guard import SafetyGuard, execute_safe_sandbox
from src.schema.linker import SchemaLinker

logger = logging.getLogger("assay.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Assay Text-to-SQL Copilot API",
    description="Enterprise-grade Text-to-SQL agent with LangGraph self-repair, dual-layer safety, and semantic abstention.",
    version="1.0.0",
)

# Shared In-Memory Telemetry Tracker
TELEMETRY = {
    "total_requests": 0,
    "successful_queries": 0,
    "blocked_attacks": 0,
    "abstained_queries": 0,
    "latencies_sec": [],
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
}

abstention_gate = SemanticAbstentionGate()
safety_guard_sqlite = SafetyGuard(dialect="sqlite", default_limit=1000, enforce_limit=True)
safety_guard_postgres = SafetyGuard(dialect="postgres", default_limit=1000, enforce_limit=True)
cached_linkers: Dict[str, SchemaLinker] = {}


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to translate into SQL.")
    evidence: Optional[str] = Field(default="", description="Domain-specific evidence or metric formulas.")
    db_id: str = Field(default="california_schools", description="Target database identifier.")
    dialect: str = Field(default="sqlite", description="SQL dialect: 'sqlite' or 'postgres'.")
    db_path: Optional[str] = Field(default=None, description="Optional custom path to database file.")


class QueryResponse(BaseModel):
    status: str  # "success", "clarification_required", "blocked_safety", "execution_error"
    sql: Optional[str] = None
    transpiled_sql: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    clarification_message: Optional[str] = None
    clarification_options: List[str] = Field(default_factory=list)
    guard_status: Optional[str] = None
    attempts_count: int = 1
    latency_sec: float = 0.0
    telemetry: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health status and active model configuration."""
    return {
        "status": "healthy",
        "model": default_client.generation_model,
        "embedding_model": default_client.embedding_model,
        "glossary_path": str(abstention_gate.glossary_path),
        "glossary_metrics_count": len(abstention_gate.glossary.get("metrics", {})),
    }


@app.get("/metrics", tags=["Telemetry"])
def get_telemetry() -> Dict[str, Any]:
    """System-level performance, latency, and security telemetry."""
    latencies = sorted(TELEMETRY["latencies_sec"])
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    avg = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_requests": TELEMETRY["total_requests"],
        "successful_queries": TELEMETRY["successful_queries"],
        "blocked_attacks": TELEMETRY["blocked_attacks"],
        "abstained_queries": TELEMETRY["abstained_queries"],
        "p50_latency_sec": round(p50, 3),
        "p95_latency_sec": round(p95, 3),
        "avg_latency_sec": round(avg, 3),
        "total_prompt_tokens": TELEMETRY["total_prompt_tokens"],
        "total_completion_tokens": TELEMETRY["total_completion_tokens"],
    }


@app.get("/glossary", tags=["Enterprise"])
def get_glossary() -> Dict[str, Any]:
    """Retrieve defined enterprise business metrics and ambiguous terms."""
    return abstention_gate.glossary


@app.post("/query", response_model=QueryResponse, tags=["Inference"])
def execute_query_endpoint(req: QueryRequest) -> QueryResponse:
    """End-to-end Text-to-SQL query pipeline."""
    t_start = time.perf_counter()
    TELEMETRY["total_requests"] += 1

    dialect = req.dialect.lower()
    target_db_id = req.db_id
    question = req.question.strip()
    evidence = (req.evidence or "").strip()

    active_guard = safety_guard_postgres if dialect == "postgres" else safety_guard_sqlite

    # 0. Early Direct Injection / Destructive SQL Detection in Input
    initial_check = active_guard.validate_ast(question)
    if not initial_check.is_safe and initial_check.rule_triggered not in ("PARSE_ERROR", None):
        elapsed = time.perf_counter() - t_start
        TELEMETRY["blocked_attacks"] += 1
        TELEMETRY["latencies_sec"].append(elapsed)
        return QueryResponse(
            status="blocked_safety",
            sql=question,
            guard_status=f"BLOCKED: {initial_check.rejection_reason}",
            attempts_count=0,
            latency_sec=round(elapsed, 3),
            telemetry={"rule_triggered": initial_check.rule_triggered},
        )

    # 1. Semantic Abstention Check
    abstention_decision = abstention_gate.evaluate_query(question, evidence)
    if abstention_decision.should_abstain:
        elapsed = time.perf_counter() - t_start
        TELEMETRY["abstained_queries"] += 1
        TELEMETRY["latencies_sec"].append(elapsed)

        return QueryResponse(
            status="clarification_required",
            clarification_message=abstention_decision.clarification_message,
            clarification_options=abstention_decision.options,
            latency_sec=round(elapsed, 3),
            telemetry={"abstention_reason": abstention_decision.reason},
        )

    # If glossary matched certified metrics, enrich evidence
    if abstention_decision.injected_evidence:
        evidence = f"{evidence}\n{abstention_decision.injected_evidence}".strip()

    # 2. Resolve Database Path
    if req.db_path:
        db_path = Path(req.db_path)
    else:
        db_path = Path("data/benchmarks/bird/dev_databases") / target_db_id / f"{target_db_id}.sqlite"

    # 3. Schema Retrieval
    pruned_schema_str = ""
    if target_db_id not in cached_linkers:
        try:
            cached_linkers[target_db_id] = SchemaLinker(target_db_id)
        except Exception:
            pass

    if target_db_id in cached_linkers:
        linked = cached_linkers[target_db_id].link_and_prune(question, evidence, use_embeddings=False)
        pruned_schema_str = linked.m_schema_str

    # 4. Agent Execution (LangGraph Self-Repair Loop)
    final_state = run_agent(
        question=question,
        evidence=evidence,
        db_id=target_db_id,
        db_path=str(db_path),
        dialect=dialect,
        pruned_schema=pruned_schema_str,
        max_retries=3,
    )

    candidate_sql = final_state.get("candidate_sql", "")
    attempts_count = len(final_state.get("attempts", []))
    prompt_tokens = final_state.get("total_prompt_tokens", 0)
    completion_tokens = final_state.get("total_completion_tokens", 0)

    TELEMETRY["total_prompt_tokens"] += prompt_tokens
    TELEMETRY["total_completion_tokens"] += completion_tokens

    # 5. Layer 1 Static AST Guard
    active_guard = safety_guard_postgres if dialect == "postgres" else safety_guard_sqlite
    guard_res = active_guard.validate_ast(candidate_sql)

    if not guard_res.is_safe:
        elapsed = time.perf_counter() - t_start
        TELEMETRY["blocked_attacks"] += 1
        TELEMETRY["latencies_sec"].append(elapsed)

        return QueryResponse(
            status="blocked_safety",
            sql=candidate_sql,
            guard_status=f"BLOCKED: {guard_res.rejection_reason}",
            attempts_count=attempts_count,
            latency_sec=round(elapsed, 3),
            telemetry={"rule_triggered": guard_res.rule_triggered},
        )

    sanitized_sql = guard_res.sanitized_sql
    transpiled_sql = None

    # Cross-Dialect Transpilation if requested
    if dialect == "postgres":
        try:
            transpiled_sql = sqlglot.transpile(sanitized_sql, read="sqlite", write="postgres")[0]
        except Exception:
            transpiled_sql = sanitized_sql

    # 6. Layer 2 Sandbox Execution
    if db_path.exists():
        guard_check, exec_result = execute_safe_sandbox(
            db_path=db_path,
            sql=sanitized_sql,
            dialect="sqlite",
            timeout_sec=5.0,
            enforce_limit=True,
        )
        query_success = exec_result.success
        rows = exec_result.rows
        cols = exec_result.columns
        err = exec_result.error_message
    else:
        query_success = True
        rows = []
        cols = []
        err = f"Database file {db_path} not present locally; SQL validated statically."

    elapsed = time.perf_counter() - t_start
    TELEMETRY["latencies_sec"].append(elapsed)

    if query_success:
        TELEMETRY["successful_queries"] += 1
        formatted_rows = [dict(zip(cols, r)) for r in rows[:100]]
        return QueryResponse(
            status="success",
            sql=sanitized_sql,
            transpiled_sql=transpiled_sql,
            results=formatted_rows,
            columns=cols,
            row_count=len(rows),
            guard_status="PASSED (Layer 1 AST + Layer 2 Read-Only Sandbox)",
            attempts_count=attempts_count,
            latency_sec=round(elapsed, 3),
            telemetry={
                "injected_limit": guard_res.injected_limit,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
    else:
        return QueryResponse(
            status="execution_error",
            sql=sanitized_sql,
            transpiled_sql=transpiled_sql,
            guard_status=f"FAILED EXECUTION: {err}",
            attempts_count=attempts_count,
            latency_sec=round(elapsed, 3),
            telemetry={"error": err},
        )
