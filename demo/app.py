"""Streamlit Interactive Web Demo for Assay (Stage 4).

Connects to the FastAPI backend service to provide:
  - Natural Language to SQL query translation with dialect selection (SQLite / PostgreSQL).
  - Real-time display of AST Guard validation & auto-injected LIMIT 1000.
  - Interactive display of self-repair traces (Attempts, CTE Fallback).
  - Business Glossary inspector & interactive clarification prompts on abstention.
"""

from __future__ import annotations

import os
import time
import requests
import streamlit as st

API_URL = os.getenv("ASSAY_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Assay — Text-to-SQL Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Assay — Enterprise Text-to-SQL Copilot")
st.caption("Locally hosted, empirically evaluated Text-to-SQL agent with LangGraph self-repair, dual-layer safety, and semantic abstention.")

# ---------------------------------------------------------------------------
# Sidebar: Configuration & Glossary
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    dialect = st.selectbox("Target SQL Dialect", ["SQLite", "PostgreSQL"], index=0)
    db_id = st.selectbox(
        "Database",
        [
            "california_schools",
            "card_games",
            "codebase_community",
            "debit_card_specializing",
            "european_football_2",
            "financial",
            "formula_1",
            "student_club",
            "superhero",
            "thrombosis_prediction",
            "toxicology",
        ],
        index=0,
    )

    st.divider()
    st.header("📖 Business Glossary")
    try:
        resp = requests.get(f"{API_URL}/glossary", timeout=2.0)
        if resp.status_code == 200:
            glossary_data = resp.json()
            metrics = glossary_data.get("metrics", {})
            st.write(f"**Certified Metrics ({len(metrics)}):**")
            for k, m in metrics.items():
                with st.expander(f"🔹 {m.get('name', k)}"):
                    st.markdown(f"**Definition:** {m.get('definition')}")
                    st.code(m.get("sql_expression"), language="sql")
                    st.caption(f"Tables: {', '.join(m.get('required_tables', []))}")

            ambiguous = glossary_data.get("ambiguous_terms", {})
            st.write(f"**Monitored Ambiguous Terms ({len(ambiguous)}):**")
            for k, amb in ambiguous.items():
                with st.expander(f"⚠️ {amb.get('term', k)}"):
                    st.caption(amb.get("clarification"))
        else:
            st.info("Glossary loaded from local default.")
    except Exception:
        st.info("API offline; glossary view unavailable.")

    st.divider()
    st.header("📊 Telemetry")
    try:
        m_resp = requests.get(f"{API_URL}/metrics", timeout=2.0)
        if m_resp.status_code == 200:
            m_data = m_resp.json()
            st.metric("Total Requests", m_data.get("total_requests", 0))
            col_a, col_b = st.columns(2)
            col_a.metric("p50 Latency", f"{m_data.get('p50_latency_sec', 0.0):.2f}s")
            col_b.metric("p95 Latency", f"{m_data.get('p95_latency_sec', 0.0):.2f}s")
            st.metric("Blocked Attacks", m_data.get("blocked_attacks", 0))
    except Exception:
        st.caption("Telemetry stream inactive.")

# ---------------------------------------------------------------------------
# Main Query Interface
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Query Specification")
    question_input = st.text_area(
        "Natural Language Analytical Question:",
        value="What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        height=100,
        placeholder="e.g. Which schools in Alameda County have math SAT scores above 500?",
    )
    evidence_input = st.text_input(
        "Domain Evidence / Business Context (Optional):",
        value="Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
        placeholder="e.g. Active users refers to status = 'active'",
    )

    run_btn = st.button("🚀 Generate & Execute SQL", type="primary", use_container_width=True)

with col2:
    st.subheader("Guardrail & Sandbox Status")
    st.info("🛡️ **Layer 1 AST Guard:** Rejects multi-statements, DDL/DML, and auto-injects `LIMIT 1000`.")
    st.info("🔒 **Layer 2 DB Sandbox:** Enforces read-only URI and isolated 5s execution timeout.")
    st.info("⚖️ **Semantic Abstention:** Clarifies ambiguous terms and stops hallucinated SQL.")

# ---------------------------------------------------------------------------
# Execution & Results
# ---------------------------------------------------------------------------
if run_btn and question_input.strip():
    with st.spinner("Linking Schema -> Generating Candidate SQL -> Sandboxing..."):
        payload = {
            "question": question_input,
            "evidence": evidence_input,
            "db_id": db_id,
            "dialect": dialect.lower(),
        }

        try:
            res = requests.post(f"{API_URL}/query", json=payload, timeout=30.0)
            data = res.json()
        except Exception as e:
            st.error(f"Failed to communicate with Assay API: {e}")
            st.stop()

        status_type = data.get("status")

        if status_type == "clarification_required":
            st.warning("⚠️ **Semantic Abstention Triggered (Clarification Required)**")
            st.markdown(data.get("clarification_message"))
            options = data.get("clarification_options", [])
            if options:
                st.write("**Clarification Options:**")
                for opt in options:
                    st.info(f"👉 {opt}")

        elif status_type == "blocked_safety":
            st.error(f"🚫 **Security Violation Detected!**")
            st.markdown(f"**AST Guard Decision:** {data.get('guard_status')}")
            if data.get("sql"):
                st.code(data.get("sql"), language="sql")

        elif status_type in ("success", "execution_error"):
            st.success("✅ **Query Verified & Sandboxed!**") if status_type == "success" else st.warning("⚠️ **Execution Returned Errors**")

            st.markdown("### Generated SQL Query")
            st.code(data.get("sql"), language="sql")

            if data.get("transpiled_sql") and dialect.lower() == "postgres":
                st.markdown("### Transpiled PostgreSQL Query")
                st.code(data.get("transpiled_sql"), language="sql")

            cols_res = data.get("columns", [])
            rows_res = data.get("results", [])

            st.markdown(f"### Query Results ({data.get('row_count', 0)} rows returned)")
            if rows_res:
                st.dataframe(rows_res, use_container_width=True)
            else:
                st.info("Result set is empty.")

            with st.expander("🛠️ Execution Telemetry & Guardrail Traces"):
                t_cols = st.columns(4)
                t_cols[0].metric("Attempts Count", data.get("attempts_count", 1))
                t_cols[1].metric("Latency", f"{data.get('latency_sec', 0.0):.2f}s")
                t_cols[2].metric("Prompt Tokens", data.get("telemetry", {}).get("prompt_tokens", 0))
                t_cols[3].metric("Completion Tokens", data.get("telemetry", {}).get("completion_tokens", 0))
                st.caption(f"Guardrail Note: {data.get('guard_status')}")
