# Assay — Project Knowledge & Agent Guidelines (`AGENTS.md`)

## 1. Project Summary

- **Name**: SQL Copilot (Repository: Assay / `sql-copilot`)
- **Type**: Modular, empirically evaluated Text-to-SQL agent with execution-guided repair, schema RAG, parameter-efficient fine-tuning (QLoRA), and dual-layer safety sandboxing.
- **Objective**: Translate natural language analytical queries into validated, read-only SQL across complex multi-table relational databases with minimal token overhead and sub-second median latency.
- **Target Environments**:
  - **Offline Benchmarking**: SQLite (faithful reproduction of BIRD and Spider benchmarks).
  - **Production / Live Serving**: PostgreSQL, FastAPI, Streamlit, Docker.
- **Primary LLMs**: Compact open-weight models (`Qwen/Qwen2.5-Coder-3B-Instruct` for 8 GB local VRAM; `Qwen2.5-Coder-7B-Instruct` for cloud training) alongside frontier API references.

---

## 2. Technical Stack

- **Language**: Python 3.10+
- **Agent Orchestration**: LangGraph (stateful directed cyclic graph)
- **Model Fine-Tuning**: PyTorch, Hugging Face `peft`, `trl`, `bitsandbytes` (4-bit NF4 QLoRA)
- **SQL Parsing & Static Analysis**: `sqlglot` (AST verification)
- **Databases**: SQLite (read-only mode `?mode=ro`), PostgreSQL (read-only roles)
- **Information Retrieval**: BM25 + Longest Common Substring (LCS) cell retrieval
- **API & UI**: FastAPI (REST endpoints, p50/p95 latency logging), Streamlit (interactive demo)
- **Containerization**: Docker, Docker Compose

---

## 3. Five-Stage Pipeline & Architecture

```
[NL Query + Evidence]
        │
        ▼
[Business Glossary & Abstention Check (glossary.yaml)] ──(Undefined/Ambiguous)──► [Clarification Request]
        │ (Grounded)
        ▼
[Bidirectional Schema Linker (Table ∪ Col) + BM25/LCS Value Retriever]
        │
        ▼
[SQL Generator: Qwen2.5-Coder Base / QLoRA Adapter]
        │
        ▼
[Layer 1: Static AST Guard (sqlglot)] ──(Non-SELECT / DDL / DML)──► [Block & Security Log]
        │ (Valid Read-Only AST)
        ▼
[Layer 2: Read-Only DB Sandbox (5s Timeout, LIMIT 1000)]
        │
        ├─► (Valid Result) ──► [Verified Output + SQL + Telemetry]
        │
        └─► (Syntax Error / Runtime Exception / Empty / All-NULL)
                 │
                 ▼
            [Attempts < 3?]
                 ├─► [Attempt 1: Execution-Guided Repair Node] ──► [Layer 1 AST]
                 ├─► [Attempt 2: Modular CTE Decomposition Fallback] ──► [Layer 1 AST]
                 └─► [Attempt 3+ Exceeded] ──► [Graceful Failure + Trace]
```

### Stage Details

1. **Stage 0 — Baseline**: Zero-shot prompting over raw DDL + domain evidence on open-weight and frontier models.
2. **Stage 1 — Compact Schema Representation & Hybrid Value RAG**:
   - _Compact DDL_: Columns, data types, primary/foreign keys, 3–5 distinct non-null values, and explicit `NULL` flags (prevents missing `IS NOT NULL` bugs).
   - _Bidirectional Schema Linking_: Merges Table-First and Column-First retrieval via set union; minimizes prompt tokens while maintaining schema recall on 3B/7B models. (Hard pruning disabled for frontier API models).
   - _Cell Retrieval_: Offline BM25 index + LCS matching on database text columns to ground noisy query literals.
   - _Dialect Guardrails_: Dialect-specific prompt rules (e.g., `CAST(x AS REAL)` for SQLite integer division).
3. **Stage 2 — LangGraph Execution-Guided Self-Repair Loop**:
   - Cyclic state machine: `link_and_retrieve` → `generate_sql` → `execute_sandbox` → `repair_sql`.
   - _Dual Error Triggers_: Catches runtime/syntax exceptions AND empty (`len(rows) == 0`) or all-`NULL` outputs.
   - _Bounded Retry Cap_: Strict maximum of 3 attempts (`k <= 3`) to prevent drift/hallucination.
   - _CTE Decomposition Fallback_: Attempt 2 enforces Common Table Expressions (`WITH step_1 AS (...)`) to isolate multi-hop join/aggregation failures.
4. **Stage 3 — QLoRA Model Specialization**:
   - _Quantization_: 4-bit NormalFloat (`NF4`) double quantization via `bitsandbytes`.
   - _LoRA Hyperparameters_: Rank `r = 32`, `alpha = 64`, applied to all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - _Data Formulation_: BIRD train split formatted with Stage 1 compact schema, execution-verified chain-of-thought (CoT: tables → join keys → filters → casting), and 2–3 distractor tables/columns for noise resistance. Optional Execution-DPO.
5. **Stage 4 — Dual-Layer Safety & Enterprise Grounding**:
   - _Layer 1 (Static AST)_: `sqlglot` validates single `SELECT` statement; rejects `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `PRAGMA`, and multi-statement/stacked injections.
   - _Layer 2 (DB Sandbox)_: Enforces `?mode=ro` (SQLite) / read-only role (Postgres), 5s query execution timeout, automatic `LIMIT 1000`.
   - _Enterprise Glossary_: `demo/glossary.yaml` binds business metrics to SQL expressions. Undefined terms trigger abstention rather than hallucinated SQL.

---

## 4. Repository Structure & File Mapping

```text
sql-copilot/
├── data/
│   ├── bird_dev_slice_250.json      # Stratified 250-query slice for rapid iteration
│   └── adversarial_50.json          # 50 adversarial / SQL injection test cases
├── eval/
│   ├── runner.py                    # Staged ablation runner producing JSONL traces
│   ├── metrics.py                   # Isolated evaluator for EX, R-VES, Soft F1, SLR
│   └── taxonomy_tagger.py           # CLI for 6-bucket error classification
├── src/
│   ├── schema/
│   │   ├── indexer.py               # Distinct-value sampler, NULL flags, BM25+LCS indexer
│   │   └── linker.py                # Bidirectional schema linking (Table-First ∪ Column-First)
│   ├── agent/
│   │   ├── graph.py                 # LangGraph state machine (max 3 retries, CTE fallback)
│   │   └── prompts.py               # Dialect-aware prompts (SQLite vs. PostgreSQL)
│   ├── safety/
│   │   └── guard.py                 # sqlglot AST validator + read-only sandbox runner
│   └── api/
│       └── main.py                  # FastAPI inference service with p50/p95 latency logging
├── training/
│   ├── build_dataset.py             # Preprocesses BIRD train, injects CoT and distractor noise
│   └── train_qlora.py               # 4-bit NF4 QLoRA training script (PEFT/TRL)
├── demo/
│   ├── glossary.yaml                # Declarative business logic / metric mappings
│   └── app.py                       # Streamlit UI (SQL view, repair traces, execution output)
├── Dockerfile                       # Production container definition
├── docker-compose.yml               # Multi-service setup (FastAPI + Streamlit + Postgres)
└── requirements.txt                 # Project dependencies
```

---

## 5. CLI Execution & Workflows

### 5.1 Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

### 5.2 Schema Indexing (BIRD / Spider)

```bash
python -m src.schema.indexer --benchmark bird --split dev --db-dir data/benchmarks/bird/dev_databases
```

### 5.3 Evaluation & Safety Audits

```bash
# Rapid ablation on 250-query slice
python -m eval.runner --stage baseline --dataset data/bird_dev_slice_250.json
python -m eval.runner --stage rag      --dataset data/bird_dev_slice_250.json
python -m eval.runner --stage agent    --dataset data/bird_dev_slice_250.json

# Full evaluation & adversarial verification
python -m eval.runner --stage qlora    --dataset data/benchmarks/bird/dev.json --full
python -m eval.runner --stage safety   --dataset data/adversarial_50.json
```

### 5.4 QLoRA Fine-Tuning

```bash
python -m training.build_dataset --bird-train data/benchmarks/bird/train.json --output data/train_qlora.jsonl
python -m training.train_qlora \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --dataset data/train_qlora.jsonl \
  --lora-r 32 --lora-alpha 64 --quantization nf4
```

### 5.5 Production Demo (Docker)

```bash
docker compose up --build
# FastAPI Swagger Docs: http://localhost:8000/docs
# Streamlit UI:        http://localhost:8501
```

---

## 6. Evaluation Framework & Error Taxonomy

### Core Metrics

- **Execution Accuracy (`EX`)**: Exact result match against gold SQL under 5s sandbox timeout.
- **Reward-Based Valid Efficiency Score (`R-VES`)**: BIRD metric measuring computational efficiency vs. gold SQL.
- **Soft F1-Score**: Cell-level partial matching (for near-miss queries).
- **Schema Linking Recall (`SLR`)**: Fraction of required tables and columns retained by Stage 1 linker.
- **Repair Recovery Rate**: % of failed/empty queries successfully repaired within `<= 3` attempts.

### 6-Bucket Failure Taxonomy (`eval/taxonomy_tagger.py`)

1. **Schema Linking Errors**: Missing columns, abbreviation mismatches, omitted bridge tables.
2. **Business Logic & Evidence**: Misapplied domain evidence, inverted boolean predicates (`AND`/`OR`).
3. **SQL Functions & Casting**: Truncation from integer division (`CAST(... AS REAL)` needed), invalid aggregate usage.
4. **Relational Join Paths**: Missing join predicates, wrong foreign keys, accidental Cartesian joins.
5. **Dirty Values & Predicates**: Case/whitespace mismatch in `LIKE`, missing `IS NOT NULL` filters.
6. **Benchmark Annotation Noise**: Ground truth SQL ambiguity or annotation discrepancies.

---

## 7. Critical Invariants for Autonomous Agents

- **Strict Read-Only Guarantee**: Never bypass `sqlglot` AST validation. Deny DDL/DML, system commands, and stacked queries before reaching any database connection.
- **Dialect Isolation**: SQLite is reserved for offline benchmark datasets (BIRD/Spider); PostgreSQL is reserved for demo/production. Keep SQL dialect specifics isolated within `src/agent/prompts.py`.
- **Bounded Graph Execution**: Ensure LangGraph self-repair cycles terminate at `<= 3` attempts. Attempt 2 must always apply CTE decomposition.
- **Abstention Over Hallucination**: When a business metric is absent from `demo/glossary.yaml`, emit a clarification request rather than generating speculative SQL.
- **Resource Limits**: Always apply a 5-second execution timeout and a default `LIMIT 1000` to sandbox queries.
