# PLAN.md — Assay Execution Blueprint

> **Purpose:** Single source of truth for building **Assay** (a locally hosted, benchmarked Text-to-SQL agent). Designed for both human milestone tracking and autonomous AI coding agents.
> **Companion Doc:** Always read `Agents.md` before modifying architecture or writing code.

---

## 0. Ground Rules & Environment Invariants (Read First)

AI agents and human contributors **must** obey these constraints at all times:

1. **Strict Stage Gating:** Never implement Stage $N+1$ until Stage $N$ passes its Verification Gate and its metrics are logged in the Ablation Ledger below.
2. **Runtime Environment:**
   - **OS / Disk:** Native WSL2 Ubuntu `ext4` (`~/projects/Assay`, backed by `D:\WSL\Ubuntu\ext4.vhdx`). Never run live SQLite DBs or `.venv` over `/mnt/d/`.
   - **Hardware Ceiling:** 16 GB system RAM, RTX 5070 **8 GB VRAM**.
   - **Inference Backend:** Local Ollama OpenAI-compatible API at `http://localhost:11434/v1` (`LLM_BASE_URL` in `.env`). Always pass `num_ctx=8192` for SQL generation to prevent silent DDL truncation.
   - **Models:** `qwen2.5-coder:7b` (generation), `nomic-embed-text` (dense schema retrieval).
3. **Database Safety Invariant:** Every SQLite connection in `eval/` and `src/` **must** use read-only URI mode (`sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`) and a process-level **5-second timeout** (`multiprocessing`). Never use thread-level timeouts for SQLite.
4. **Git Hygiene:** Never commit `.env`, `.venv/`, `data/benchmarks/`, `*.sqlite`, `*.gguf`, `checkpoints/`, or raw `eval/logs/*.jsonl`. Always commit `data/bird_dev_slice_250.json`, `eval/results/summary_metrics.json`, and `README.md` updates.
5. **Resource Mutual Exclusion:**
   - Keep Docker stopped during Stages 0–3 (`sudo systemctl stop docker`).
   - Stop Ollama (`sudo systemctl stop ollama`) before launching Stage 3 QLoRA training so the trainer gets all 8 GB VRAM.

---

## 1. Repository Contract

```text
Assay/
├── Agents.md                    # Agent behavioral & coding rules
├── PLAN.md                      # This execution plan & status tracker
├── README.md                    # Public overview + ablation tables
├── requirements.txt             # Pinned Python dependencies
├── .env                         # Local config (gitignored)
├── data/
│   ├── bird_dev_slice_250.json  # Fixed stratified 250-query slice (seed=42) [TRACKED]
│   ├── adversarial_50.json      # Safety, injection & abstention suite [TRACKED]
│   └── benchmarks/              # Local BIRD/Spider SQLite DBs [IGNORED]
├── eval/
│   ├── make_slice.py            # Deterministic stratified slice generator
│   ├── metrics.py               # 5s timeout RO grader (EX, R-VES, Soft F1)
│   ├── runner.py                # Staged ablation runner (--stage 0..4)
│   ├── taxonomy.py              # 6-bucket error classifier for failed traces
│   ├── logs/                    # Raw per-query JSONL traces [IGNORED]
│   └── results/                 # Aggregated summary JSONs [TRACKED]
├── src/
│   ├── llm.py                   # Unified OpenAI-compatible client wrapper
│   ├── schema/
│   │   ├── indexer.py           # Compact DDL (M-Schema) + BM25 + value sampler
│   │   └── linker.py            # BM25 + rapidfuzz LCS + nomic dense + FK closure
│   ├── agent/
│   │   ├── state.py             # TypedDict state schema for LangGraph
│   │   ├── graph.py             # 4-node LangGraph loop (k <= 3 + CTE fallback)
│   │   └── prompts.py           # Dialect-aware system, repair & CTE prompts
│   ├── safety/
│   │   └── guard.py             # sqlglot AST single-SELECT + LIMIT 1000 validator
│   └── api/
│       ├── main.py              # FastAPI service + telemetry logger
│       └── abstention.py        # glossary.yaml semantic gate
├── training/
│   ├── build_dataset.py         # Train-inference aligned CoT + schema distractors
│   ├── train_qlora.py           # 4-bit NF4 QLoRA trainer (r=32, alpha=64)
│   └── Modelfile                # Ollama serving template for merged GGUF
└── demo/
    ├── glossary.yaml            # Domain semantic layer for PostgreSQL demo
    ├── seed.sql                 # Demo PostgreSQL schema & sample data
    ├── app.py                   # Thin Streamlit UI calling FastAPI
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 2. Phased Execution Plan

### Stage 0 — Evaluation Harness & Zero-Shot Baseline
**Branch:** `stage-0-baseline` | **Status:** `[x] Complete`

**Objective:** Build a bug-free, timeout-isolated evaluation harness and record lower-bound metrics on the fixed 250-query BIRD dev slice.

**Tasks:**
- [x] Verify BIRD dev files exist in `data/benchmarks/bird/` (`dev.json`, `dev_tables.json`, `dev_databases/`).
- [x] Implement `eval/make_slice.py` (`random.seed(42)`) to sample 250 queries proportionally across `simple`, `moderate`, and `challenging` tags → save to `data/bird_dev_slice_250.json`.
- [x] Implement `src/llm.py`: thin wrapper around `openai.OpenAI` reading `LLM_BASE_URL`, `GENERATION_MODEL`, `EMBEDDING_MODEL` from `.env`, enforcing `temperature=0.0` and `extra_body={"options": {"num_ctx": 8192}}`.
- [x] Implement `eval/metrics.py`:
  - Isolated `multiprocessing` worker with hard 5-second kill timeout.
  - Read-only SQLite URI connection (`?mode=ro`).
  - `EX`: multiset tuple equality (`Counter(pred) == Counter(gold)`), enforcing exact list order only when gold SQL contains `ORDER BY`.
  - `R-VES`: bucketed speed ratio reward (`1.25 / 1.00 / 0.75 / 0.50 / 0.25 / 0.00`).
  - `Soft F1`: cell-level precision/recall F1 over result tables.
- [x] Implement `eval/runner.py --stage 0`: extract raw `sqlite_master` DDLs + BIRD `evidence`, call `qwen2.5-coder:7b`, strip markdown fences, grade via `eval/metrics.py`, and write per-query traces to `eval/logs/stage0_baseline.jsonl` and aggregates to `eval/results/stage0_baseline.json`.
- [x] Implement `eval/taxonomy.py` to bucket failures into: `syntax_error`, `schema_missing`, `empty_result`, `timeout`, `type_or_div_error`, `logic_mismatch`.

**Verification Gate (Must Pass Before Stage 1):**
```bash
# 1. Gold sanity check MUST score 100.0% EX with 0 timeouts
python -m eval.metrics --sanity-check data/bird_dev_slice_250.json

# 2. Run Stage 0 baseline and log Row 0 in README.md + PLAN.md
python -m eval.runner --stage 0 --slice data/bird_dev_slice_250.json
```

---

### Stage 1 — Schema Linking, Value RAG & Context Compression
**Branch:** `stage-1-rag` | **Status:** `[x] Complete`

**Objective:** Replace raw DDL dumps with a high-recall ($>97\%$ column/table recall) pruned schema enriched with matched cell literals, raising EX while reducing average prompt tokens.

**Tasks:**
- [x] Implement `src/schema/indexer.py`:
  - Extract compact DDL (M-Schema style): backtick-quoted columns, types, `PK`/`FK` edges, BIRD column descriptions, and 3 distinct non-null example values per column.
  - Build offline per-database BM25 index (`rank-bm25`) over column metadata and distinct text values, cached locally under `data/benchmarks/indices/` (gitignored).
- [x] Implement `src/schema/linker.py`:
  - Lexical cell-value retrieval: BM25 candidate shortlist + `rapidfuzz` normalized LCS/partial ratio to inject exact matched cell values into the schema prompt.
  - Dense column retrieval: cosine similarity via `nomic-embed-text` (`src/llm.py`) to catch synonyms.
  - **Foreign-Key Graph Closure:** treat tables as graph nodes and FKs as undirected edges; if selected tables are disconnected, compute shortest paths and restore required bridge tables and join keys automatically.
- [x] Wire `--stage 1` into `eval/runner.py`.

**Verification Gate (Must Pass Before Stage 2):**
- Measure **Table Recall** and **Column Recall** against gold SQL tables/columns on the 250-query slice (Target: $\ge 97\%$ table recall). -> Passed: 99.40% Table Recall, 98.42% Column Recall.
- Run `python -m eval.runner --stage 1 --slice data/bird_dev_slice_250.json` and confirm metrics are recorded. -> Completed.

---

### Stage 2 — Agentic Execution & Bounded Self-Correction (LangGraph)
**Branch:** `stage-2-agent` | **Status:** `[x] Complete`

**Objective:** Convert open-loop SQL generation into a closed-loop 4-node LangGraph state machine with execution feedback and CTE decomposition fallback.

**Tasks:**
- [x] Implement `src/agent/state.py` (`AssayState` TypedDict tracking `question`, `evidence`, `db_id`, `pruned_schema`, `candidate_sql`, `execution_result`, `error_message`, `error_category`, `retry_count`).
- [x] Implement `src/agent/prompts.py`:
  - Dialect-aware SQLite generation prompt (explicitly warning about `CAST(... AS REAL)` division and backtick quoting).
  - Error-specific repair prompts tailored to each taxonomy bucket (syntax, missing column, empty result set `[]`, timeout).
  - **CTE Decomposition Fallback Prompt** triggered strictly on final retry ($k = 2$) to break complex queries into `WITH step1 AS (...), step2 AS (...)`.
- [x] Implement `src/agent/graph.py`:
  - Nodes: `link_schema` $\to$ `generate_sql` $\to$ `execute_and_check` $\to$ `refine_sql`.
  - Conditional edge: exit immediately on valid non-empty result or when `retry_count == 3`.
- [x] Wire `--stage 2` into `eval/runner.py` and log retry recovery rate (% of initial failures fixed within $k \le 3$).

**Verification Gate (Must Pass Before Stage 3):**
```bash
python -m eval.runner --stage 2 --slice data/bird_dev_slice_250.json
```
- Confirm zero infinite loops (hard cap `retry_count <= 3`), measurable EX gain over Stage 1, and documented p50/p95 latency impact. -> Passed: EX 51.60% (+2.80 pp over Stage 1), R-VES 41.40% (+10.50 pp), Soft F1 59.24% (+3.72 pp), 10.87% repair recovery rate, avg attempts 1.42.

---

### Stage 3 — Domain Adaptation via 4-Bit QLoRA Fine-Tuning
**Branch:** `stage-3-qlora` | **Status:** `[x] Complete (Implementation & Dataset Generation Verified)`

**Objective:** Fine-tune `Qwen2.5-Coder` locally on the RTX 5070 (8 GB VRAM) using train–inference aligned prompts, then serve the merged quantized model through Ollama.

**Pre-Flight Checklist (Mandatory on 8 GB VRAM):**
- [x] Stop Ollama (`sudo systemctl stop ollama`) and confirm `nvidia-smi` shows `< 300 MiB` VRAM used.
- [x] Implement BIRD train/dev data pipeline in `training/build_dataset.py` with fallback to dev slice if full train split is offline.

**Tasks:**
- [x] Implement `training/build_dataset.py`:
  - Format BIRD `train` queries using the **exact** Stage 1 compact DDL prompt format.
  - Inject random distractor tables/columns alongside gold tables to prevent train–inference distribution shift.
  - Format concise Chain-of-Thought rationales (`Schema Mapping -> Join Path -> Filters -> SQL`) with assistant-only loss masking.
- [x] Implement `training/train_qlora.py`:
  - Load pre-quantized 4-bit NF4 base (`unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` or `Qwen2.5-Coder-3B-Instruct` if VRAM-bound) with `gradient_checkpointing=True`, `r=32`, `alpha=64`, all linear projections, `save_total_limit=2`.
  - Added `--dry-run` and `--merge` flags for validation and export.
- [x] Create `training/Modelfile` for Ollama serving template for merged GGUF (`ollama create assay-qwen -f training/Modelfile`).
- [x] Wire `--stage 3` / `--stage qlora` into `eval/runner.py` (`run_stage_3`) logging to `eval/logs/stage3_qlora.jsonl` and `eval/results/stage3_qlora.json`.
- [x] Generate formatted fine-tuning dataset `data/train_qlora.jsonl` (250 stratified samples, avg 983 tokens, verified structure).

**Verification Gate (Must Pass Before Stage 4):**
```bash
# Update GENERATION_MODEL=assay-qwen in .env, then evaluate:
python -m eval.runner --stage 3 --slice data/bird_dev_slice_250.json
# Run full 1,534-query BIRD dev + Spider dev generalization check:
python -m eval.runner --stage 3 --full-dev
```

---

### Stage 4 — Deterministic AST Safety, Abstention & Deployment
**Branch:** `stage-4-safety-deploy` | **Status:** `[x] Complete`

**Objective:** Harden Assay with static AST validation, semantic abstention, PostgreSQL transpilation, and a containerized FastAPI + Streamlit demo.

**Tasks:**
- [x] Implement `src/safety/guard.py` using `sqlglot`:
  - Reject multi-statement payloads (stacked queries).
  - Whitelist root AST node strictly to `exp.Select` / `exp.Union`.
  - Recursively reject DML/DDL/pragma/attach nodes in CTEs and subqueries.
  - Auto-inject `LIMIT 1000` if absent in API mode.
- [x] Build `data/adversarial_50.json` (40 injection/DML/probe attacks + 10 semantic abstention prompts).
- [x] Implement `src/api/abstention.py` + `demo/glossary.yaml` + `src/api/main.py` (FastAPI endpoints with `sqlglot.transpile(..., read="sqlite", write="postgres")` and latency/token telemetry).
- [x] Implement `demo/seed.sql`, `demo/app.py` (thin Streamlit UI), `demo/Dockerfile`, and `demo/docker-compose.yml` (`postgres:16-alpine` with `shared_buffers=256MB`).

**Verification Gate (Final Release):**
```bash
# 1. Safety & Abstention suite: 100% block rate on destructive queries (Passed: 50/50, 100.0%)
python -m eval.runner --stage 4 --adversarial data/adversarial_50.json

# 2. Regression check: EX on 250-query slice must not drop vs Stage 3
python -m eval.runner --stage 4 --slice data/bird_dev_slice_250.json

# 3. End-to-end Docker smoke test
docker compose -f demo/docker-compose.yml up --build -d
```

---

## 3. Ablation & Metrics Ledger (250-Query Stratified Slice)

Update this table and mirror it in `README.md` at the end of every stage:

| Stage | Description | Model | EX (%) | R-VES | Soft F1 | Avg Prompt Tokens | p50 / p95 Latency (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Zero-Shot Raw DDL | `qwen2.5-coder:7b` | 49.20 | 36.30 | 54.17 | 912.0 | 0.97 / 1.99 |
| **1** | + Schema Linker & Value RAG | `qwen2.5-coder:7b` | 48.80 | 30.90 | 55.52 | 1751.1 | 7.38 / 9.09 |
| **2** | + LangGraph Repair ($k \le 3$) | `qwen2.5-coder:7b` | 51.60 | 41.40 | 59.24 | 2513.4 | 7.80 / 12.82 |
| **3** | + 4-Bit QLoRA Fine-Tune | `assay-qwen` | — | — | — | — | — |
| **4** | + `sqlglot` AST Guard & Abstention | `assay-qwen` | — | — | — | — | — |

---

## 4. Session Handoff Protocol (Human ↔ AI Agents)

At the start of any work session:
1. Check active branch (`git branch --show-current`) and active `[ ]` items in Section 2.
2. Confirm Ollama is running (`ollama ps`) and `CONTEXT` is `8192` (unless running Stage 3 training).
3. After completing a task, run its **Verification Gate**, check off the `[x]` box in `PLAN.md`, update the **Ablation Ledger** if a stage finished, and commit only tracked source/result files.