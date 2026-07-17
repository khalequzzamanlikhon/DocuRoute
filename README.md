<div align="center">

# 🔎 DocuRoute

### Agentic, Query-Routing RAG System for Financial Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Gemini-orange?style=flat)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DocuRoute** is a production-grade Retrieval-Augmented Generation (RAG) system
that intelligently routes financial questions to the right retrieval strategy —
semantic search over document text, or a text-to-SQL agent against extracted
tables — and measures its own quality with a full evaluation pipeline.

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Usage](#usage) · [Evaluation](#evaluation) · [Roadmap](#roadmap)

</div>

---

## Demo

```
![DocuRoute Application Demo](demo.gif)
```

---

## Features

| Feature | Description |
|---|---|
| **Intelligent Query Routing** | LLM classifies each question as `unstructured`, `structured`, or `both` based on actual table schemas discovered at ingest time — not keyword rules |
| **Hybrid Retrieval** | BM25 keyword search + dense vector search (BGE), fused with Reciprocal Rank Fusion (RRF) — catches both exact terms and semantic meaning |
| **Cross-Encoder Reranking** | Second-pass reranker reads (query, chunk) jointly before handing context to the LLM, eliminating "similar but not relevant" results |
| **Text-to-SQL Agent** | Generates and safely executes SQL against tables extracted from PDFs — sandboxed with read-only DuckDB, keyword denylist, and row cap |
| **Grounded Generation** | Every factual sentence in an answer carries a `[C_n]` / `[T_n]` citation back to the specific chunk or SQL result that produced it |
| **Provider Fallback** | Groq (Llama 3.3 70B) is the primary LLM. On rate-limit, automatically falls back to Gemini 2.5 Flash — invisible to the user |
| **Evaluation Harness** | Ragas computes Context Precision, Context Recall, Faithfulness, Answer Relevancy against a golden set; DeepEval runs the same as a CI gate |

---

## Architecture

```
                          ┌─────────────────────────┐
                          │       User Question      │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   LLM Query Router       │
                          │  (grounded in real table │
                          │   schemas from DuckDB)   │
                          └──────┬──────────┬────────┘
                                 │          │
               ┌─────────────────▼──┐  ┌───▼──────────────────┐
               │   UNSTRUCTURED     │  │    STRUCTURED         │
               │   TEXT PATH        │  │    TABLE PATH         │
               │                    │  │                       │
               │ Query Rewriter     │  │ Text-to-SQL Agent     │
               │       ↓            │  │       ↓               │
               │ Hybrid Retrieval   │  │ Sandboxed DuckDB      │
               │ (BM25 + Vector,    │  │ Execution             │
               │  RRF fusion)       │  │ (read-only, denylisted│
               │       ↓            │  │  row-capped)          │
               │ Cross-Encoder      │  │       ↓               │
               │ Reranker           │  │ Structured Result     │
               └────────┬───────────┘  └───────────┬───────────┘
                        │                           │
                        └────────────┬──────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  Grounded Synthesizer    │
                        │  (refuses below          │
                        │   confidence threshold)  │
                        └────────────┬────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  Answer + [C_n]/[T_n]    │
                        │  Citations               │
                        └─────────────────────────┘
```

### Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **Primary LLM** | Groq — Llama 3.3 70B | Free tier, 14,400 req/day, GPT-4 class quality |
| **Fallback LLM** | Gemini 2.5 Flash | Auto-fallback on Groq rate-limit |
| **Embeddings** | BGE-small-en-v1.5 (local) | Runs on CPU, no API key, no cost per query |
| **Reranker** | BGE-reranker-base (local) | Cross-encoder accuracy, runs locally |
| **Vector Store** | Qdrant (embedded mode) | On-disk, no Docker required for local dev |
| **Keyword Search** | BM25 via rank-bm25 | Catches exact terms embeddings miss |
| **Structured Store** | DuckDB | Zero-setup, embeds in-process, fast analytics |
| **PDF Text** | PyMuPDF | Fast, accurate page-level text extraction |
| **PDF Tables** | Camelot | Grid-aware table extraction, preserves structure |
| **API** | FastAPI | Clean REST layer, auto-generated Swagger docs |
| **UI** | Streamlit | Rapid demo interface |
| **Evaluation** | Ragas + DeepEval | LLM-as-judge metrics + CI regression gate |

---

## Project Structure

```
DocuRoute/
│
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI app — /query, /health, /debug/tables
│
├── data/
│   ├── raw/                     # ← drop your PDFs here
│   │   └── .gitkeep
│   └── processed/               # auto-generated by ingest.py, git-ignored
│       ├── qdrant_storage/      # vector index (on-disk Qdrant)
│       ├── bm25_index.pkl       # serialised BM25 index
│       └── tables.duckdb        # extracted financial tables
│
├── evaluation/
│   ├── results/                 # versioned eval run CSVs, git-ignored
│   ├── golden_set.json          # your ground-truth Q&A pairs ← edit this
│   ├── metrics.py               # Ragas metric wiring (Groq as judge)
│   ├── run_eval.py              # versioned before/after eval runner
│   └── run_deepeval_ci.py       # CI pass/fail gate
│
├── frontend/
│   └── app.py                   # Streamlit demo UI
│
├── generation/
│   ├── __init__.py
│   ├── prompts/
│   │   └── synthesis_prompt.txt # grounded generation system prompt
│   └── synthesizer.py           # grounded answer + citation extraction
│
├── ingestion/
│   ├── __init__.py
│   ├── chunking.py              # recursive overlap-aware text chunking
│   ├── index_builder.py         # Qdrant + DuckDB population
│   └── loaders.py               # PyMuPDF text + Camelot table extraction
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py      # BM25 + vector + RRF fusion
│   ├── query_rewriter.py        # LLM query rewriting + multi-query expansion
│   ├── reranker.py              # cross-encoder rerank + confidence gate
│   └── router.py                # LLM query classifier
│
├── sql_agent/
│   ├── __init__.py
│   ├── executor.py              # sandboxed read-only DuckDB execution
│   └── text_to_sql.py           # NL → SQL grounded in real schema
│
├── tests/
│   ├── test_chunking.py         # chunking unit tests
│   └── test_sql_executor.py     # SQL sandbox security tests
│
├── .env.example                 # environment variable template
├── .github/
│   └── workflows/
│       └── eval-gate.yml        # CI: unit tests + DeepEval regression gate
├── .gitignore
├── config.py                    # single source of truth for all settings
├── docker-compose.yml           # optional: Qdrant server + API + frontend
├── Dockerfile
├── generate_synthetic_pdf.py    # generates aurora_dynamics demo PDF
├── ingest.py                    # CLI: PDFs → vector + BM25 + DuckDB indices
├── llm_client.py                # Groq primary / Gemini fallback LLM wrapper
├── pipeline.py                  # orchestrator tying all components together
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html) — required for PDF table extraction
  - Windows: download and run the installer from the link above
  - macOS: `brew install ghostscript`
  - Linux: `sudo apt-get install ghostscript`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/DocuRoute.git
cd DocuRoute

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install CPU-only PyTorch first (avoids a 4 GB CUDA download on machines without GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Copy the environment template
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

---

## Configuration

Open `.env` and fill in your API keys:

```env
# ── PRIMARY LLM: Groq ────────────────────────────────────────────────────
# Free key (no card needed): https://console.groq.com → API Keys → Create
# Free tier: 14,400 requests/day
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile

# ── FALLBACK LLM: Gemini ─────────────────────────────────────────────────
# Free key: https://aistudio.google.com/apikey
# Used automatically when Groq hits its rate limit
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gemini-2.5-flash

# ── Retrieval tuning (safe to leave as defaults) ──────────────────────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
TOP_K_DENSE=20
TOP_K_SPARSE=20
TOP_K_FUSED=10
TOP_K_FINAL=5

# ── Storage paths ─────────────────────────────────────────────────────────
DUCKDB_PATH=./data/processed/tables.duckdb
QDRANT_STORAGE_PATH=./data/processed/qdrant_storage
```

> **You only need both keys if you want automatic fallback.**
> The system works with just `GROQ_API_KEY`. Gemini is only called when Groq
> returns a 429 rate-limit error.

---

## Usage

### Step 1 — Add your documents

Drop PDF files into `data/raw/`. The system works best with 3–5 years of
annual reports (10-K filings) for a single company. Download free from
[sec.gov/edgar/search](https://www.sec.gov/edgar/search/).

Name them descriptively — the filename (minus extension) becomes the `doc_id`
shown in citations:

```
data/raw/etsy_10k_2023.pdf
data/raw/etsy_10k_2024.pdf
data/raw/etsy_10k_2025.pdf
```

A synthetic demo document (`aurora_dynamics_10k_2025.pdf`) is included so
you can verify the pipeline works before obtaining real filings.

### Step 2 — Run ingestion

```bash
python ingest.py
```

This extracts text and tables from every PDF in `data/raw/`, builds the
vector index, BM25 index, and DuckDB structured tables. Re-run this command
whenever you add or change documents.

Verify what tables were extracted:
```
http://localhost:8000/debug/tables
```

### Step 3 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Check the API is alive:
```bash
curl http://localhost:8000/health
```

Interactive API docs (Swagger UI):
```
http://localhost:8000/docs
```

### Step 4 — Start the UI (optional, separate terminal)

```bash
streamlit run frontend/app.py
```

Open in browser:
```
http://localhost:8501
```

> The Streamlit UI and the Swagger docs both talk to the same FastAPI backend.
> Keep the `uvicorn` terminal running while using either interface.

---

## Scope — what this system can and can't answer

**This is not a general chatbot.** It only answers questions about the PDFs
you have ingested. With Etsy's 2023–2025 10-Ks loaded, it can answer
questions about Etsy's business, risks, and financials from those filings.
It has no access to live stock prices, news, or any document you haven't
ingested.

### Example questions

**Unstructured** — pure narrative retrieval:
```
What are the main risk factors related to competition?
How does management describe Etsy's international growth strategy?
What does the company say about seller trust and safety?
```

**Structured** — SQL against extracted tables:
```
What was total revenue in fiscal year 2024?
What was net income across 2023, 2024, and 2025?
Which year had the highest income from operations?
```

**Both** — number + narrative explanation (strongest routing demo):
```
Why did operating margin change between 2023 and 2024, and by how much?
What drove the change in active buyers, and what was the actual count?
```

---

## Evaluation

DocuRoute measures its own retrieval and generation quality with four
standard RAG metrics computed by Ragas (using the same LLM as judge):

| Metric | Measures |
|---|---|
| **Context Precision** | Of everything retrieved, how much was actually relevant? |
| **Context Recall** | Of everything relevant, how much did we retrieve? |
| **Faithfulness** | Does the answer only state things the retrieved context supports? |
| **Answer Relevancy** | Does the answer actually address the question asked? |

### Run an evaluation

```bash
# Edit evaluation/golden_set.json with real Q&A pairs first
python evaluation/run_eval.py --version baseline_vector_only
```

Results are appended to `evaluation/results/history.csv`. Run again after
pipeline changes to build a before/after comparison table:

| Version | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| baseline_vector_only | — | — | — | — |
| hybrid_rrf | — | — | — | — |
| hybrid_rrf_rerank | — | — | — | — |

### CI regression gate

```bash
python evaluation/run_deepeval_ci.py
```

Returns exit code `0` (pass) or `1` (fail). Wired into
`.github/workflows/eval-gate.yml` — runs on every push to `main` and
blocks merges if Faithfulness or Answer Relevancy drop below threshold.

---

## Design Decisions

| Decision | Reasoning |
|---|---|
| **RRF over score-weighted fusion** | BM25 scores and cosine similarities live on incomparable scales. RRF fuses on rank, not score, sidestepping the calibration problem entirely |
| **Cross-encoder reranker as a separate stage** | Bi-encoder retrieval scores query and document independently. A cross-encoder reads the pair jointly — much more accurate, but too slow to run on the full corpus, so it runs only on the top fused candidates |
| **Open-source embedder + reranker (BGE)** | The pipeline design is the point, not which embedding API you pay for. Runs on CPU, zero cost per query, fully reproducible |
| **Qdrant embedded mode by default** | No Docker required to run locally. `docker-compose.yml` adds a real client-server deployment when needed for a demo |
| **DuckDB read-only + keyword denylist + row cap** | Defense in depth: `read_only=True` is the real security boundary; the regex denylist catches injection attempts early with a clear error; the row cap prevents runaway aggregations from blocking the API |
| **Groq primary / Gemini fallback** | Groq gives 14,400 free requests/day for development. Gemini activates silently on rate-limit. The architecture is provider-agnostic — swapping both is a one-file change in `llm_client.py` |

---

## Roadmap

- [ ] Fine-tune the reranker on the golden set's hard negatives
- [ ] Add multi-hop retrieval for questions requiring two linked lookups
- [ ] Per-component latency tracing via Arize Phoenix surfaced in the UI
- [ ] Query result caching to reduce LLM calls on repeated/similar questions
- [ ] Streaming responses in the Streamlit UI
- [ ] Support for HTML filings directly from EDGAR (skip Print-to-PDF step)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built as a portfolio project demonstrating production ML engineering practices:<br>
hybrid retrieval · agentic routing · grounded generation · automated evaluation
</div>
