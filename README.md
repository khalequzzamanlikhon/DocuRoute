# DocuRoute

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Gemini-orange?style=flat)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

I built DocuRoute to answer questions about company annual reports (SEC 10-K
filings). Some questions need a paragraph from the text ("what are the main risk
factors?"); others need a number from a table ("what was revenue in 2024?"). So an
LLM first decides which kind of question it is, then sends it either to text
retrieval or to a text-to-SQL agent over the tables, and every answer cites where it
came from.

I also wanted to know which retrieval setup actually works, so I built an evaluation
harness and measured it. The most useful result: adding a general-purpose reranker,
which is supposed to help, **lowered faithfulness from 0.93 to 0.77** on these
filings. So I made plain hybrid retrieval the default ([Evaluation](#evaluation)).

<p align="center">
  <img src="demo.gif" alt="DocuRoute demo" width="55%" style="max-width: 650px;">
</p>

## How it works

| Part | What I did |
|---|---|
| **Routing** | An LLM labels each question `unstructured`, `structured` or `both`. It sees the real table schemas found at ingest time, so it isn't guessing from keywords. |
| **Text retrieval** | BM25 keyword search plus dense search (BGE embeddings), merged with Reciprocal Rank Fusion. Keywords catch exact terms; embeddings catch meaning. A cross-encoder reranker is available but off by default. |
| **Table questions** | A text-to-SQL agent queries the tables I extract from the PDFs. It runs on read-only DuckDB with a keyword denylist and a row cap, so a bad query can't change or dump the data. |
| **Cited answers** | Every factual sentence carries a `[C_n]` (text chunk) or `[T_n]` (SQL result) citation. If retrieval confidence is too low, it refuses instead of guessing. |
| **Provider fallback** | Groq first, then OpenRouter, then Gemini, when one hits its rate limit. |
| **Evaluation** | Ragas-style metrics I implemented myself (context precision, context recall, faithfulness, answer relevancy), scored by an LLM judge against a 15-question golden set. DeepEval runs the same checks as a CI gate. |

```
                  question
                     │
            LLM router (sees the real table schemas)
              │                         │
      text path                     table path
  query rewrite                  text-to-SQL agent
  BM25 + dense, RRF              read-only DuckDB
  (optional reranker)            (denylist, row cap)
              │                         │
              └──────── answer writer ──┘
                  (refuses below a confidence threshold)
                              │
                  answer with [C_n] / [T_n] citations
```

### Stack

| Layer | Tool | Why I chose it |
|---|---|---|
| Main LLM | Groq, Llama 3.3 70B | Free tier with a high daily limit |
| Backup LLMs | OpenRouter (Llama 3.3 70B), Gemini 2.5 Flash | Used only when the one before hits its rate limit |
| Embeddings | BGE-large-en-v1.5, local | Runs on CPU, no API key, no cost per query |
| Reranker | BGE-reranker-base, local | Optional; see the evaluation |
| Vector store | Qdrant, embedded | On disk, no Docker needed for local use |
| Keyword search | rank-bm25 | Catches exact terms that embeddings miss |
| Tables | DuckDB | In-process and fast for analytics |
| PDF text / tables | PyMuPDF / Camelot | Page-level text; grid-aware table extraction |
| API / UI | FastAPI / Streamlit | REST API with Swagger docs; a simple demo UI |
| Evaluation | My Ragas-style metrics + DeepEval | LLM-judge metrics and a CI regression gate |

## Quick Start

### Prerequisites

- Python 3.11+
- [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html), needed for PDF table extraction
  - Windows: download and run the installer from the link above
  - macOS: `brew install ghostscript`
  - Linux: `sudo apt-get install ghostscript`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/khalequzzamanlikhon/DocuRoute.git
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

# ── SECONDARY LLM: OpenRouter (optional, auto-fallback on Groq rate-limit) ──
# Free key: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── FALLBACK LLM: Gemini (optional) ──────────────────────────────────────
# Free key: https://aistudio.google.com/apikey
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gemini-2.5-flash

# ── Retrieval tuning (safe to leave as defaults) ──────────────────────────
# hybrid_rrf is the default: it measured better than the reranker on this
# corpus (see the Evaluation section). Opt into the reranker with:
# RETRIEVAL_MODE=hybrid_rrf_rerank
RETRIEVAL_MODE=hybrid_rrf
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
TOP_K_DENSE=20
TOP_K_SPARSE=20
TOP_K_FUSED=10
TOP_K_FINAL=5
# Confidence gate (raw cross-encoder logits, see retrieval/reranker.py)
MIN_RERANK_LOGIT=0.0
MIN_RERANK_MARGIN=0.5

# ── Semantic query cache (API layer only; eval always measures cold runs) ──
CACHE_ENABLED=true
CACHE_MAX_ENTRIES=256
CACHE_SIMILARITY=0.97

# ── Storage paths ─────────────────────────────────────────────────────────
DUCKDB_PATH=./data/processed/tables.duckdb
QDRANT_STORAGE_PATH=./data/processed/qdrant_storage
```

> **You only need `GROQ_API_KEY` to run.** OpenRouter and Gemini keys are
> optional; they're used automatically only when Groq hits its rate limit.

---

## Usage

### Step 1: Add your documents

Drop PDF files into `data/raw/`. It works best with 3–5 years of
annual reports (10-K filings) for a single company. Download free from
[sec.gov/edgar/search](https://www.sec.gov/edgar/search/).

Name them descriptively: the filename (minus extension) becomes the `doc_id`
shown in citations:

```
data/raw/etsy_10k_2023.pdf
data/raw/etsy_10k_2024.pdf
data/raw/etsy_10k_2025.pdf
```

### Step 2: Run ingestion

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

### Step 3: Start the API

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

Other endpoints:

| Endpoint | Description |
|---|---|
| `POST /query` | Synchronous answer (JSON) |
| `POST /query/stream` | Server-Sent Events: token-by-token answer, then the final JSON |
| `GET /health` | Liveness + whether the index is loaded |
| `GET /metrics` | In-process counters: requests, refusals, per-route, latencies, cache stats |
| `GET /debug/tables` | Every table extracted into DuckDB, with column types |

### Step 4: Start the UI (optional, separate terminal)

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

## What it can and can't answer

It only answers questions about the PDFs you ingest; it isn't a general chatbot. With
Etsy's 2023–2025 10-Ks loaded, it can answer questions about Etsy's business, risks
and financials from those filings. It has no live stock prices, news, or anything
you haven't ingested.

**Text questions**
```
What are the main risk factors related to competition?
How does management describe Etsy's international growth strategy?
What does the company say about seller trust and safety?
```

**Table questions** (SQL over the extracted tables)
```
What was total revenue in fiscal year 2024?
What was net income across 2023, 2024, and 2025?
Which year had the highest income from operations?
```

**Both** (a number plus the explanation behind it)
```
Why did operating margin change between 2023 and 2024, and by how much?
What drove the change in active buyers, and what was the actual count?
```

## Evaluation

I score the system against the golden set in `evaluation/golden_set.json` with four
metrics: context precision, context recall, faithfulness and answer relevancy.

```bash
# put real question/answer pairs in evaluation/golden_set.json first
python run_eval.py --version hybrid_rrf   # --version sets RETRIEVAL_MODE
```

Each run is appended to `evaluation/results/history.csv`, so I can compare before and
after a change. These are my results on the original 15-question Etsy set (before I
added the refusal test cases); full detail is in `first_query_result.md`:

| Metric | hybrid RRF (**default**) | hybrid RRF + reranker | Change |
|---|---|---|---|
| context precision | **0.42** | 0.32 | −0.10 |
| context recall | **0.41** | 0.33 | −0.08 |
| faithfulness | **0.93** | 0.77 | −0.16 |
| answer relevancy | **0.79** | 0.69 | −0.10 |

**What I took from this:** the reranker (BGE-reranker-base) made every metric worse
on 10-K text, likely because it was trained on general web text, not financial
filings. So I made plain hybrid RRF the default and kept the reranker as an option
(`RETRIEVAL_MODE=hybrid_rrf_rerank`). I also rewrote the confidence gate: the old
one used an absolute threshold on sigmoid scores, which was close to random, so the
new one uses raw logits with a relative margin.

The main limit is size: 15 questions from one company's filings is enough to show a
large drop like this one, not to rank small differences.

### CI gate

```bash
python evaluation/run_deepeval_ci.py
```

It exits with `0` (pass) or `1` (fail). `.github/workflows/eval-gate.yml` runs it on
every push to `main` and blocks merges if faithfulness or answer relevancy drops below
the threshold.

## License

MIT, see [LICENSE](LICENSE).
