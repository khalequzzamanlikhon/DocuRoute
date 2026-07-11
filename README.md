# Agentic, Query-Routing RAG System

A research-assistant RAG system that routes questions to the right retrieval
strategy — semantic search over document text, or a text-to-SQL agent against
extracted tables — and measures its own retrieval/generation quality with
Ragas + a DeepEval CI gate.

## Architecture

```
User Query
    │
    ▼
Query Router (LLM classifier, grounded in real table schemas)
    │
    ├── unstructured ──► Query Rewriter ──► Hybrid Retrieval (BM25 + Vector, RRF) ──► Reranker ──► Context
    │
    └── structured    ──► Text-to-SQL Agent ──► Sandboxed DuckDB Execution ──► Result
    │
    ▼
Grounded Synthesizer (must cite [C_n]/[T_n], refuses below confidence threshold)
    │
    ▼
Answer + Citations
```

## Why these design choices

| Decision | Reasoning |
|---|---|
| RRF over score-weighted fusion | BM25 and cosine similarity live on incomparable scales; RRF fuses on **rank**, sidestepping the scale mismatch. |
| Cross-encoder rerank as a separate stage | Bi-encoder retrieval scores are computed independently per document; a cross-encoder reads (query, chunk) jointly, which is what actually fixes "similar but not relevant" results. |
| Open-source embedder + reranker (BGE) by default | The pipeline design is the point, not which embedding API you pay for. Swappable via `.env` (`EMBEDDING_MODEL`, `RERANKER_MODEL`). |
| Qdrant in embedded/local mode by default | No Docker required to run the whole thing locally; `docker-compose.yml` is there when you want a real client-server demo. |
| DuckDB, `read_only=True` connection + SQL denylist + row cap | Defense in depth: the read-only flag is the actual security boundary; the regex denylist and row cap catch injection attempts and runaway queries early with clear errors. |
| Ragas for dev-loop metrics, DeepEval for CI | Ragas is fast to iterate with locally; DeepEval turns the same four metrics into an automated pass/fail gate you can put in GitHub Actions. |

## Setup

```bash
git clone <this-repo>
cd agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
```

System dependency for table extraction: `apt-get install ghostscript` (Linux) or `brew install ghostscript` (macOS).

## Running it

```bash
# 1. Drop PDFs into data/raw/, then build the indices
python ingest.py

# 2. Start the API
uvicorn api.main:app --reload --port 8000

# 3. In another terminal, start the demo UI
streamlit run frontend/app.py

# or just curl it:
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What was the average quarterly revenue for the last 3 years?"}'
```

## Building your golden set

`evaluation/golden_set.json` ships with 3 placeholder rows — **replace them and
grow to 30-50 real question/answer pairs before your eval numbers mean
anything.** Write the `ground_truth` yourself by reading the source
documents; don't generate it with the same model you're evaluating, or
you're grading your own homework.

## Scope — what this system can and can't answer

**This is not a general chatbot.** It only knows what's in the PDFs you put in
`data/raw/` and ran through `python ingest.py`. If you ingest Etsy's 2023-2025
10-Ks, it can answer questions about Etsy's business, risks, and financials
from those filings — nothing else. It has no access to today's stock price,
news, or any company/document you haven't ingested.

Every answer is grounded in retrieved text or an executed SQL query, and every
factual sentence carries a `[C_n]`/`[T_n]` citation back to the specific chunk
or query that produced it. If the system can't find enough relevant material,
it says so explicitly rather than guessing — check `/debug/tables` after
ingestion to see exactly what structured data it has to work with.

## Example questions (adapt these to whatever you've ingested)

Once you've skimmed the actual filings, write real questions like these —
mix all three route types so you can demonstrate the router actually works:

**Unstructured** (pure narrative, no tables involved):
- "What are the main risk factors related to competition?"
- "What does management say about seller/buyer trust and safety?"
- "How does the company describe its growth strategy for international markets?"

**Structured** (pure number, answerable by SQL alone):
- "What was total revenue in fiscal year 2024?"
- "What was the year-over-year change in gross merchandise sales?"
- "What was the average quarterly marketing spend across the filings?"

**Both** (needs a number AND the narrative explaining it — the strongest demo of routing):
- "Why did operating margin change between 2023 and 2024, and by how much?"
- "What drove the change in active buyers, and what was the actual number?"

Write 30-50 of these into `evaluation/golden_set.json` with real ground-truth
answers you compute/read yourself — that's what the eval numbers in your
README will be based on.

## Eval workflow (this is the point of the whole project)

```bash
# Run against your current pipeline configuration and tag the result
python evaluation/run_eval.py --version baseline_vector_only

# ... add hybrid search / reranking / query rewriting in the code ...

python evaluation/run_eval.py --version hybrid_rrf_rerank
```

Every run appends a row to `evaluation/results/history.csv`. That table is
the before/after artifact for the README:

| version | context_precision | context_recall | faithfulness | answer_relevancy |
|---|---|---|---|---|
| baseline_vector_only | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| hybrid_rrf | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| hybrid_rrf_rerank | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| + query_rewriting | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

Run `python evaluation/run_deepeval_ci.py` to apply the same metrics as a
hard pass/fail gate — this is what's wired into `.github/workflows/eval-gate.yml`.

## Project structure

```
agentic-rag/
├── config.py                 # single source of truth for all settings
├── llm_client.py              # Gemini wrapper (used by router/rewriter/sql/synthesis)
├── pipeline.py                 # orchestrator: router -> retrieval/sql -> synthesis
├── ingest.py                   # CLI: PDFs -> vector index + BM25 + DuckDB
├── ingestion/
│   ├── loaders.py               # PDF text + table extraction (pymupdf + camelot)
│   ├── chunking.py               # recursive, overlap-aware chunking
│   └── index_builder.py          # Qdrant + DuckDB population
├── retrieval/
│   ├── router.py                  # LLM query classifier (structured/unstructured/both)
│   ├── query_rewriter.py           # query rewriting + multi-query variants
│   ├── hybrid_retriever.py          # BM25 + vector, fused with RRF
│   └── reranker.py                  # cross-encoder rerank + confidence gate
├── sql_agent/
│   ├── text_to_sql.py                # NL -> SQL, grounded in real schema
│   └── executor.py                    # sandboxed, read-only, denylisted execution
├── generation/
│   ├── synthesizer.py                  # grounded generation with [C_n]/[T_n] citations
│   └── prompts/synthesis_prompt.txt
├── evaluation/
│   ├── golden_set.json                  # your ground-truth Q&A set
│   ├── metrics.py                        # Ragas metric wiring
│   ├── run_eval.py                        # versioned before/after eval runs
│   └── run_deepeval_ci.py                  # CI pass/fail gate
├── api/main.py                              # FastAPI /query endpoint
├── frontend/app.py                           # Streamlit demo
├── tests/                                    # pytest — chunking + SQL sandboxing
└── .github/workflows/eval-gate.yml            # CI: unit tests + eval gate
```

## What I'd do with more time

- Fine-tune the reranker on the golden set's hard negatives instead of using it off-the-shelf.
- Add multi-hop retrieval for questions that need chaining two lookups (e.g., "compare this year's biggest risk to last year's").
- Add per-component latency tracing (Arize Phoenix) surfaced in the Streamlit UI, not just in logs.
- Cache query embeddings and rewritten variants (many repeated/similar questions in real usage).
