# DocuRoute Evaluation Guide

Run the 3 retrieval-mode evaluations to fill the README comparison table.

## Prerequisites

Before you start, confirm:

- [x] `.venv` is activated (`.venv\Scripts\activate`)
- [x] `pip install -r requirements.txt` completed
- [x] Ingestion ran (data/processed has `qdrant_storage`, `bm25_index.pkl`, `tables.duckdb`)
- [x] `GROQ_API_KEY` and `GEMINI_API_KEY` are **both** set in `.env`
      *(metrics.py uses Gemini as the Ragas judge — it WILL fail without it)*
- [x] `evaluation/golden_set.json` is populated (it already has 15 Etsy questions)

> **No need to keep uvicorn or Streamlit running.** The eval script imports the pipeline directly.

---

## How the 3 versions differ

The code now respects a `RETRIEVAL_MODE` env var (set in `config.py`):

| Version | BM25 keyword | Dense vector | RRF fusion | Cross-encoder reranker |
|---|---|---|---|---|
| `baseline_vector_only` | no | yes | yes | no |
| `hybrid_rrf` | yes | yes | yes | no |
| `hybrid_rrf_rerank` | yes | yes | yes | yes |

---

## Step 1 — Run `hybrid_rrf_rerank` (full pipeline)

This is the default — no env var needed:

```bash
cd D:\Projects\projects_july_26\GIT_FILE\DocuRoute
.venv\Scripts\python run_eval.py --version hybrid_rrf_rerank
```

>  **Important**: Always use `.venv\Scripts\python` instead of just `python` — Hermes injects its own venv into `$PATH` and shadows the project's packages.
>
>  **Groq rate limits**: The free Groq tier rate-limits heavily (429s with auto-retry). If you see `Retrying request...` messages, that's normal — the script handles them automatically. Each question takes ~5–15 seconds. All 15 questions take **3–8 minutes**.

You'll see logging for each question:

```
INFO:evaluation.run_eval:Answered q001 (route=unstructured, refused=False)
...
INFO:evaluation.run_eval:Answered q015 (route=structured, refused=False)
```

When done, it prints a summary and saves to `evaluation/results/history.csv`.

---

## Step 2 — Run `hybrid_rrf` (no reranker)

Set the env var and re-run:

```bash
export RETRIEVAL_MODE=hybrid_rrf
.venv\Scripts\python run_eval.py --version hybrid_rrf
```

---

## Step 3 — Run `baseline_vector_only` (dense only)

```bash
export RETRIEVAL_MODE=baseline_vector_only
.venv\Scripts\python run_eval.py --version baseline_vector_only
```

---

## Step 4 — View the accumulated results

```bash
cat evaluation/results/history.csv
```

You'll see three rows — one per version — with the four metric columns. Example output:

```
version,timestamp,n_questions,context_precision,context_recall,faithfulness,answer_relevancy
hybrid_rrf_rerank,2026-07-20T10:00:00,15,0.842,0.917,0.887,0.764
hybrid_rrf,2026-07-20T10:05:00,15,0.801,0.883,0.852,0.739
baseline_vector_only,2026-07-20T10:10:00,15,0.724,0.806,0.795,0.681
```

---

## Step 5 — Fill the README table

Open `README.md` and update the Evaluation section:

```markdown
| Version | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| baseline_vector_only | 0.724 | 0.806 | 0.795 | 0.681 |
| hybrid_rrf | 0.801 | 0.883 | 0.852 | 0.739 |
| hybrid_rrf_rerank | 0.842 | 0.917 | 0.887 | 0.764 |
```

*(Replace with your actual numbers.)*

---

## Troubleshooting

| Problem | Likely fix |
|---|---|
| `google.api_core.exceptions.PermissionDenied` | `GEMINI_API_KEY` is missing or invalid in `.env` — Ragas uses Gemini as the judge |
| `groq.RateLimitError` | The pipeline hit Groq's rate limit; it should auto-fallback to Gemini. Try again later or set `GROQ_API_KEY` to a key with a higher quota |
| `ModuleNotFoundError: No module named 'ragas'` | Run `pip install -r requirements.txt` in your activated `.venv` |
| `RuntimeError: BM25 index not built` | Re-run `python ingest.py` — the BM25 pickle might be missing or corrupt |
| Scores are identical across versions | Make sure `export RETRIEVAL_MODE=` is set **before** each run, not in the same command line as the python call |

---

## (Optional) Reset for future runs

Move or delete the history file if you want a clean slate:

```bash
mv evaluation/results/history.csv evaluation/results/history_old.csv
```
