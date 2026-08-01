# DocuRoute — Project Assessment

> Review date: 2026-08-02 · Reviewed files: all modules in `retrieval/`, `generation/`,
> `ingestion/`, `sql_agent/`, `evaluation/`, `pipeline.py`, `llm_client.py`, `config.py`,
> `api/main.py`, `frontend/app.py`, `tests/`, `first_query_result.md`, README & CI config.

---

## 1. Verdict (TL;DR)

**Yes — this reads like work from a senior ML engineer.** The architecture is genuinely
agentic, safety is defense-in-depth rather than lip service, and the project measures its own
quality with a real eval harness instead of just shipping a demo. It is comfortably above the
average "RAG portfolio project" and demonstrates several practices that working senior
engineers would recognize and respect.

It is **not yet "best-in-class / top-notch"** — and the gap is small but specific. The single
most damning detail: **your own evaluation data shows the default configuration is the
*weaker* one.** `first_query_result.md` proves the reranker *hurts* every metric
(0.32→0.42 precision, 0.77→0.93 faithfulness when skipped), yet `config.py` still defaults to
`retrieval_mode = "hybrid_rrf_rerank"` — the version that measured worst. A top-notch engineer
would have flipped the default the moment the data said so, or fixed the reranker. That
"measure → decide → change" loop is the difference between senior and excellent.

---

## 2. What is already senior-level (with evidence)

| Practice | Where | Why it's senior |
|---|---|---|
| **Production path == evaluated path** | `pipeline.py` is the only orchestrator; both `api/main.py` and both eval scripts call `answer_question()` | You can never "test one thing and ship another." No duplicated logic between serving and eval. |
| **Defense-in-depth SQL safety** | `sql_agent/executor.py`: `read_only=True` connection + regex denylist + multi-statement rejection + 500-row cap + 5s timeout | Read-only mode is the *real* security boundary; the denylist is belt-and-suspenders with better errors. `tests/test_sql_executor.py` even proves a write is physically impossible. |
| **Provider fallback chain** | `llm_client.py`: Groq → OpenRouter → Gemini, 429 detection, JSON-mode with truncation retry (doubled token budget) | Resilience engineering — the system degrades gracefully under rate limits instead of 500ing. |
| **Schema-grounded routing** | `retrieval/router.py` classifies against *actual* DuckDB schemas, not keyword rules | This is the "agentic" claim made real: the LLM sees real columns and decides `unstructured` / `structured` / `both`. |
| **Measurement discipline** | `evaluation/`: golden set (15 Q), versioned `history.csv`, Ragas-style metrics implemented directly, DeepEval CI gate in `.github/workflows/eval-gate.yml` | "We measured it once" vs. "we can't silently regress it" — the CI gate is the senior tell. |
| **Real-world data wrangling** | `ingestion/index_builder.py`: `$`/comma/parens-negative/dash cleaning, multi-line header repair in `loaders.py`, numeric column coercion | Financial PDFs are gnarly; this shows the engineer has actually run into real data, not just tutorials. |
| **Grounded generation with refusal** | `generation/synthesizer.py`: `[C_n]`/`[T_n]` citations, confidence-gate refusal | Auditable answers + honest "I don't know" instead of hallucination. |
| **Why-comments, not what-comments** | Docstrings explain *why* RRF over score fusion, *why* embedded Qdrant, *why* the sigmoid | This is how senior engineers document. |

---

## 3. Honest gaps (what keeps it from "top-notch")

1. **Default config contradicts measured best results.**
   `first_query_result.md` shows `hybrid_rrf` (no reranker) beats `hybrid_rrf_rerank` on all 4
   metrics, but `config.py` still defaults to `hybrid_rrf_rerank`, and the README presents the
   reranker as a selling point. Either flip the default, or fix the reranker (below). Shipping
   the worse configuration by default is the #1 credibility issue.

2. **The confidence gate can starve hybrid questions.**
   In `pipeline.py`, when the reranker's confidence gate fails, `reranked = []` wipes *all*
   text chunks even when a SQL result is present — so a hybrid question gets SQL with zero
   narrative context. `first_query_result.md` §3a documents exactly this failure mode. The gate
   should only refuse when *nothing* passed, never discard one modality because the other's
   scores were low.

3. **`min_rerank_score=0.15` is arbitrary and the sigmoid makes it near-random.**
   Cross-encoder logits for out-of-domain text cluster near 0, so `sigmoid(x)` yields ~0.12–0.5
   and the 0.15 threshold randomly passes/fails. Threshold should be calibrated on the golden
   set or replaced with a relative (top-N) gate.

4. **Eval blind spot: structured questions score 0 on context_recall.**
   `evaluation/run_eval.py` feeds only `reranked_chunks` into `retrieved_contexts` for
   scoring — SQL results are excluded. So table questions are *measured as if retrieval
   failed* even when the SQL answer is correct (see per-question table: q006/q008/q009 got
   recall 0.0 with correct answers). The eval harness needs to include SQL results as
   retrieved context.

5. **Test coverage is thin on the core pipeline.**
   Only `test_chunking.py` and `test_sql_executor.py` exist. Router, rewriter, retriever,
   reranker, synthesizer, and the orchestrator itself have zero tests. Mocked-LLM unit tests
   for these would be the highest-ROI addition.

6. **Dead code & machine-specific hacks.**
   - `ingestion/index_builder_initial.py` is a legacy duplicate of `index_builder.py`.
   - Root `run_eval.py` contains a `sys.path` filter hardcoded to a local "hermes" venv
     (`.venv` shadowing) — that belongs in a local script, not the repo.
   - `EVALUATION_GUIDE.md` has machine-specific absolute paths (`D:\Projects\...\GIT_FILE\DocuRoute`).

7. **Dependency bloat / drift.**
   - `requirements.txt` pins `ragas==0.2.8` + `langchain-google-genai` + `langchain-huggingface`
     + `datasets`, but `evaluation/metrics.py` deliberately implements metrics itself "to avoid
     dependency conflicts." So several heavy deps are effectively unused dead weight.
   - `@app.on_event("startup")` in `api/main.py` is deprecated in modern FastAPI (use `lifespan`).

8. **No lint/type gate.**
   No `ruff`, `mypy`, or `pyproject.toml` configuration, and CI runs only `pytest`. For a
   "production-grade" claim, a lint+type check in the workflow is table stakes.

9. **README + repo hygiene.**
   - ✅ *Fixed in this pass:* `demo.gif` is a healthy 390-frame animation, but the README
     wrapped the image tag in a code fence so it rendered as literal text. Fence removed.
   - ✅ *Fixed in this pass:* README referenced `generate_synthetic_pdf.py` and an included
     `aurora_dynamics_10k_2025.pdf` demo document — neither exists in `data/raw/` (empty).
     Stale claims removed.
   - ⚠️ **Still open:** the MIT badge links to `LICENSE`, but **no LICENSE file exists in the
     repo.** Add one (or the badge and License section point nowhere).

---

## 4. Verdict on "senior ML engineer vs. best ML engineer"

- **Senior ML engineer: yes.** Architecture, safety posture, eval discipline, and
  documentation all clear that bar.
- **"Best ML engineer" (staff / principal tier): not yet.** The distinguishing features of the
  top tier are missing: closing the loop between measurement and configuration, comprehensive
  test coverage of the core logic, operational hardening (observability, caching, streaming),
  and ruthlessly removing dead code. None of these require talent — they require the *discipline
  to finish*.

---

## 5. Top-notch upgrade plan (prioritized)

### P0 — Fix what the data already told you
1. **Flip the default**: set `RETRIEVAL_MODE=hybrid_rrf` as the default (or keep both modes but
   document the measured trade-off honestly in the README).
2. **Fix the confidence gate**: never wipe text chunks when a SQL result exists; refuse only if
   both modalities failed.
3. **Calibrate or replace the reranker**: try `BAAI/bge-reranker-v2-m3`, calibrate
   `min_rerank_score` against the golden set, or use a top-N relative gate instead of an
   absolute sigmoid threshold.

### P1 — Make the eval honest and automatic
4. **Feed SQL results into eval context** so structured questions stop scoring recall 0.
5. **Expand the golden set** to 30–50 questions across 2–3 companies (currently single-company
   Etsy), including multi-hop and refusal test cases.
6. **Auto-run eval in CI on every PR** (not just on push to main) with a diff report vs. the
   last good run — turn the gate from "blocked on push" into "reviewable on PR."

### P2 — Test coverage & engineering rigor
7. **Unit tests with mocked LLM** for router (each route + malformed JSON), rewriter, retriever
   fusion math (RRF on known lists), reranker gate, synthesizer citation extraction, and
   orchestrator routing combinations.
8. **Property-based tests** for chunking (no data loss, overlap invariants).
9. **Add ruff + mypy + pre-commit**, wire into the existing GitHub Action.
10. **Remove dead code**: delete `index_builder_initial.py`, dedupe `run_eval.py`, strip
    machine-specific paths, drop unused deps from `requirements.txt` (or actually use ragas).

### P3 — Operational polish (the "portfolio wow")
11. **Add a LICENSE file** (MIT, since the badge claims it).
12. **Fix README demo.gif** (remove the code fence) and the stale `data/raw` claims.
13. **Observability**: request IDs + structured logging, per-stage latency in
    `PipelineTrace`, and a `/metrics` endpoint — the trace dataclass is already 80% of the way
    there.
14. **Query-result caching** (semantic cache) and **streaming** in the Streamlit UI — both are
    already on your roadmap; they're also the two features demo viewers notice most.
15. **Docker hygiene**: non-root user, healthchecks, and pin the `qdrant/qdrant` image in
    `docker-compose.yml` (already pinned — good).

---

## 6. Scorecard

| Dimension | Grade | Note |
|---|---|---|
| Architecture & modularity | A | Single orchestrator, clean seams |
| Security (SQL sandbox) | A | Best-in-class for a portfolio project |
| Evaluation discipline | B+ | Real harness, CI gate — but a measurement blind spot (SQL context) |
| Data engineering (PDF/table wrangling) | A− | Handles the gnarly real-world cases |
| Test coverage | C | Core pipeline untested |
| Repo/ops hygiene | C+ | Dead code, missing LICENSE, README bugs |
| Documentation | A− | Great "why" docs; README needs the fixes above |
| **Overall** | **B+ → A− (senior, not yet top-tier)** | Close the measure→decide loop and the gaps close fast |

---

*Bottom line: the engineering brain is already there. What's missing is finishing the loop —
let the evaluation data drive the defaults, test the core logic, and clean house. That's a
weekend of work away from "top-notch."*
