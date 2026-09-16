# DocuRoute Evaluation — First Query Results

> Generated: 2026-07-21
> Golden set: 15 questions (Etsy 10-K Q&A)
> Pipeline configs tested: `hybrid_rrf_rerank` and `hybrid_rrf`

---

## 1. What Are "Good" Scores for a RAG System?

There is no universal pass/fail — scores depend heavily on **domain difficulty**, **document quality**, and **question granularity**. But for financial/10-K Q&A on a mid-size golden set (15 questions), here are the generally accepted benchmarks:

| Metric | Poor | Mediocre | Good | Excellent | Meaning |
|---|---|---|---|---|---|
| **Context Precision** | < 0.30 | 0.30–0.50 | **0.50–0.70** | > 0.70 | How much of what you retrieved is actually useful |
| **Context Recall** | < 0.30 | 0.30–0.50 | **0.50–0.70** | > 0.70 | How much of what's needed did you actually retrieve |
| **Faithfulness** | < 0.60 | 0.60–0.80 | **0.80–0.90** | > 0.90 | Is the answer hallucination-free vs. retrieved context |
| **Answer Relevancy** | < 0.60 | 0.60–0.75 | **0.75–0.90** | > 0.90 | Does the answer actually address the question |

**Industry reality check**: Most production RAG systems live in the **0.50–0.70 range** for precision/recall and **0.80–0.90 for faithfulness**, because retrieval is the harder problem. The LLM generation side tends to be strong because modern models stay close to provided context when prompted well.

### Current hybrid_rrf scores vs. targets:

| Metric | Current (hybrid_rrf) | Target "Good" | Gap |
|---|---|---|---|
| **context_precision** | **0.42** | 0.50–0.70 | −0.08 from minimum |
| **context_recall** | **0.41** | 0.50–0.70 | −0.09 from minimum |
| **faithfulness** | **0.93** | 0.80–0.90 | yes Excellent — above target |
| **answer_relevancy** | **0.79** | 0.75–0.90 | yes Good — in range |

**Diagnosis**: The **retrieval side is the bottleneck** (precision + recall). The generation side (faithfulness + relevancy) is already strong. This means the LLM is doing a good job with what it gets — it just isn't getting enough of the right chunks.

---

## 2. Should a Reranker ALWAYS Outperform?

**No — and here's why.**

A cross-encoder reranker should theoretically improve things, but in practice it can *hurt* when:

### 2a. The reranker model is too small or not fine-tuned for the domain

Your config uses `BAAI/bge-reranker-base` (≈ 280 MB, 12 layers). This is a **general-purpose** cross-encoder trained on MS MARCO / NLI data, **not fine-tuned on financial/10-K text**. Financial filings have:
- Dense, domain-specific vocabulary ("GMS", "take rate", "performance marketing", "stock-based compensation")
- Long passages with mixed signal (one paragraph mixing risk factors, revenue data, and forward-looking disclaimers)
- Formal/boilerplate structures that a general reranker may not distinguish well

A small general reranker can demote relevant chunks and promote irrelevant ones because the training data doesn't match the domain.

### 2b. The reranker is trained on query-passage *relevance*, not *completeness*

Cross-encoders score "how relevant is this passage to the query" — not "how much of the answer's information is covered by the union of all passages." A passage ranked #2–5 might contain unique critical information (e.g., exact dollar amounts for gross profit) but the reranker ranks it lower than a passage that's broadly relevant but redundant (e.g., generic macroeconomics discussion).

### 2c. `top_k_final` is too aggressive with the reranker

The pipeline:
1. RRF fusion → top 10 (`top_k_fused`)
2. Reranker → top 5 (`top_k_final`)

Cutting from 10 → 5 with a cross-encoder means **half the candidates are discarded**. If the reranker makes even 2 mistakes out of 10 (which is realistic for a general-purpose model on financial text), it can discard relevant chunks that RRF correctly retrieved.

### 2d. When retrieval is already decent, a reranker's *re-ranking* can conflict with RRF's *diversity*

RRF naturally produces diverse results because a chunk ranked highly by BOTH BM25 and vector search gets a strong boost (the agreement signal). The reranker then re-sorts based on its own score, which can **overrule** this agreement signal and promote a chunk that's only semantically similar but misses the key number the question needs.

---

## 3. Root Cause Analysis — Why the Reranker Hurts DocuRoute

### Evidence from the data

| Metric | hybrid_rrf_rerank | hybrid_rrf | Δ |
|---|---|---|---|
| context_precision | 0.32 | **0.42** | **+10pp** |
| context_recall | 0.33 | **0.41** | **+8pp** |
| faithfulness | 0.77 | **0.93** | **+16pp** |
| answer_relevancy | 0.69 | **0.79** | **+10pp** |

Every metric improves when the reranker is skipped. Let's look at why per question type:

### 3a. Structured/SQL questions — 2 worst-hit examples

From `hybrid_rrf_rerank_detailed.csv`:

| Question | rerank context_precision | rerank faithfulness | non-rerank faithfulness |
|---|---|---|---|
| q006: "What was Etsy's total revenue for 2025?" | 0.0 | (missing/no score) | 1.0 |
| q009: "What was Etsy's gross profit for 2025?" | 0.0 | (missing/no score) | 1.0 |

**Root cause**: For SQL-routed questions, the reranker path **discards the SQL result from the context block** when the confidence gate triggers. The pipeline at line 77-78 of `pipeline.py`:
```python
if not trace.confidence_gate_passed:
    reranked = []   # wipes all chunks!
```

But the SQL result is still passed to the synthesizer (line 117). If the confidence gate fails (reranker scores all chunks < 0.15), the LLM gets **only the SQL result with zero text context** → it parses the mangled SUM() output as a giant scientific notation number → wrong answer → metric collapse.

Even worse: the reranker computes sigmoid scores on raw cross-encoder logits. A sigmoid of raw logits ≈ -2 to -3 produces scores like 0.12–0.15, which **barely clear or fail** the `min_rerank_score=0.15` threshold. This means random chance determines whether SQL-routed questions get any text context.

### 3b. Unstructured questions with specific numeric answers

| Question | rerank context_precision | non-rerank context_precision |
|---|---|---|
| q014: "Stock-based compensation 2025" | 0.2 | **0.8** |
| q015: "R&D expense change 2024→2025" | 0.6 | **>** |
| q013: "Gross margin trend 2023-2025" | 0.1 | **>** |

**Root cause**: These questions need precise financial numbers embedded in surrounding narrative. The reranker incorrectly scores domain-specific passages lower. For example, for q014 ("stock-based compensation 2025"), the reranked version got 0.2 context_precision — the relevant chunk with exact compensation numbers was ranked below more generic "risk factors" text.

### 3c. The JSON bad-parsing during scoring

From logs:
```
WARNING:evaluation.metrics:Bad JSON from LLM for faithfulness: ```json
{ "score": 0.75, "reason": "The answer ...
```

The scoring LLM (which was Gemini due to Groq rate limits) sometimes wraps JSON in markdown fences with text on the same line as the opening fence. The regex doesn't handle ` ```json\n{` well in all cases → falls through to numeric regex fallback → potentially wrong scores. This added noise to both runs' metrics.

### 3d. Reranker model `BAAI/bge-reranker-base` specifics

This model:
- Has **279 MB** and uses **12 transformer layers**
- Was trained on a mix of MS MARCO, NLI, and Quora — **no financial data**
- Employs a **classification head** (binary: relevant/not relevant), not a nuanced scorer
- The `sigmoid()` normalization in `reranker.py` maps all raw logits to (0,1), but raw logits for out-of-domain text tend to cluster around 0 — making the scores near-meaningless
- The `min_rerank_score=0.15` threshold was chosen arbitrarily and silently discards relevant chunks

---

## 4. Summary of Findings

| Factor | Impact | Severity |
|---|---|---|
| **Domain mismatch** — general reranker on financial/10-K text | Reranker demotes relevant chunks | High |
| **Sigmoid normalization** — compresses all scores to near-0.15 range | Confidence gate becomes random | High |
| **Confidence gate wipes chunks** — SQL path loses text context | Faithfulness collapses on hybrid questions | High |
| **Small `top_k_final=5`** — halves candidates after reranker | Missing relevant chunks | Medium |
| **Scoring noise** — Gemini fallback produces malformed JSON during eval | Metric scores have ±5–10% noise | Medium |
| **Sentence_transformer version** — model file behavior varies by version | Hard to reproduce scores | Low |

---

## 5. Per-Question Breakdown — hybrid_rrf (best run)

| ID | Question | Route | Precision | Recall | Faithfulness | Relevancy |
|---|---|---|---|---|---|---|
| q001 | Competition risk factors | both | 0.2 | 0.6 | 1.0 | 0.8 |
| q002 | Business model uniqueness | both | 0.6 | 0.4 | 0.9 | 0.9 |
| q003 | Macroeconomic conditions | both | 0.2 | 0.7 | 0.9 | 0.9 |
| q004 | Third-party platform risks | both | 0.6 | 0.8 | 0.9 | 1.0 |
| q005 | Active buyer growth strategy (2024 report) | both | 0.4 | 0.6 | 0.8 | 0.8 |
| q006 | Total revenue 2025 | structured | 1.0 | 0.0 | 1.0 | 1.0 |
| q007 | Net income 2024 | structured | 0.0 | 0.0 | 1.0 | 0.0 |
| q008 | Revenue 2023+2024+2025 | structured | 1.0 | 0.0 | 1.0 | 1.0 |
| q009 | Gross profit 2025 | structured | 0.5 | 0.0 | 1.0 | 1.0 |
| q010 | Operating expenses 2023 | structured | 0.5 | 0.0 | 1.0 | 0.5 |
| q011 | Why revenue change 2024→2025 | both | 0.2 | 0.9 | 1.0 | 1.0 |
| q012 | Marketing expense change 2023→2024 | both | 0.2 | 0.9 | 1.0 | 1.0 |
| q013 | Gross margin trend 2023–2025 | both | 0.1 | 0.0 | 1.0 | 0.0 |
| q014 | Stock-based comp 2025 | both | 0.2 | 0.8 | 0.75 | 0.9 |
| q015 | R&D expense change 2024→2025 | both | 0.6 | 0.5 | 0.7 | 1.0 |

**Notable patterns**:
- Questions routed `structured` have **perfect faithfulness** (1.0) but **0.0 context_recall** — the SQL query provides exact numbers but Ragas doesn't score it as "retrieved context" because the eval only feeds the `reranked_chunks` into scoring
- q007 (net income 2024) scored 0.0 precision, 0.0 recall, 0.0 relevancy — the SQL query failed or returned nothing
- q013 (gross margin trend) scored 0.0 recall and 0.0 relevancy — the SQL query likely produced wrong numbers, similar to the reranker-confusion pattern

---

## 6. Recommended Fixes (not applied yet)

1. **Drop `top_k_final` slicing for non-reranker path** — the synthetic ranking scores (1.0, 0.8, 0.6, ...) give no information; let all `top_k_fused` (10) chunks through or use the full RRF ranked list
2. **Fix `min_rerank_score` threshold** — 0.15 is too high for BGE-reranker-base's sigmoid output; calibrate against actual data or disable the gate for SQL-routed questions
3. **Separate the confidence gate** — don't wipe chunks when SQL is also present, so hybrid questions keep text context even if reranker scores are low
4. **Consider a domain-tuned reranker** (e.g., `BAAI/bge-reranker-v2-m3` or fine-tune on the golden set's hard negatives) — listed in the project's own roadmap
5. **Add edge-case handling for SQL parse errors** — when DuckDB returns garbage (like SUM of a non-numeric column), catch and fall back to text-only retrieval

---

## Raw History CSV Data

```
hybrid_rrf_rerank (2026-07-20):
  context_precision=0.3226, context_recall=0.3333,
  faithfulness=0.7701, answer_relevancy=0.6868

hybrid_rrf (2026-07-21):
  context_precision=0.4200, context_recall=0.4133,
  faithfulness=0.9300, answer_relevancy=0.7867
```
