# Kestrel Multi-Agent Research Assistant

## Problem and Goal

Kestrel is a local retrieval-augmented research assistant for answering questions from a verified Kestrel knowledge corpus. The goal is to retrieve relevant evidence, check whether it supports or conflicts with the question, and produce a concise cited answer. The corpus contains 154 chunks from 25 documents, and its loader verifies the expected SHA-256 hash before reading it.

## Architecture

The pipeline has exactly four logical agents:

`Planner -> Retriever -> Verifier -> Synthesizer`

- **Planner** classifies the question, detects follow-ups, rewrites follow-ups using recent conversation history, and marks likely multi-hop questions.
- **Retriever** searches a local FAISS vector store and returns the top five chunks with metadata and similarity scores.
- **Verifier** assesses retrieved evidence as `supported`, `partially_supported`, `conflicting_evidence`, or `insufficient_evidence`.
- **Synthesizer** answers from verified evidence and adds inline citations in the form `[chunk_id — title]`.

LangGraph preserves this orchestration as a shared-state graph. The Gradio frontend provides a multi-turn chat, question input, answer, citations, agent status, and retrieved chunk display.

## Retrieval and Generation

Embeddings use the local `sentence-transformers/all-MiniLM-L6-v2` model. FAISS uses normalized vectors with inner-product similarity and a lightweight lexical/entity reranking step. No hosted embedding API is used.

Answer generation uses the Groq model `groq/compound-mini` when `GROQ_API_KEY` is configured. The synthesizer falls back to evidence-based local formatting when generation is unavailable.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file. Do not commit secrets. A matching `.env.example` configuration is:

```dotenv
GROQ_API_KEY=your_groq_api_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=kestrel-multi-agent
```

The corpus must remain at the repository root as `corpus.jsonl`. The loader checks its SHA-256 before loading.

## Run the Gradio App

```bash
python3 -m app.ui.gradio_app
```

The interface supports multi-turn questions by passing conversation history into the Planner. Unsupported questions are reported when the corpus does not establish an answer. Conflicting evidence is surfaced with both source citations and publication/version context.

## Run Evaluation

```bash
python3 evaluation/run_eval.py
```

Evaluation questions preserve conversation IDs and turn order for follow-up cases. The evaluator measures end-to-end latency, records retrieved chunk IDs and citations, stores the verifier verdict, and records a LangSmith run URL when one is available.

Evaluation artifacts:

- `results/eval_questions.jsonl`
- `results/eval_results.jsonl`
- `results/metrics_summary.json`
- `results/improvement.md`

## LangSmith

The complete LangGraph workflow is traced by node: Planner, Retriever, Verifier, and Synthesizer. Retrieval operations and agent inputs/outputs are visible in the LangSmith project `kestrel-multi-agent` when tracing is enabled.

## Limitations

- Retrieval quality depends on the local embedding model, corpus coverage, and reranking heuristics.
- The corpus is static and answers cannot exceed the available evidence.
- Unsupported and conflicting cases require careful verifier behavior; false positives and missed conflicts remain possible.
- The recorded aggregate results show room for improvement: retrieval quality was 0.6778, faithfulness 0.6556, relevance 0.5167, end-to-end correctness 0.4222, citation precision 0.2222, and citation recall 0.5 over the 18-question suite.
- Token usage was not logged in the evaluation results.
