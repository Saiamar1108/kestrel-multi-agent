# Reflection

## What Worked

The fixed four-agent graph gives the system a clear handoff structure: planning produces a query, retrieval produces ranked evidence, verification classifies the evidence, and synthesis produces a cited response. The shared LangGraph state also supports multi-turn follow-up questions by carrying recent conversation messages into the Planner.

The corpus loader provides an important integrity boundary. It validates JSONL records and checks the expected SHA-256 before loading. The final corpus count is 154 chunks across 25 documents.

Local embeddings and FAISS keep retrieval self-contained. LangSmith tracing was added around the complete workflow so the four logical stages and retrieval operation can be inspected in the `kestrel-multi-agent` project.

## Observed Retrieval Problem

Initial vector-only retrieval missed the expected `pricing-plans:0` chunk for the question about Starter queryable retention. The initial top five contained general retention and billing material instead, and the measured `expected_chunk_recall` and `citation_chunk_recall` were both 0.0 for `single-001`.

The documented root causes were that indexing omitted document titles, dense embeddings favored general retention language, and vector-only top-k retrieval had no lexical or entity-aware reranking.

## Retrieval Improvement

The retrieval store was improved without hardcoding evaluation questions or chunk IDs. It now embeds the document title with the text, retrieves an expanded candidate set, and applies general reranking based on non-stopword token overlap, title token matches, and capitalized proper-noun/entity matches.

For `single-001`, the measured result changed to `expected_chunk_recall: 1.0` and `citation_chunk_recall: 1.0`. The expected `pricing-plans:0` chunk appeared in the final top five at rank four.

## Before and After Measured Results

The before/after values below are the recorded measurements for `single-001`, not a system-wide benchmark:

| Measurement | Before | After |
|---|---:|---:|
| Expected chunk recall | 0.0 | 1.0 |
| Citation chunk recall | 0.0 | 1.0 |

The full 18-question aggregate results are recorded separately in `results/metrics_summary.json`. The aggregate values are retrieval quality 0.6778, faithfulness 0.6556, relevance 0.5167, end-to-end correctness 0.4222, citation precision 0.2222, and citation recall 0.5.

## Limitations

The single-question retrieval improvement does not establish uniform performance across question types. The full evaluation shows that citation precision and end-to-end correctness remain lower than desired. Multi-hop retrieval is also imperfect, and unsupported questions can still retrieve related but non-answering material. The current evaluation does not include token usage because it was not logged.

The system depends on the static corpus and local model behavior. Reranking is general-purpose but heuristic, so it can still prefer a related chunk over the most authoritative chunk. Verifier and synthesis quality also depend on the retrieved evidence and, when configured, the Groq generation response.

## What Would Be Improved With More Time

- Improve proposition-level verification and conflict detection with broader tests for complementary and contradictory evidence.
- Add stronger query decomposition for multi-hop questions so each required document or claim is retrieved reliably.
- Improve citation selection so cited chunks more consistently match expected evidence.
- Add reproducible evaluation logging for model/token usage and per-question trace metadata.
- Expand tests for follow-up history, unsupported questions, conflict recency, and concise final-answer formatting.
