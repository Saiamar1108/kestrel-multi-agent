# Retrieval Ranking Improvement

## Observed Issue
The initial vector-only FAISS retrieval failed to retrieve the required ground-truth evidence chunk `pricing-plans:0` for the single-hop evaluation question:
> *"How many days of queryable event retention does the Starter plan include?"*

Instead, dense vector retrieval returned general policy chunks (`policy-data-retention:0`, `policy-data-retention:1`, `pricing-billing-faq:2`, `policy-data-retention:2`, `spec-trails:3`), omitting `pricing-plans:0` entirely from the top 5 retrieved results.

## Root Cause
1. **Title Exclusion in Indexing**: FAISS vector indexing previously embedded only the raw `chunk.text` without including document title context (`chunk.title`).
2. **Dense Embedding Bias**: The sentence transformer embedding model (`all-MiniLM-L6-v2`) prioritized general data retention policy documents over specific plan pricing specifications due to semantic density.
3. **Lack of Lexical/Entity Reranking**: Standard top-$k=5$ vector retrieval lacked a secondary lexical or proper noun re-scoring mechanism to boost chunks matching specific plan entities (such as "Starter").

## Change Implemented
Implemented a general-purpose, non-hardcoded retrieval enhancement in `app/retrieval/faiss_store.py`:
1. **Title-Aware Indexing**: Embedded `f"{chunk.title}\n{chunk.text}"` during FAISS index building to incorporate document titles directly into vector embeddings.
2. **Top-K Candidate Expansion**: Expanded candidate retrieval from $k=5$ to $k_{candidate} = \max(5k, 25)$ before final selection.
3. **General Reranking**: Implemented a lightweight, general-purpose hybrid reranker (`_rerank_score`) scoring candidates based on non-stopword token overlap, title token matches, and capitalized proper-noun/entity matches.

*Note: No questions, chunk IDs, or specific terms were hardcoded, keeping the solution fully domain-agnostic and generalizable to unseen queries.*

## Before
- **Question ID**: `single-001`
- **Retrieved Chunks**: `['policy-data-retention:0', 'policy-data-retention:1', 'pricing-billing-faq:2', 'policy-data-retention:2', 'spec-trails:3']`
- **Scores**:
  - `expected_chunk_recall`: `0.0`
  - `citation_chunk_recall`: `0.0`

## After
- **Question ID**: `single-001`
- **Retrieved Chunks**: `['policy-data-retention:0', 'policy-data-retention:1', 'pricing-billing-faq:2', 'pricing-plans:0', 'spec-ingest-api:4']`
- **Scores**:
  - `expected_chunk_recall`: `1.0`
  - `citation_chunk_recall`: `1.0`

## Interpretation
The general hybrid candidate retrieval and reranking strategy successfully surfaced `pricing-plans:0` into the top 5 retrieved context chunks (ranking 4th), enabling the synthesizer to generate a direct, accurately cited answer. Both `expected_chunk_recall` and `citation_chunk_recall` for `single-001` increased from 0.0 to 1.0 (100%).

## Limitations
The metrics documented above reflect the specific performance improvement measured on the representative single-hop question `single-001`. They demonstrate candidate recall recovery on entity-specific plan queries and should not be interpreted or presented as an overall system-wide benchmark score across all question types.
