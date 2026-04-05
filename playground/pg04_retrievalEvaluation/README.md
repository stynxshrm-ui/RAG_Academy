
## Hybrid Search Tuning

```
python -m pg04_retrievalEvaluation.hybrid_search_tuning
```
### Which chunk size performed better for each query?

Chunk Size = 200
=============================================================================
**Query: What are the penalties for falls?**

Hybrid search results:
  [1] Score: 1.000 - LTIES FOR FALLS In ice dance, when skaters fall during competition, they lose points. A fall by one ...
      ✓ Contains penalty/fall info
  [2] Score: 0.465 -  lose 2 points.  INTERRUPTION PENALTIES If a performance is interrupted, penalties apply based on du...
      ✓ Contains penalty/fall info

**Query: When did ice dance become Olympic?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
      ✓ Contains Olympic date info
  [2] Score: 0.562 - ISTORICAL) Before 2010, the compulsory dance was the first segment. Teams performed the same pattern...


Chunk Size = 500
================================================================================
**Query: What are the penalties for falls?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
  [2] Score: 0.914 - LTIES FOR FALLS In ice dance, when skaters fall during competition, they lose points. A fall by one ...

**Query: When did ice dance become Olympic?**

Hybrid search results:
  [1] Score: 1.000 - ICE DANCE - INTRODUCTION Ice dance is a discipline of figure skating that draws from ballroom dancin...
      ✓ Contains Olympic date info
  [2] Score: 0.607 - ISTORICAL) Before 2010, the compulsory dance was the first segment. Teams performed the same pattern...


RECOMMENDATION
================================================================================

200 char chunks are BETTER for specific factual queries
   (Correctly retrieved penalty information)

    - Use 200 char chunks for: Specific questions, factual lookups
    - Use 500 char chunks for: Broad overview, conceptual questions

## Reranking Results: Cross-Encoder Performance
```
python -m pg04_retrievalEvaluation.reranking
```

### Hybrid Search vs Cross-Encoder Reranking


| Aspect | Hybrid Search | Cross-Encoder Reranking |
|--------|--------------|------------------------|
| **Role** | First-pass retrieval | Second-pass re-ranking |
| **Speed** | Fast (~10-50ms) | Slow (~50-200ms) |
| **Scalability** | Millions of docs | Top 10-20 candidates only |
| **How it works** | Combines sparse + dense vectors | Processes query+chunk together |
| **Score range** | 0 to 1 (similarity) | -∞ to +∞ (relevance logits) |


### Hybrid Search (Current Implementation)
```python
# Combines two fast methods
sparse_score = TF-IDF(query, chunk)      # Keyword matching
dense_score = cosine_similarity(query_emb, chunk_emb)  # Semantic matching
final_score = 0.3*sparse + 0.7*dense
```
- **Bi-encoder architecture**: Query and chunk encoded separately
- **Pre-computed**: Chunk embeddings stored in advance
- **Fast**: O(n) where n = total chunks

### Cross-Encoder Reranking
```python
# Processes query and chunk TOGETHER
input = f"[CLS] {query} [SEP] {chunk} [SEP]"
score = cross_encoder_model(input)  # Single forward pass
```
- **Cross-encoder architecture**: Query and chunk processed together
- **Cannot pre-compute**: Must process each query-chunk pair
- **Slow**: O(k) where k = candidates (usually 10-100)


| Aspect | Hybrid Search | Cross-Encoder Reranking |
|--------|--------------|------------------------|
| **Stage** | First-pass retrieval | Second-pass re-ranking |
| **Purpose** | Fast candidate retrieval | Precise relevance scoring |
| **Speed** | Very fast (~10-50ms) | Slow (~50-200ms per query) |
| **Scalability** | Can search millions of docs | Only works on top-k candidates |
| **How it works** | Combines sparse + dense vectors | Processes query+chunk together |

## Visual Pipeline

```
Query: "What happens when skaters fall?"
                    │
                    ▼
         ┌─────────────────────┐
         │   HYBRID SEARCH     │  ← Fast, retrieves 10-20 candidates
         │ (First-pass)        │
         └─────────────────────┘
                    │
                    ▼
         Retrieved 10 candidates (scores ~0.3-1.0)
                    │
                    ▼
         ┌─────────────────────┐
         │ CROSS-ENCODER       │  ← Slow, reranks top candidates
         │ (Second-pass)       │
         └─────────────────────┘
                    │
                    ▼
         Reranked top 3 candidates (scores ~ -10 to +10)
```

### Test Query: "What happens when skaters fall?"

| Rank | Without Reranking (Hybrid Only) | Score | With Reranking (Cross-Encoder) | Score |
|------|--------------------------------|-------|-------------------------------|-------|
| 1 | PENALTIES FOR FALLS | 1.000 | PENALTIES FOR FALLS | 7.322 |
| 2 | ICE DANCE - INTRODUCTION | 0.767 | ICE DANCE - INTRODUCTION | 1.294 |
| 3 | Historical section | 0.255 | INTERRUPTION PENALTIES | -9.461 |

### Key Findings

**Score Interpretation:**
- **Hybrid search scores**: 0-1 range (cosine similarity)
- **Cross-encoder scores**: Raw logits (can be negative to positive)

**Observed Improvements:**

| Aspect | Before Reranking | After Reranking |
|--------|------------------|-----------------|
| Relevant chunk score | 1.000 | 7.322 |
| Irrelevant chunk score | 0.255 | -9.461 |
| Score spread | 0.745 | 16.783 |

**What the reranker understood:**
- "PENALTIES FOR FALLS" = highly relevant (+7.32)
- "INTERRUPTION PENALTIES" = NOT relevant to falls (-9.46) **NEGATIVE SCORE**

The cross-encoder successfully distinguished between semantically similar but distinct concepts ("falls" vs "interruptions").

**Why it matters:** Cross-encoder created 16.8x score spread between relevant and irrelevant chunks, clearly distinguishing "falls" from semantically similar "interruptions."

**Rule of thumb:** Use hybrid for fast retrieval (millions of docs), cross-encoder for precise ranking (top 10-20 candidates).

### Score Normalization (Optional)

Cross-encoder logits can be converted to 0-1 probabilities using softmax

## RAGAS Evaluation Results (Ollama + MISTRAL:LATEST)

```
python -m pg04_retrievalEvaluation.evaluate_ragas_ollama
```
| Metric | Score | Grade |
|--------|-------|-------|
| Faithfulness | 0.917 | Excellent |
| Answer Relevancy | 0.599 | Moderate |

**Interpretation:**
- **Faithfulness 0.917**: Answers are highly factual with minimal hallucination
- **Answer Relevancy 0.599**: Answers are correct but sometimes verbose or include extra context

**Improvements planned:**
- Prompt engineering for conciseness
- Reduce top_k from 3 to 2
- Add answer extraction post-processing