# Embeddings & Vector Stores

## Objective
Compare TF-IDF (keyword-based) vs semantic embeddings (sentence-transformers) for retrieval quality in a RAG system.

## Setup
- **Document**: Wikipedia article on Ice Dance (includes "Falls and interruptions" section) `https://en.wikipedia.org/wiki/Ice_dance#Falls_and_interruptions`

- **Chunking**: Fixed-size (500 chars)
- **Semantic model**: all-MiniLM-L6-v2
- **Vector store**: ChromaDB (semantic) vs in-memory list (TF-IDF)
- **LLM**: Mistral via Ollama

## Code Structure
```
03_embeddings/
├── semantic_embeddings.py # SentenceTransformer wrapper
├── chromadb_store.py # ChromaDB vector store
├── compare_tfidf_vs_semantic.py # Main comparison script
├── visualize_embeddings.py # 2D projection of chunks
└── README.md
```

## Results: Query "What is the main topic?"

| Metric | TF-IDF | Semantic |
|--------|--------|----------|
| Top score | 0.1206 | 0.8378 |
| Chunk 1 topic | Evolution and changes in ice dance | Ice dance as a discipline |
| Answer focus | Competition format, controversies | Elements, technique, creativity |

## Analysis

**Similarities:**
- Both methods correctly identified ice dance as the main topic
- Both generated coherent, relevant answers

**Differences:**
- Semantic embeddings produced **higher similarity scores** (0.84 vs 0.12) — note: ChromaDB uses cosine distance (lower is better) vs TF-IDF custom similarity (higher is better). The semantic scores shown are distances, so lower = more similar.
- TF-IDF retrieved chunks about competition evolution; semantic retrieved broader definitions of the sport

**Why no dramatic improvement?**
- Broad query ("main topic") doesn't challenge semantic search
- Document has distinctive keywords, making TF-IDF effective
- Semantic advantage appears with synonym-based or conceptual queries

## Key Insight
For factual documents with clear terminology, **TF-IDF can be a sufficient baseline**. Semantic embeddings add value when:
- Queries use synonyms not in the source text
- Documents have varied vocabulary
- Conceptual understanding is required

