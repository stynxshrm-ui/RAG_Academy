# Simple RAG Pipeline

This example demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline. The pipeline includes the following steps:

1. **Load Documents**: Load a small set of example documents.
2. **Chunk Documents**: Split documents into smaller chunks (e.g., sentences).
3. **Embed Chunks**: Generate simple embeddings for each chunk.
4. **Retrieve Relevant Chunks**: Find chunks relevant to a query.
5. **Generate Response**: Combine retrieved chunks into a response.

## How to Run

1. Ensure you have Python installed.
2. Run the script:

```bash
python simple_rag.py
```

3. Enter a query when prompted and observe the response.

## Further Exploration

- Modify the chunking logic to split by paragraphs instead of sentences.
- Replace the hash-based embedding with a real embedding model (e.g., Sentence Transformers).
- Experiment with different queries to see how retrieval works.

## Troubleshooting

- If you encounter errors, ensure all dependencies are installed.
- Check the script for typos or syntax errors.