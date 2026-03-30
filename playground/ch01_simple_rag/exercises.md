# Exercises: Simple RAG Pipeline

## Exercise 1: Modify Chunk Size
- Change the chunking logic in `chunk_documents` to split by paragraphs instead of sentences.
- Observe how this affects retrieval and response generation.

## Exercise 2: Experiment with Queries
- Try different queries, such as:
  - "What does RAG stand for?"
  - "How does RAG work?"
- Note how the retrieved chunks and responses change.

## Exercise 3: Improve Embeddings
- Replace the hash-based embedding in `embed_chunks` with a real embedding model, such as Sentence Transformers.
- Compare the quality of responses before and after the change.

## Exercise 4: Add More Documents
- Add new documents to the `load_documents` function.
- Test how the pipeline handles a larger document set.

## Exercise 5: Debugging Practice
- Intentionally introduce an error in one of the functions (e.g., a typo in `chunk_documents`).
- Use print statements or a debugger to identify and fix the issue.