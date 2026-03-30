# compare_with_llm.py
"""
Compare retrieval-only vs retrieval-augmented generation.
Baseline: Simple RAG with TF-IDF + Ollama
"""
from pathlib import Path
import ollama
from simple_rag import SimpleEmbeddings, VectorStore, chunk_documents, cosine_similarity

class SimpleRAG:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve top-k relevant chunks"""
        query_embedding = self.embedding_model.embed(query)
        # Get similarities from vector store
        similarities = []
        for chunk_embedding in self.vector_store.embeddings:
            sim = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append(sim)
        
        # Get top-k indices
        top_indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], 
                           reverse=True)[:top_k]
        
        return [self.vector_store.chunks[i] for i in top_indices]
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer using Ollama"""
        prompt = f"""Answer the question based on the context below.

                Context: {context}

                Question: {query}

                Answer:"""
        
        response = ollama.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    
    def query(self, query: str, top_k: int = 3):
        """Full RAG pipeline"""
        chunks = self.retrieve(query, top_k)
        context = "\n\n".join(chunks)
        
        print(f"\n=== Query: {query} ===\n")
        print("--- Retrieved Chunks ---")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] {chunk[:200]}...")
        
        print("\n--- Generated Answer ---")
        answer = self.generate(query, context)
        print(answer)
        
        return {
            'query': query,
            'chunks': chunks,
            'answer': answer
        }

def main():
    # Load your existing data
    with open(Path("data/sample.txt"), 'r') as f:
        text = f.read()
    
    # Chunk
    chunks = chunk_documents(text, chunk_size=30, overlap=10)
    print(f"Number of chunks: {len(chunks)}")
    
    # Build vector store
    embedding_model = SimpleEmbeddings()
    embedding_model.build_vocabulary(chunks)  # TF-IDF
    
    vector_store = VectorStore()
    for chunk in chunks:
        emb = embedding_model.embed(chunk)
        vector_store.add(chunk, emb)
    
    # Create RAG system
    rag = SimpleRAG(embedding_model, vector_store)
    
    # Test queries
    test_queries = [
        "What is the main topic?",
        "What are the key points?",
        # Add your own questions
    ]
    
    for query in test_queries:
        rag.query(query)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()