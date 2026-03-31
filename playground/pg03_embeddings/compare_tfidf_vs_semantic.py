"""
Compare retrieval quality: TF-IDF vs Semantic embeddings
"""
import sys


from .simple_rag import SimpleEmbeddings, VectorStore as TFIDFStore
from pg02_document_chunking.web_loader import load_web
from pg03_embeddings.semantic_embeddings import SemanticEmbeddings
from pg03_embeddings.chromadb_store import ChromaVectorStore
import ollama
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RAGWithStore:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 3):
        query_emb = self.embedding_model.embed(query)
        results = self.vector_store.search(query_emb, top_k)
        
        # Handle different return types
        if isinstance(results[0], tuple):
            # TF-IDF returns tuples
            return [{'chunk': chunk, 'score': score} for chunk, score in results]
        else:
            # Chroma returns dicts
            return results
    
    def generate(self, query: str, context: str) -> str:
        prompt = f"""Answer based on context below.

        Context: {context}

        Question: {query}

        Answer:"""
        
        response = ollama.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    
    def query(self, query: str, top_k: int = 3):
        retrieved = self.retrieve(query, top_k)
        chunks = [r['chunk'] for r in retrieved]
        context = "\n\n".join(chunks)
        
        print(f"\nQuery: {query}")
        print("\nRetrieved chunks (with scores):")
        for i, r in enumerate(retrieved, 1):
            r['chunk'] = r['chunk'].strip().replace("\n", " ")  # Clean up chunk for display
            print(f"  [{i}] Score: {r['score']:.4f} - {r['chunk'][:20]}...")
        
        print("\nGenerated answer:")
        answer = self.generate(query, context)
        print(answer)
        return answer

def build_tfidf_rag(chunks):
    """Build RAG with TF-IDF (Week 1 baseline)"""
    embedding_model = SimpleEmbeddings()
    embedding_model.build_vocabulary(chunks)
    
    vector_store = TFIDFStore()
    for chunk in chunks:
        emb = embedding_model.embed(chunk)
        vector_store.add(chunk, emb)
    
    return RAGWithStore(embedding_model, vector_store)

def build_semantic_rag(chunks):
    """Build RAG with semantic embeddings (Week 3)"""
    embedding_model = SemanticEmbeddings()
    
    vector_store = ChromaVectorStore()
    embeddings = embedding_model.embed_batch(chunks)
    vector_store.add_batch(chunks, embeddings)
    
    return RAGWithStore(embedding_model, vector_store)

def main():
    # Load document
    text = load_web('https://en.wikipedia.org/wiki/Ice_dance#Falls_and_interruptions')
    
    # Chunk
    def chunk_text(text, size=500):
        chunks = []
        for i in range(0, len(text), size):
            chunks.append(text[i:i+size])
        return chunks
    
    chunks = chunk_text(text, 500)
    print(f"Document chunked into {len(chunks)} pieces")
    
    # Test queries
    queries = [
        "What is the main topic?",
        # "Summarize 3 key points",
        # "What specific details are mentioned?"
    ]
    
    # Compare both approaches
    print("\n" + "="*80)
    print("TF-IDF BASELINE")
    print("="*80)
    tfidf_rag = build_tfidf_rag(chunks)
    for q in queries:
        tfidf_rag.query(q)
        print("-"*80)
    
    print("\n" + "="*80)
    print("SEMANTIC EMBEDDINGS")
    print("="*80)
    semantic_rag = build_semantic_rag(chunks)
    for q in queries:
        semantic_rag.query(q)
        print("-"*80)

if __name__ == "__main__":
    main()