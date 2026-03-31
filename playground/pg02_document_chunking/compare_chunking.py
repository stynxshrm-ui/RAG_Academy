"""
compare_chunking_with_llm.py
Test chunking strategies by generating actual RAG responses
"""

import os
# sys.path.append('../01_simple_rag')

from pg01_simple_rag.simple_rag import SimpleEmbeddings, VectorStore, cosine_similarity
import ollama
from typing import List
import re
from web_loader import load_web

# ============================================
# CHUNKING STRATEGIES
# ============================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Fixed-size by characters"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_fixed_words(text: str, chunk_words: int = 100, overlap_words: int = 20) -> List[str]:
    """Fixed-size by words"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_words - overlap_words
    return chunks


def chunk_semantic(text: str, min_size: int = 200, max_size: int = 500) -> List[str]:
    """Semantic by paragraphs/sentences"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""
    
    for para in paragraphs:
        if len(para) < min_size:
            if len(current) + len(para) < max_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) < max_size:
                    current += sent + " "
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent + " "
    
    if current:
        chunks.append(current.strip())
    return chunks


# ============================================
# RAG SYSTEM
# ============================================

class SimpleRAG:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self.embedding_model.embed(query)
        similarities = []
        for emb in self.vector_store.embeddings:
            sim = cosine_similarity(query_emb, emb)
            similarities.append(sim)
        
        top_indices = sorted(range(len(similarities)), 
                            key=lambda i: similarities[i], 
                            reverse=True)[:top_k]
        return [self.vector_store.chunks[i] for i in top_indices]
    
    def generate(self, query: str, context: str) -> str:
        prompt = f"""Answer based on context below.

Context: {context}

Question: {query}

Answer:"""
        
        response = ollama.chat(model='mistral', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    
    def query(self, query: str, strategy_name: str = "", top_k: int = 3):
        chunks = self.retrieve(query, top_k)
        context = "\n\n".join(chunks)
        
        print(f"\nQuery: {query}")
        print("\nRetrieved chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c[:150]}...")
        
        print("\nGenerated answer:")
        answer = self.generate(query, context)
        print(answer)
        self.log_result(strategy_name, query, answer, chunks)
        return answer

    def log_result(self, strategy_name, query, answer, chunks):
        os.makedirs('outputs', exist_ok=True)
        # Create filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'outputs/chunking_comparison_{timestamp}.txt'
        
        with open(filename, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"STRATEGY: {strategy_name}\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"RETRIEVED CHUNKS:\n")
            for i, c in enumerate(chunks, 1):
                f.write(f"[{i}] {c}\n")
            f.write(f"\nANSWER:\n{answer}\n")

# ============================================
# TEST FUNCTION
# ============================================

def test_strategy(text: str, strategy_name: str, chunk_func, queries: List[str]):
    """Test one chunking strategy with all queries"""
    print("\n" + "="*80)
    print(f"STRATEGY: {strategy_name}")
    print("="*80)
    
    # Chunk
    chunks = chunk_func(text)
    print(f"Created {len(chunks)} chunks")
    
    # Build vector store
    embedding_model = SimpleEmbeddings()
    embedding_model.build_vocabulary(chunks)
    
    vector_store = VectorStore()
    for chunk in chunks:
        emb = embedding_model.embed(chunk)
        vector_store.add(chunk, emb)
    
    # Create RAG
    rag = SimpleRAG(embedding_model, vector_store)
    
    # Test queries
    for query in queries:
        rag.query(query, strategy_name)
        print("-"*80)


# ============================================
# MAIN
# ============================================

def main():
    # Load document
    text = load_web('https://en.wikipedia.org/wiki/Ice_dance#Falls_and_interruptions')

    
    # Define test queries
    queries = [
        "Give four reasons that can cause interruption of an ice dance?",
    ]
    
    # Test all strategies
    strategies = [
        ("Fixed Size (500 chars, 50 overlap)", lambda t: chunk_fixed_size(t, 500, 50)),
        ("Fixed Words (100 words, 20 overlap)", lambda t: chunk_fixed_words(t, 100, 20)),
        ("Semantic (200-500 chars)", chunk_semantic),
    ]
    
    for name, func in strategies:
        test_strategy(text, name, func, queries)
        input("\nPress Enter to continue to next strategy...")
       

if __name__ == "__main__":
    main()
