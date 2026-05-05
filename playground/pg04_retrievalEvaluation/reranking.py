# reranking.py
import sys

from .hybrid_search_tuning import HybridRetriever
from sentence_transformers import CrossEncoder
from typing import List, Dict

class Reranker:
    def __init__(self):
        print("Loading cross-encoder model...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, chunks: List[str], top_k: int = 3) -> List[Dict]:
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [{'chunk': chunk, 'relevance': float(score)} for chunk, score in scored[:top_k]]

class RAGWithReranking:
    def __init__(self, chunks: List[str]):
        self.retriever = HybridRetriever(chunks)
        self.reranker = Reranker()
    
    def retrieve(self, query: str, top_k: int = 3, rerank: bool = True):
        candidates = self.retriever.hybrid_search(query, top_k=top_k*3 if rerank else top_k)
        candidate_chunks = [c['chunk'] for c in candidates]
        
        if rerank and candidate_chunks:
            return self.reranker.rerank(query, candidate_chunks, top_k)
        return [{'chunk': c['chunk'], 'relevance': c['score']} for c in candidates[:top_k]]

def main():
    with open('data/sample_document.txt', 'r') as f:
        text = f.read()
    
    chunks = [text[i:i+200] for i in range(0, len(text), 150)]
    rag = RAGWithReranking(chunks)
    
    query = "What happens when skaters fall?"
    print(f"Query: {query}\n")
    
    print("Without reranking:")
    for r in rag.retrieve(query, rerank=False):
        print(f"  {r['relevance']:.3f} - {r['chunk'][:80]}...")
    
    print("\nWith reranking:")
    for r in rag.retrieve(query, rerank=True):
        print(f"  {r['relevance']:.3f} - {r['chunk'][:80]}...")

if __name__ == "__main__":
    main()