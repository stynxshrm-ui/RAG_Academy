# hybrid_search.py
import sys
sys.path.append('../01_simple_rag')
sys.path.append('../03_embeddings')

from pg03_embeddings.simple_rag import SimpleEmbeddings
from pg03_embeddings.semantic_embeddings import SemanticEmbeddings
from pg03_embeddings.chromadb_store import ChromaVectorStore
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRetriever:
    """Combines sparse (TF-IDF) and dense (semantic) retrieval"""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        
        # Sparse retriever (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks)
        
        # Dense retriever (semantic)
        self.semantic_model = SemanticEmbeddings()
        self.semantic_store = ChromaVectorStore()
        embeddings = self.semantic_model.embed_batch(chunks)
        self.semantic_store.add_batch(chunks, embeddings)
        
    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Keyword-based retrieval"""
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [{
            'chunk': self.chunks[idx],
            'score': float(similarities[idx]),
            'index': idx
        } for idx in top_indices if similarities[idx] > 0]
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic retrieval"""
        query_emb = self.semantic_model.embed(query)
        results = self.semantic_store.search(query_emb, top_k)
        
        return [{
            'chunk': r['chunk'],
            'score': 1 - r['score'],
            'type': 'dense'
        } for r in results]
    
    # Give more weight to dense (semantic) search for specific queries
    # More weight to sparse (keyword) search for fact-based queries
    def hybrid_search(self, query: str, top_k: int = 5, 
                     sparse_weight: float = 0.2, dense_weight: float = 0.8) -> List[Dict]:
        """Combine sparse and dense scores"""
        sparse_results = self.sparse_search(query, top_k=top_k*2)
        dense_results = self.dense_search(query, top_k=top_k*2)
        
        combined = {}
        
        # Add sparse scores
        if sparse_results:
            max_sparse = max(r['score'] for r in sparse_results)
            min_sparse = min(r['score'] for r in sparse_results)
            for r in sparse_results:
                norm = (r['score'] - min_sparse) / (max_sparse - min_sparse) if max_sparse > min_sparse else 0
                combined[r['index']] = norm * sparse_weight
        
        # Add dense scores
        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            min_dense = min(r['score'] for r in dense_results)
            for r in dense_results:
                try:
                    idx = self.chunks.index(r['chunk'])
                    norm = (r['score'] - min_dense) / (max_dense - min_dense) if max_dense > min_dense else 0
                    combined[idx] = combined.get(idx, 0) + norm * dense_weight
                except ValueError:
                    pass
        
        top_indices = sorted(combined.keys(), key=lambda i: combined[i], reverse=True)[:top_k]
        return [{'chunk': self.chunks[idx], 'score': combined[idx]} for idx in top_indices]

def main():
    with open('data/sample_document.txt', 'r') as f:
        text = f.read()
    
    chunk_sizes = [200, 500]
    queries = ["What are the penalties for falls?", "When did ice dance become Olympic?"]
    
    # Store results for comparison
    all_results = {}
    
    for size in chunk_sizes:
        print(f"\n{'='*60}")
        print(f"CHUNK SIZE: {size} chars")
        chunks = [text[i:i+size] for i in range(0, len(text), 150)]
        print(f"Created {len(chunks)} chunks")
        
        retriever = HybridRetriever(chunks)
        
        results_for_size = {}
        
        for query in queries:
            print(f"\n{'='*50}\nQuery: {query}")
            print("\nHybrid search results:")
            
            results = retriever.hybrid_search(query, top_k=2)
            results_for_size[query] = results
            
            for i, r in enumerate(results, 1):
                preview = r['chunk'][:100].replace('\n', ' ')
                print(f"  [{i}] Score: {r['score']:.3f} - {preview}...")
                
                # Manual relevance check
                if query == "What are the penalties for falls?":
                    if 'point' in r['chunk'].lower() or 'fall' in r['chunk'].lower():
                        print(f"      ✓ Contains penalty/fall info")
                elif query == "When did ice dance become Olympic?":
                    if '1976' in r['chunk'] or 'Olympic' in r['chunk']:
                        print(f"      ✓ Contains Olympic date info")
        
        all_results[size] = results_for_size
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\n| Chunk Size | Query | Top Result Preview | Relevant? |")
    print("|------------|-------|-------------------|-----------|")
    
    for size in chunk_sizes:
        for query in queries:
            results = all_results[size][query]
            top_preview = results[0]['chunk'][:50].replace('\n', ' ') + "..."
            
            # Determine relevance
            if query == "What are the penalties for falls?":
                relevant = "✓" if ('point' in results[0]['chunk'].lower() or 
                                   'fall' in results[0]['chunk'].lower()) else "✗"
            elif query == "When did ice dance become Olympic?":
                relevant = "✓" if ('1976' in results[0]['chunk'] or 
                                   'Olympic' in results[0]['chunk']) else "✗"
            else:
                relevant = "?"
            
            print(f"| {size} | {query[:30]} | {top_preview[:40]} | {relevant} |")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Check which size performed better for penalty query
    penalty_200_relevant = False
    penalty_500_relevant = False
    
    if 200 in all_results and "What are the penalties for falls?" in all_results[200]:
        top_chunk = all_results[200]["What are the penalties for falls?"][0]['chunk']
        penalty_200_relevant = 'point' in top_chunk.lower() or 'fall' in top_chunk.lower()
    
    if 500 in all_results and "What are the penalties for falls?" in all_results[500]:
        top_chunk = all_results[500]["What are the penalties for falls?"][0]['chunk']
        penalty_500_relevant = 'point' in top_chunk.lower() or 'fall' in top_chunk.lower()
    
    if penalty_200_relevant and not penalty_500_relevant:
        print("\n 200 char chunks are BETTER for specific factual queries")
        print("   (Correctly retrieved penalty information)")
        print("\n Use 200 char chunks for: Specific questions, factual lookups")
        print(" Use 500 char chunks for: Broad overview, conceptual questions")
    elif penalty_500_relevant and not penalty_200_relevant:
        print("\n 500 char chunks are BETTER for this document")
    else:
        print("\n Both chunk sizes work. Consider:")
        print("   - 200 chars: More precise, better for specific facts")
        print("   - 500 chars: More context, better for summaries")


if __name__ == "__main__":
    main()
