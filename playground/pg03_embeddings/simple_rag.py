import numpy as np
from pathlib import Path
from typing import List, Tuple

'''
From pg01_simple_rag/simple_rag.py - A simple RAG implementation with TF-IDF embeddings.
'''

class SimpleEmbeddings:
    """
    A simplified keyword-based embedding method using TF-IDF.
    This is simpler than neural networks but still captures meaning.
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
    
    def build_vocabulary(self, chunks: List[str]):
        """Build vocabulary from all chunks"""
        word_doc_count = {}
        
        for chunk in chunks:
            words = set(chunk.lower().split())
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Assign index to each word
        self.vocabulary = {word: idx for idx, word in enumerate(word_doc_count.keys())}
        
        # Calculate IDF (inverse document frequency)
        n_docs = len(chunks)
        for word, count in word_doc_count.items():
            self.idf[word] = np.log(n_docs / (1 + count))
    
    def embed(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text"""
        if not self.vocabulary:
            raise ValueError("Build vocabulary first!")
        
        # Initialize zero vector
        embedding = np.zeros(len(self.vocabulary))
        
        # Count words in text
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if word in self.vocabulary:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate TF-IDF for each word
        for word, count in word_counts.items():
            tf = count / len(words)
            idx = self.vocabulary[word]
            embedding[idx] = tf * self.idf.get(word, 1.0)
        
        # Normalize ensures all vectors are on the same scale for similarity comparison.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

# ============================================================================
# PART 4: SIMILARITY SEARCH
# ============================================================================
class VectorStore:
    """Simple vector store for embeddings"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add(self, chunk: str, embedding: np.ndarray):
        """Add a chunk and its embedding"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Search for top-k similar chunks"""
        similarities = []
        
        for i, chunk_embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((self.chunks[i], sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
