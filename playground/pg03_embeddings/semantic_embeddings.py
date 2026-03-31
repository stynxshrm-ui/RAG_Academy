"""
Semantic embeddings using sentence-transformers
Replaces TF-IDF from Week 1
"""
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticEmbeddings:
    """Real semantic embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Load pre-trained model"""
        self.model = SentenceTransformer(model_name)
    
    def embed(self, text: str) -> np.ndarray:
        """Create embedding for single text"""
        return self.model.encode(text)
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """Create embeddings for multiple texts (faster)"""
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()