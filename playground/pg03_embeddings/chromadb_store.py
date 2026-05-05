"""
Vector store using ChromaDB
Replaces simple list-based store from Week 1
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict

class ChromaVectorStore:
    """Production vector store with persistence"""
    
    def __init__(self, collection_name: str = "rag_docs", persist_dir: str = "./chroma_db"):
        # Updated ChromaDB client initialization
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Delete existing if needed (for fresh start)
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.next_id = 0
    
    def add(self, chunk: str, embedding: np.ndarray | list, metadata: Dict = None):
        """Add a chunk with its embedding"""
        if metadata is None:
            metadata = {"index": self.next_id}  # Non-empty metadata

        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        self.collection.add(
            embeddings=[embedding_list],
            documents=[chunk],
            metadatas=[metadata],
            ids=[str(self.next_id)]
        )
        self.next_id += 1
    
    def add_batch(self, chunks: List[str], embeddings: np.ndarray | list, metadatas: List[Dict] = None):
        """Add multiple chunks at once"""
        ids = [str(i + self.next_id) for i in range(len(chunks))]
        
        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(chunks))]

        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]
        
        # Or use empty dict with a placeholder (Chroma requires at least one field)
        # metadatas = [{"source": "document"} for _ in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        self.next_id += len(chunks)
    
    def search(self, query_embedding: np.ndarray | list, top_k: int = 3) -> List[Dict]:
        """Search for similar chunks"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved.append({
                    'chunk': doc,
                    'score': results['distances'][0][i] if results['distances'] else 1.0,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        return retrieved
    
    def persist(self):
        """Save to disk (ChromaDB auto-persists)"""
        # ChromaDB auto-persists with PersistentClient
        pass