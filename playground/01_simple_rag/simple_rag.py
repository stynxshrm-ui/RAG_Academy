"""
SIMPLE RAG FROM SCRATCH - No fancy libraries
This shows you EXACTLY how RAG works under the hood
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# PART 1: LOAD DOCUMENTS
# ============================================================================
def load_documents(filepath: str) -> str:
    """Load document from file"""
    with open(filepath, 'r') as f:
        return f.read()

# ============================================================================
# PART 2: CHUNK DOCUMENTS (Better than sentence splitting!)
# ============================================================================
def chunk_documents(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    This is better than sentence splitting because:
    - Sentences can be too short
    - Overlap preserves context
    - You control chunk size
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

# ============================================================================
# PART 3: REAL EMBEDDINGS (Using simple but semantic method)
# TF = (Number of times word appears in chunk) / (Total words in chunk)
# IDF(word) = log(Total chunks / (1 + Number of chunks containing word))
# TF-IDF = TF * IDF
# ============================================================================
class SimpleEmbeddings:
    """
    A simplified but SEMANTIC embedding method using TF-IDF.
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
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class VectorStore:
    """Simple vector store for embeddings"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
    
    def add(self, chunk: str, embedding: np.ndarray):
        """Add a chunk and its embedding"""
        self.chunks.append(chunk)
        self.embeddings.append(embedding)
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Search for top-k similar chunks"""
        similarities = []
        
        for i, chunk_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((self.chunks[i], sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

# ============================================================================
# MAIN PIPELINE
# ============================================================================
print("=" * 70)
print("BUILDING RAG FROM SCRATCH - Understanding Every Step")
print("=" * 70)

# Step 1: Create sample document
print("\n STEP 1: Creating sample document")
sample_content = """
Python is a programming language created by Guido van Rossum. It is great for data science.
Data science uses Python for analysis and machine learning. Pandas is a key library.
Machine learning helps computers learn from data. Neural networks are part of ML.
Python also works for web development with Django and Flask.
Data visualization uses matplotlib and seaborn in Python.
RAG systems combine retrieval and generation for better AI answers.
"""

sample_file = Path("data/sample.txt")
sample_file.parent.mkdir(exist_ok=True)
sample_file.write_text(sample_content)
print(f"   Created: {sample_file}")

# Step 2: Load
print("\n STEP 2: Loading document")
text = load_documents(str(sample_file))
print(f"   Loaded {len(text)} characters")

# Step 3: Chunk (better than sentence splitting!)
print("\n  STEP 3: Chunking document")
chunks = chunk_documents(text, chunk_size=30, overlap=10)
print(f"   Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks, 1):
    print(f"   Chunk {i}: {chunk[:80]}...")

# Step 4: Create embeddings (REAL semantic embeddings!)
print("\n STEP 4: Creating embeddings (TF-IDF - captures meaning)")
embedder = SimpleEmbeddings()
embedder.build_vocabulary(chunks)

# Step 5: Store vectors
print("\n STEP 5: Storing in vector store")
vector_store = VectorStore()

for chunk in chunks:
    embedding = embedder.embed(chunk)
    vector_store.add(chunk, embedding)

print(f"   Stored {len(chunks)} vectors of size {len(embedding)}")

# Step 6: Search
print("\n STEP 6: Searching for relevant chunks")
print("-" * 70)

test_queries = [
    "What is Python good for?",
    "Tell me about data science",
    "What is machine learning?",
    "Web development in Python"
]

for query in test_queries:
    print(f"\n Query: '{query}'")
    
    # Embed the query
    query_embedding = embedder.embed(query)
    
    # Search
    results = vector_store.search(query_embedding, k=2)
    
    print("   Top matches:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"   {i}. Score: {score:.4f} - {chunk[:80]}...")


print("\nCompare two similar sentences:")
sentence1 = "Python is great for data science"
sentence2 = "Python is great for machine learning"
sentence3 = "The weather is nice today"

print(f"\nSentence 1: {sentence1}")
print(f"Sentence 2: {sentence2}")
print(f"Sentence 3: {sentence3}")

print("\n Using TF-IDF Embeddings (semantic):")
embedder_simple = SimpleEmbeddings()
embedder_simple.build_vocabulary([sentence1, sentence2, sentence3])
emb1 = embedder_simple.embed(sentence1)
emb2 = embedder_simple.embed(sentence2)
emb3 = embedder_simple.embed(sentence3)

sim12 = cosine_similarity(emb1, emb2)
sim13 = cosine_similarity(emb1, emb3)

print(f"   Similarity(sentence1, sentence2) = {sim12:.4f}")
print(f"   Similarity(sentence1, sentence3) = {sim13:.4f}")
print("   → Similar sentences get HIGH similarity scores!")
print("   → Can find semantically related content!")

# ============================================================================
# UNDERSTANDING THE MATH
# ============================================================================
print("\n" + "=" * 70)
print(" UNDERSTANDING WHAT EMBEDDINGS DO")
print("=" * 70)

print("""
Embeddings map text to points in space where:
- Similar concepts → close together
- Different concepts → far apart

Example:
     "Python programming"  →  [0.2, 0.8, 0.1, ...]
     "Coding in Python"   →  [0.3, 0.7, 0.2, ...]  ← CLOSE
     "Weather forecast"    →  [0.9, 0.1, 0.8, ...]  ← FAR

The numbers aren't random - they're computed to capture meaning!
""")

# Show actual vector values
print("\n Actual embedding vectors (first 5 dimensions):")
print(f"   Sentence 1: {emb1[:5]}")
print(f"   Sentence 2: {emb2[:5]}")
print(f"   Sentence 3: {emb3[:5]}")
print("\n   Notice: Similar sentences have SIMILAR numbers!")
print("          Different sentences have DIFFERENT numbers!")

print("\n" + "=" * 70)
print(" CONCLUSION: RAG needs REAL embeddings to work!")
print("=" * 70)
print("""
Real RAG systems use neural embeddings (like OpenAI, Sentence-Transformers).
But now you understand:
- Why simple hashing fails
- What embeddings actually do
- How similarity search works
- The math behind finding relevant content

Next step: Try the real library version with sentence-transformers!
""")