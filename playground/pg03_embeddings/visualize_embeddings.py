"""
Visualize embeddings in 2D to see semantic clustering
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pg02_document_chunking.web_loader import load_web
from pg03_embeddings.semantic_embeddings import SemanticEmbeddings
import re

def visualize_chunks(chunks: list, model: SemanticEmbeddings):
    """Create 2D visualization of chunk embeddings"""
    
    # Create embeddings
    embeddings = model.embed_batch(chunks)
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(chunks)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create labels (first few words of each chunk)
    labels = [chunk[:30] + "..." if len(chunk) > 30 else chunk for chunk in chunks]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.title("Document Chunks - Semantic Embedding Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig("embedding_visualization.png", dpi=150)
    plt.show()
    
    print(f"Saved visualization to embedding_visualization.png")

def main():
    # Load your document
    text = load_web('https://en.wikipedia.org/wiki/Ice_dance#Falls_and_interruptions')
    text = re.sub(r'\n+\s+', ' ', text)  # Clean up whitespace
    print(text[:500] + "...\n")  # Print start of document for reference
    
    # Simple chunking (reuse from Week 2)
    def simple_chunk(text, size=500):
        chunks = []
        for i in range(0, len(text), size):
            chunks.append(text[i:i+size])
        return chunks
    
    chunks = simple_chunk(text, 500)
    print(f"Created {len(chunks)} chunks")
    
    # Create semantic embeddings
    model = SemanticEmbeddings()
    
    # Visualize
    visualize_chunks(chunks, model)

if __name__ == "__main__":
    main()