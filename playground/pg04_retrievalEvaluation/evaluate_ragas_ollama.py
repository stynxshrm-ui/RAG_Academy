"""
RAGAS Evaluation with Local Ollama Model
No OpenAI API key required - runs completely locally
"""

from pg03_embeddings.chromadb_store import ChromaVectorStore
import ollama
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from typing import List, Dict

class DenseRetriever:
    """Pure dense semantic retriever using local Ollama embeddings."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.semantic_model = OllamaEmbeddings(model='mistral:latest')
        self.semantic_store = ChromaVectorStore()
        document_embeddings = self._embed_documents(chunks)
        self.semantic_store.add_batch(chunks, document_embeddings)

    def _embed_documents(self, texts: List[str]):
        if hasattr(self.semantic_model, 'embed_documents'):
            return self.semantic_model.embed_documents(texts)
        if hasattr(self.semantic_model, 'embed_batch'):
            return self.semantic_model.embed_batch(texts)
        raise AttributeError(
            'The embedding model must implement embed_documents or embed_batch'
        )

    def _embed_query(self, query: str):
        if hasattr(self.semantic_model, 'embed_query'):
            return self.semantic_model.embed_query(query)
        if hasattr(self.semantic_model, 'embed'):
            return self.semantic_model.embed(query)
        if hasattr(self.semantic_model, 'embed_documents'):
            return self.semantic_model.embed_documents([query])[0]
        raise AttributeError(
            'The embedding model must implement embed_query, embed, or embed_documents'
        )

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self._embed_query(query)
        results = self.semantic_store.search(query_emb, top_k)
        return [
            {'chunk': r['chunk'], 'score': 1 - r['score']}
            for r in results
        ]

class RAGEvaluatorOllama:
    def __init__(self, chunks: List[str]):
        self.eval_llm = LangchainLLMWrapper(OllamaLLM(model='mistral:latest'))
        self.retriever = DenseRetriever(chunks)
        
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using your RAG pipeline"""
        context_text = "\n\n".join(contexts)
        prompt = f"""Answer based ONLY on the context below.

Context:
{context_text}

Question: {query}

Answer:"""
        
        response = ollama.chat(model='mistral:latest', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']
    
    def create_evaluation_dataset(self, test_questions: List[Dict]) -> Dataset:
        """Create dataset for RAGAS evaluation"""
        data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for item in test_questions:
            question = item['question']
            ground_truth = item['ground_truth']
            
            # Retrieve contexts using dense semantic search
            retrieved = self.retriever.dense_search(question, top_k=3)
            contexts = [r['chunk'] for r in retrieved]
            
            # Generate answer
            answer = self.generate_answer(question, contexts)
            
            data['question'].append(question)
            data['answer'].append(answer)
            data['contexts'].append(contexts)
            data['ground_truth'].append(ground_truth)
            
            print(f"✓ Processed: {question[:50]}...")
        
        return Dataset.from_dict(data)
    
    def run_evaluation(self, test_questions: List[Dict]):
        """Run RAGAS evaluation using local Ollama"""
        
        print("\n" + "="*70)
        print("RAGAS EVALUATION (Ollama - Local)")
        print("="*70)
        
        eval_dataset = self.create_evaluation_dataset(test_questions)
        
        print("\nRunning evaluation metrics...")
        
        # Use metrics that work well with local models
        metrics = [
            faithfulness,
            answer_relevancy,
        ]
        
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.eval_llm,  # Use Ollama instead of OpenAI
            embeddings=self.retriever.semantic_model, # Use Ollama embeddings instead of openAI for evaluation if needed
        )
        
        return result


def print_results(result):
    """Pretty print evaluation results"""
    
    print("\n" + "="*70)
    print(" RAGAS EVALUATION RESULTS (Ollama)")
    print("="*70)
    
    print("\n METRICS (0-1 scale, higher is better):")
    print("-" * 50)
    
    for metric in ['faithfulness', 'answer_relevancy']:
        try:
            score = result._repr_dict[metric]
        except (AttributeError, KeyError):
            continue
        bar_length = int(score * 30)
        bar = "=" * bar_length + "-" * (30 - bar_length)
        print(f"{metric:20}: [{bar}] {score:.3f}")
    
    print("\n INTERPRETATION:")
    print("="*70)
    print("""
    faithfulness (0-1):     How factually accurate is the answer given the context?
    answer_relevancy (0-1): How directly does the answer address the question?
    
    Target:
    > 0.8: Excellent
    0.6-0.8: Good
    0.4-0.6: Needs work
    < 0.4: Significant issues
    """)


def main():
    # Load document
    try:
        with open('data/sample_document.txt', 'r') as f:
            text = f.read()
    except FileNotFoundError:  
        print("sample_document.txt not found. Please create it with relevant content for testing.")
        return
    
    # Use optimal chunking (200 chars)
    chunks = [text[i:i+200] for i in range(0, len(text), 75)]
    print(f" Document split into {len(chunks)} chunks (200 chars, 50 overlap)")
    
    # Test questions with ground truth
    test_questions = [
        {
            'question': 'What happens when skaters fall in ice dance?',
            'ground_truth': 'Teams lose one point for a fall by one partner, and two points if both partners fall.',
        },
        {
            'question': 'When did ice dance become an Olympic sport?',
            'ground_truth': 'Ice dance became an Olympic medal sport in 1976.',
        },
        {
            'question': 'What are the penalties for interruptions?',
            'ground_truth': 'Interruptions of 10-20 seconds cost one point, 20-30 seconds cost two points.',
        }
    ]
    
    # Run evaluation
    evaluator = RAGEvaluatorOllama(chunks)
    results = evaluator.run_evaluation(test_questions)
    
    # Print results
    print_results(results)
    
    # Save results
    try:
        df = results.to_pandas()
        df.to_csv('ragas_results_ollama.csv', index=False)
        print("\n Results saved to 'ragas_results_ollama.csv'")
    except:
        pass
    
    return results


if __name__ == "__main__":
    main()