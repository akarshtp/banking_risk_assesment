import os
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness
)

from schemas import EvalResult
from logger_config import app_logger
from rag.retrieval import retrieve_documents
from chain import get_primary_llm

# =============================================================================
# GOLDEN TEST SET
# Add more domain-specific questions as your knowledge base grows
# =============================================================================
GOLDEN_DATASET = [
    {
        "question": "What is the maximum recommended DTI for a new loan?",
        "ground_truth": "The maximum recommended DTI for a new loan is 45%, or conditionally up to 50% for medium risk profiles."
    },
    {
        "question": "Which documents are required for KYC verification?",
        "ground_truth": "Standard KYC verification requires a PAN card, Aadhaar card, and income proof like a Pay Slip or ITR."
    }
]

def run_evaluation_suite(custom_test_set: List[Dict[str, str]] = None) -> EvalResult:
    """
    Runs the Ragas evaluation suite against the RAG pipeline.
    
    Args:
        custom_test_set: Optional list of dicts with 'question' and 'ground_truth'.
                         Defaults to GOLDEN_DATASET.
    """
    app_logger.info("Starting RAG Evaluation Suite...")

    test_set = custom_test_set if custom_test_set else GOLDEN_DATASET
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # We need an LLM to generate the answers for evaluation
    llm = get_primary_llm()

    for item in test_set:
        query = item["question"]
        questions.append(query)
        ground_truths.append(item["ground_truth"])

        # 1. Retrieve Contexts via our Week 3 Pipeline
        retrieved_docs = retrieve_documents(query, top_k=3)
        doc_contents = [doc["content"] for doc in retrieved_docs]
        contexts.append(doc_contents)

        # 2. Generate Answer 
        # For pure RAG evaluation, we use a direct QA prompt to isolate the retrieval quality
        context_str = "\n---\n".join(doc_contents)
        prompt = (
            f"Answer the question based ONLY on the context provided.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        )
        
        try:
            response = llm.invoke(prompt)
            # Extract plain text whether response is string or message object
            ans_text = response.content if hasattr(response, 'content') else str(response)
            answers.append(ans_text)
        except Exception as e:
            app_logger.error(f"Generation failed during eval for query '{query}': {e}")
            answers.append("Error generating answer.")

    # Prepare dataset for Ragas HuggingFace format
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(eval_data)

    # Run Ragas Evaluation
    app_logger.info("Running Ragas metrics (Note: Uses OpenAI as judge by default)...")
    try:
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                answer_relevancy,
                faithfulness
            ]
        )
        
        # Convert Ragas result object to a dictionary
        metrics_dict = {k: float(v) for k, v in result.items()}
        
        # Calculate composite end-to-end quality score (average of the 4 core metrics)
        scores = [
            metrics_dict.get('context_precision', 0),
            metrics_dict.get('context_recall', 0),
            metrics_dict.get('answer_relevancy', 0),
            metrics_dict.get('faithfulness', 0)
        ]
        end_to_end = sum(scores) / len(scores) if scores else 0.0

        app_logger.info(f"Evaluation complete. Composite Score: {end_to_end:.2f}")

        return EvalResult(
            context_precision=metrics_dict.get('context_precision', 0.0),
            context_recall=metrics_dict.get('context_recall', 0.0),
            answer_relevance=metrics_dict.get('answer_relevancy', 0.0),
            faithfulness=metrics_dict.get('faithfulness', 0.0),
            end_to_end_quality=end_to_end,
            details={"raw_results": metrics_dict}
        )

    except Exception as e:
        app_logger.error(f"Ragas evaluation failed: {e}")
        raise RuntimeError(f"Evaluation suite failed: {e}")