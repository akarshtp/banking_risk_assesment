import os
import json
import re # <-- Add this import
from typing import List, Dict, Any
from src.core.schemas import EvalResult
from src.core.logger_config import app_logger
from src.rag.retrieval import retrieve_documents
from src.agent.chain import get_primary_llm

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
    Runs an LLM-as-a-Judge evaluation suite against the RAG pipeline.
    """
    app_logger.info("Starting LLM-as-a-Judge Evaluation Suite...")
    test_set = custom_test_set if custom_test_set else GOLDEN_DATASET
    
    llm = get_primary_llm()

    total_precision = 0.0
    total_recall = 0.0
    total_relevance = 0.0
    total_faithfulness = 0.0
    results_details = []

    for item in test_set:
        query = item["question"]
        ground_truth = item["ground_truth"]

        # 1. Retrieve Contexts
        retrieved_docs = retrieve_documents(query, top_k=3)
        doc_contents = [doc["content"] for doc in retrieved_docs]
        context_str = "\n---\n".join(doc_contents) if doc_contents else "NO CONTEXT RETRIEVED."

        # 2. Generate Answer 
        ans_prompt = (
            f"Answer the question based ONLY on the context provided.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        )
        try:
            response = llm.invoke(ans_prompt)
            generated_answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            app_logger.error(f"Generation failed: {e}")
            generated_answer = "Error generating answer."

        # 3. LLM-as-a-Judge Evaluation Prompt
        judge_prompt = f"""
        You are an expert evaluator. Evaluate the following RAG system outputs on a scale of 0.0 to 1.0 for four metrics:
        1. context_precision: Is the retrieved context relevant to the question?
        2. context_recall: Does the context contain all info needed to match the ground truth?
        3. answer_relevance: Does the generated answer directly address the question?
        4. faithfulness: Is the generated answer fully supported by the retrieved context (no hallucinations)?

        Output strictly as a JSON object with keys: "context_precision", "context_recall", "answer_relevance", "faithfulness". Do not include any other text.

        Question: {query}
        Ground Truth: {ground_truth}
        Retrieved Context: {context_str}
        Generated Answer: {generated_answer}
        """

        try:
            eval_response = llm.invoke(judge_prompt)
            eval_text = eval_response.content if hasattr(eval_response, 'content') else str(eval_response)
            
            # --- NEW ROBUST JSON EXTRACTION ---
            # This regex finds the first '{' and the last '}' and extracts everything inside
            match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if match:
                scores = json.loads(match.group(0))
            else:
                raise ValueError("No JSON object found in LLM response.")
                
        except Exception as e:
            app_logger.error(f"Failed to parse LLM judge JSON: {e}")
            # If it fails, we return zeros so the app doesn't crash
            scores = {"context_precision": 0, "context_recall": 0, "answer_relevance": 0, "faithfulness": 0}

        results_details.append({"query": query, "scores": scores})
        
        total_precision += float(scores.get("context_precision", 0))
        total_recall += float(scores.get("context_recall", 0))
        total_relevance += float(scores.get("answer_relevance", 0))
        total_faithfulness += float(scores.get("faithfulness", 0))

    # Calculate Aggregates
    n = len(test_set)
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_relevance = total_relevance / n
    avg_faithfulness = total_faithfulness / n
    end_to_end = (avg_precision + avg_recall + avg_relevance + avg_faithfulness) / 4.0

    app_logger.info(f"Evaluation complete. Composite Score: {end_to_end:.2f}")

    return EvalResult(
        context_precision=avg_precision,
        context_recall=avg_recall,
        answer_relevance=avg_relevance,
        faithfulness=avg_faithfulness,
        end_to_end_quality=end_to_end,
        details={"raw_results": results_details}
    )
