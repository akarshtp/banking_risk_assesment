import os
import json
import re
from chain import get_fallback_llm
from logger_config import app_logger

def run_llm_judge(query: str, ground_truth: str, context: str, generated_answer: str) -> dict:
    """
    DECOUPLED LLM JUDGING HARNESS: Runs evaluation via a separate LLM provider
    (OpenAI) to eliminate self-grading bias from the generation system.
    """
    judge_llm = get_fallback_llm()

    judge_prompt = f"""
    You are an objective QA system judge. Grade the generated answer against the ground truth and context on four metrics.
    Assign an integer score from 1 to 5 (where 1 is worst, 5 is perfect) for each metric:

    1. correctness: Does the generated answer factually align with the ground truth?
    2. completeness: Does the answer completely resolve all elements raised in the prompt?
    3. citation_quality: Does the answer anchor assertions accurately inside the context without fabricating data?
    4. clarity: Is the response professional, well-formatted, clean, and free of filler?

    You must format your response strictly as a JSON object with these exact keys:
    "correctness", "completeness", "citation_quality", "clarity". No other text or markdown formatting.

    Question: {query}
    Ground Truth: {ground_truth}
    Retrieved Context: {context}
    Generated Answer: {generated_answer}
    """

    try:
        response = judge_llm.invoke(judge_prompt)
        text_output = response.content if hasattr(response, 'content') else str(response)
        
        # Strip out potential markdown wrapping safely
        match = re.search(r'\{.*\}', text_output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("Failed to extract structural JSON blocks from judge.")
    except Exception as e:
        app_logger.error(f"LLM Judge API execution failed: {e}")
        return {"correctness": 1, "completeness": 1, "citation_quality": 1, "clarity": 1}