import os
import json
import random
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class LLMJudge:
    def __init__(self):
        """
        Initializes the LLM Judge. It uses a different LLM than the primary generator
        to reduce bias. If primary is Anthropic, judge is OpenAI, and vice versa.
        """
        primary = os.getenv("PRIMARY_PROVIDER", "anthropic").lower()
        if primary == "anthropic":
            # Use OpenAI as Judge
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.0
            )
        else:
            # Use Anthropic as Judge
            self.llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.0
            )
            
        self.parser = JsonOutputParser()

    def evaluate_response(self, user_query: str, context: str, ai_response: str) -> Dict[str, Any]:
        """
        Evaluates a response on 4 dimensions (1-5 scale) using the evaluation prompt template.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are an impartial expert judge evaluating an AI underwriter's response.\n"
            "Please evaluate the following response based on the user's query and the provided context.\n\n"
            "User Query:\n{user_query}\n\n"
            "Context Provided to AI:\n{context}\n\n"
            "AI Response:\n{ai_response}\n\n"
            "Rate the response on a scale of 1 to 5 (1=Poor, 5=Excellent) for each dimension:\n"
            "1. correctness: Is the information factually accurate according to the context?\n"
            "2. completeness: Did the AI answer all parts of the user's query?\n"
            "3. citation_quality: Did the AI properly reference the context?\n"
            "4. clarity: Is the response easy to understand and well-structured?\n\n"
            "Output ONLY a valid JSON object with the exact keys: "
            "'correctness', 'completeness', 'citation_quality', 'clarity', and a 'reasoning' string.\n"
            "Do not include markdown blocks like ```json."
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "user_query": user_query,
                "context": context,
                "ai_response": ai_response
            })
            return result
        except Exception as e:
            print(f"Error during LLM Judge evaluation: {e}")
            return {
                "correctness": 0,
                "completeness": 0,
                "citation_quality": 0,
                "clarity": 0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    def bucket_failure(self, user_query: str, expected_answer: str, actual_response: str) -> str:
        """
        Failure analysis script that buckets misses by root cause.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are a failure analysis expert. An AI system failed to provide the expected answer.\n"
            "Based on the following inputs, categorize the root cause of the failure into exactly ONE of the following buckets:\n"
            "- 'Missing Context': The AI lacked the necessary information to answer.\n"
            "- 'Hallucination': The AI made up facts not supported by truth.\n"
            "- 'Off-topic': The AI refused to answer or digressed inappropriately.\n"
            "- 'Logic Error': The AI had the right information but calculated or reasoned incorrectly.\n"
            "- 'Formatting Error': The AI failed to follow output structure instructions.\n"
            "- 'Other': Anything else.\n\n"
            "User Query: {user_query}\n"
            "Expected Answer: {expected_answer}\n"
            "Actual AI Response: {actual_response}\n\n"
            "Output ONLY a JSON object with the key 'root_cause' containing the exact bucket name, and 'explanation' explaining why."
        )
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "user_query": user_query,
                "expected_answer": expected_answer,
                "actual_response": actual_response
            })
            return result.get("root_cause", "Other")
        except Exception as e:
            print(f"Failure bucketing failed: {e}")
            return "Evaluation Error"

# Expose a singleton instance
llm_judge = LLMJudge()

def run_llm_judge(query: str, ground_truth: str, context: str, generated_answer: str) -> Dict[str, Any]:
    """
    Adapter function to resolve ImportError in run_eval.py.
    """
    return llm_judge.evaluate_response(user_query=query, context=context, ai_response=generated_answer)