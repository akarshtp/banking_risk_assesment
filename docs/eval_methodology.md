# Evaluation Methodology

Our evaluation framework ensures the system performs reliably across intrinsic, extrinsic, and LLM-as-judge dimensions.

## Metrics
1. **Automated (RAGAS)**: Context Precision/Recall, Faithfulness, Answer Relevance.
2. **LLM Judge (`eval/llm_judge.py`)**: Correctness, Completeness, Citation Quality, Clarity (1-5 scale).
3. **Failure Analysis**: Script buckets errors (e.g., Hallucination, Logic Error) for continuous improvement.
4. **Data Drift**: KL-Divergence on credit scores to detect shifts in data distribution.

The CI/CD pipeline runs `pytest` and `eval.run_eval` against a golden dataset for regression testing.
