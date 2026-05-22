# Week 4 Baseline Evaluation Report

## Setup
- **Model**: claude-3-haiku
- **Dataset**: Golden Set (10 queries)
- **Vector DB**: FAISS

## Metrics
- **Context Precision**: 0.72
- **Context Recall**: 0.65
- **Faithfulness**: 0.88
- **Answer Relevance**: 0.79
- **LLM Judge Correctness**: 3.8 / 5.0
- **Composite Score**: 0.75

## Notes
Baseline run before full LCEL and MCP refactor. Hit@K for K=3 was 60%.
