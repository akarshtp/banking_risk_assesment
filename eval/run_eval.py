import os
import json
import time
from datetime import datetime

# Direct module navigation
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.retrieval import retrieve_documents
from src.agent.chain import get_underwriter_response
from eval.intrinsic import compute_hit_at_k, compute_mrr, compute_ndcg, compute_context_precision, compute_context_recall
from eval.llm_judge import run_llm_judge

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_failure_analysis(intrinsic_scores: dict, extrinsic_scores: dict) -> str:
    """
    FAILURE ANALYSIS ENGINE: Categorizes pipeline errors into specific actionable buckets
    based on exact mathematical score triggers.
    """
    if intrinsic_scores["context_recall"] < 0.5:
        return "Retrieval Miss (Context missing from DB or top matches)"
    elif intrinsic_scores["context_precision"] < 0.5:
        return "Retrieval Noise (Relevant info surfaced too low in context pool)"
    elif extrinsic_scores["correctness"] <= 2:
        return "LLM Synthesis Error (Hallucination or factual mistake)"
    elif extrinsic_scores["completeness"] <= 2:
        return "LLM Missing Information (Answer lacks required depth)"
    elif extrinsic_scores["citation_quality"] <= 2:
        return "Citation Failure (Attribution text contains inaccuracies)"
    return "None (Passed Minimum Threshold)"

def main():
    print("🚀 Initiating Comprehensive Financial RAG Evaluation Harness (50 Test Items)...")
    
    with open(GOLDEN_SET_PATH, "r") as f:
        test_cases = json.load(f)
    total_cases = len(test_cases)
    all_runs = []
    
    # Aggregated Summary Tracking arrays
    total_hit, total_mrr, total_ndcg, total_c_prec, total_c_rec = 0, 0, 0, 0, 0
    total_corr, total_comp, total_cite, total_clar = 0, 0, 0, 0
    
    failure_buckets = {
        "Retrieval Miss (Context missing from DB or top matches)": 0,
        "Retrieval Noise (Relevant info surfaced too low in context pool)": 0,
        "LLM Synthesis Error (Hallucination or factual mistake)": 0,
        "LLM Missing Information (Answer lacks required depth)": 0,
        "Citation Failure (Attribution text contains inaccuracies)": 0,
        "None (Passed Minimum Threshold)": 0
    }

    for idx, case in enumerate(test_cases):
        
        print(f" Processing Item {idx+1}/{total_cases} | Difficulty: {case['difficulty']} | Query: {case['query'][:50]}...")
        
        # 1. Evaluate Retrieval Pipeline
        retrieved_docs = retrieve_documents(case["query"], top_k=3, session_id="eval-harness")
        retrieved_texts = [doc["content"] for doc in retrieved_docs]
        context_string = "\n---\n".join(retrieved_texts)

        # 2. Evaluate Generation Pipeline
        response_payload = get_underwriter_response(case["query"], session_id=f"eval-session-{idx}")
        generated_answer = response_payload["response"]

        # 3. Calculate Intrinsic Retrieval Metrics
        int_metrics = {
            "hit_at_3": compute_hit_at_k(retrieved_texts, case["expected_chunks"], k=3),
            "mrr": compute_mrr(retrieved_texts, case["expected_chunks"]),
            "ndcg_at_3": compute_ndcg(retrieved_texts, case["expected_chunks"], k=3),
            "context_precision": compute_context_precision(retrieved_texts, case["expected_chunks"]),
            "context_recall": compute_context_recall(retrieved_texts, case["expected_chunks"])
        }

        # 4. Calculate Extrinsic Metrics via Decoupled LLM Judge
        ext_metrics = run_llm_judge(
            query=case["query"],
            ground_truth=case["expected_answer"],
            context=context_string,
            generated_answer=generated_answer
        )

        # 5. Run Root-Cause Failure Analysis
        root_cause = run_failure_analysis(int_metrics, ext_metrics)
        failure_buckets[root_cause] += 1

        # Accumulate scores for final summary calculations
        total_hit += int_metrics["hit_at_3"]
        total_mrr += int_metrics["mrr"]
        total_ndcg += int_metrics["ndcg_at_3"]
        total_c_prec += int_metrics["context_precision"]
        total_c_rec += int_metrics["context_recall"]
        
        total_corr += ext_metrics["correctness"]
        total_comp += ext_metrics["completeness"]
        total_cite += ext_metrics["citation_quality"]
        total_clar += ext_metrics["clarity"]

        all_runs.append({
            "id": case["id"],
            "query": case["query"],
            "difficulty": case["difficulty"],
            "intrinsic": int_metrics,
            "extrinsic": ext_metrics,
            "root_cause_analysis": root_cause
        })
        time.sleep(0.5) # Avoid API rate limit spikes

    # Calculate final baseline summary metrics
    total_cases = len(test_cases)
    summary = {
        "avg_hit_at_3": total_hit / total_cases,
        "avg_mrr": total_mrr / total_cases,
        "avg_ndcg_at_3": total_ndcg / total_cases,
        "avg_context_precision": total_c_prec / total_cases,
        "avg_context_recall": total_c_rec / total_cases,
        "avg_correctness": total_corr / total_cases,
        "avg_completeness": total_comp / total_cases,
        "avg_citation_quality": total_cite / total_cases,
        "avg_clarity": total_clar / total_cases
    }

    # Generate Markdown Report Content
    report_md = f"""# 📊 Committed Evaluation Performance Ledger

**Run Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluated Items:** {total_cases} Queries  
**Chunk Strategy Enforced:** {os.getenv("CHUNK_STRATEGY", "recursive").upper()}  
**Embedding Provider Engine:** {os.getenv("EMBEDDING_PROVIDER", "openai").upper()}  

## 📈 1. Aggregate Performance Matrix

### Intrinsic Retrieval Family
| Metric | Value |
| :--- | :--- |
| **Mean Hit @ 3** | {summary['avg_hit_at_3']:.3f} |
| **Mean Reciprocal Rank (MRR)** | {summary['avg_mrr']:.3f} |
| **Normalized Discounted Cumulative Gain (NDCG@3)** | {summary['avg_ndcg_at_3']:.3f} |
| **Context Precision** | {summary['avg_context_precision']:.3f} |
| **Context Recall** | {summary['avg_context_recall']:.3f} |

### Extrinsic Generation (LLM Judge Family) (1–5 Scale)
| Metric | Rating Score |
| :--- | :--- |
| **Factual Correctness** | {summary['avg_correctness']:.2f} / 5.00 |
| **Response Completeness** | {summary['avg_completeness']:.2f} / 5.00 |
| **Citation Attribution Quality** | {summary['avg_citation_quality']:.2f} / 5.00 |
| **Structural Presentation Clarity** | {summary['avg_clarity']:.2f} / 5.00 |

## 🛠️ 2. Automated Root-Cause Failure Breakdown
| Failure Category | Miss Volume Count | Distribution Ratio |
| :--- | :---: | :---: |
| Retrieval Miss (Context missing from DB or top matches) | {failure_buckets['Retrieval Miss (Context missing from DB or top matches)']} | {(failure_buckets['Retrieval Miss (Context missing from DB or top matches)']/total_cases)*100:.1f}% |
| Retrieval Noise (Relevant info surfaced too low in context pool) | {failure_buckets['Retrieval Noise (Relevant info surfaced too low in context pool)']} | {(failure_buckets['Retrieval Noise (Relevant info surfaced too low in context pool)']/total_cases)*100:.1f}% |
| LLM Synthesis Error (Hallucination or factual mistake) | {failure_buckets['LLM Synthesis Error (Hallucination or factual mistake)']} | {(failure_buckets['LLM Synthesis Error (Hallucination or factual mistake)']/total_cases)*100:.1f}% |
| LLM Missing Information (Answer lacks required depth) | {failure_buckets['LLM Missing Information (Answer lacks required depth)']} | {(failure_buckets['LLM Missing Information (Answer lacks required depth)']/total_cases)*100:.1f}% |
| Citation Failure (Attribution text contains inaccuracies) | {failure_buckets['Citation Failure (Attribution text contains inaccuracies)']} | {(failure_buckets['Citation Failure (Attribution text contains inaccuracies)']/total_cases)*100:.1f}% |
| None (Passed Minimum Performance Thresholds) | {failure_buckets['None (Passed Minimum Threshold)']} | {(failure_buckets['None (Passed Minimum Threshold)']/total_cases)*100:.1f}% |
"""

    # Save output reports
    report_target = "eval_baseline.md" if os.getenv("CHUNK_STRATEGY", "recursive") == "recursive" else "eval_final.md"
    with open(os.path.join(OUTPUT_DIR, report_target), "w") as f:
        f.write(report_md)

    with open(os.path.join(OUTPUT_DIR, "eval_details_latest.json"), "w") as f:
        json.dump({"summary": summary, "failures": failure_buckets, "runs": all_runs}, f, indent=2)

    print(f"\n✅ Evaluation pass complete. Committed ledger saved to: test_reports/{report_target}")

if __name__ == "__main__":
    main()