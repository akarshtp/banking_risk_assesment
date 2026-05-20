import math
from typing import List

def compute_hit_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int) -> float:
    """Calculates if at least one expected chunk appears in the top K retrieved slots."""
    top_k_retrieved = [src.lower().strip() for src in retrieved_sources[:k]]
    for target in expected_sources:
        target_clean = target.lower().strip()
        if any(target_clean in fetched or fetched in target_clean for fetched in top_k_retrieved):
            return 1.0
    return 0.0

def compute_mrr(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Calculates the Reciprocal Rank of the first correctly retrieved matching context chunk."""
    for index, fetched in enumerate(retrieved_sources):
        fetched_clean = fetched.lower().strip()
        for target in expected_sources:
            target_clean = target.lower().strip()
            if target_clean in fetched_clean or fetched_clean in target_clean:
                return 1.0 / (index + 1)
    return 0.0

def compute_ndcg(retrieved_sources: List[str], expected_sources: List[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain for binary relevance ratings."""
    dcg = 0.0
    for index, fetched in enumerate(retrieved_sources[:k]):
        fetched_clean = fetched.lower().strip()
        is_relevant = 0.0
        for target in expected_sources:
            target_clean = target.lower().strip()
            if target_clean in fetched_clean or fetched_clean in target_clean:
                is_relevant = 1.0
                break
        if is_relevant > 0:
            dcg += is_relevant / math.log2(index + 2)
            
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_sources), k)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_context_precision(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Calculates how densely relevant chunks are concentrated near the top of retrieval results."""
    hits = 0
    precision_sum = 0.0
    for index, fetched in enumerate(retrieved_sources):
        fetched_clean = fetched.lower().strip()
        matched = False
        for target in expected_sources:
            target_clean = target.lower().strip()
            if target_clean in fetched_clean or fetched_clean in target_clean:
                matched = True
                break
        if matched:
            hits += 1
            precision_sum += hits / (index + 1)
    return precision_sum / hits if hits > 0 else 0.0

def compute_context_recall(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Calculates the percentage of ground-truth chunks successfully surfaced in results."""
    hits = 0
    retrieved_flat = " ".join(retrieved_sources).lower().strip()
    for target in expected_sources:
        target_clean = target.lower().strip()
        if target_clean in retrieved_flat or any(target_clean in r.lower() for r in retrieved_sources):
            hits += 1
    return hits / len(expected_sources) if expected_sources else 0.0