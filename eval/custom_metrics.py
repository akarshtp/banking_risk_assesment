import re

def compute_compliance_score(generated_answer: str, ground_truth: str) -> float:
    """
    Custom metric: Domain-specific regulatory compliance score.
    Checks if the generated answer contains required compliance keywords if the ground truth requires them.
    """
    compliance_keywords = ["policy", "regulation", "guideline", "mandatory", "must", "required"]
    
    gt_lower = ground_truth.lower()
    needs_compliance = any(kw in gt_lower for kw in compliance_keywords)
    
    if not needs_compliance:
        return 1.0 # N/A, full score
        
    ans_lower = generated_answer.lower()
    has_compliance = any(kw in ans_lower for kw in compliance_keywords)
    
    return 1.0 if has_compliance else 0.0

def compute_citation_precision(generated_answer: str, retrieved_texts: list) -> float:
    """
    Custom metric: Citation precision.
    Verifies that claims in the answer can be backed by retrieved chunks.
    """
    # Extremely simplified heuristic for demonstration:
    # If answer contains numbers, those numbers should exist in the retrieved texts.
    numbers = re.findall(r'\b\d+\b', generated_answer)
    if not numbers:
        return 1.0
        
    combined_context = " ".join(retrieved_texts)
    matched = sum(1 for num in numbers if num in combined_context)
    
    return matched / len(numbers)
