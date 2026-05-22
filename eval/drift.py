import numpy as np

class DriftDetector:
    """
    Evaluates data drift between baseline datasets and current production distributions.
    """
    
    @staticmethod
    def compute_kl_divergence(p: list, q: list, epsilon: float = 1e-10) -> float:
        """
        Compute Kullback-Leibler (KL) divergence between two discrete probability distributions.
        P represents the baseline distribution, Q represents the new distribution.
        """
        p = np.array(p, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        
        # Normalize to ensure they sum to 1
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Add epsilon to prevent log(0) and division by zero
        p = np.where(p == 0, epsilon, p)
        q = np.where(q == 0, epsilon, q)
        
        return np.sum(p * np.log(p / q))
        
    def analyze_credit_score_drift(self, baseline_scores: list, recent_scores: list, num_bins: int = 10) -> dict:
        """
        Analyzes if recent credit scores have drifted significantly from the baseline.
        """
        if not baseline_scores or not recent_scores:
            return {"status": "insufficient_data"}
            
        # Create histograms to form probability distributions
        min_val = min(min(baseline_scores), min(recent_scores))
        max_val = max(max(baseline_scores), max(recent_scores))
        
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        p_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
        q_hist, _ = np.histogram(recent_scores, bins=bins, density=True)
        
        # If density=True but sum is 0, handle it
        if np.sum(p_hist) == 0 or np.sum(q_hist) == 0:
             return {"status": "error", "message": "Zero variance distributions"}
             
        kl_div = self.compute_kl_divergence(p_hist, q_hist)
        
        threshold = 0.1 # Example threshold for significant drift
        drift_detected = bool(kl_div > threshold)
        
        return {
            "status": "success",
            "kl_divergence": float(kl_div),
            "drift_detected": drift_detected,
            "threshold": threshold
        }

drift_detector = DriftDetector()
