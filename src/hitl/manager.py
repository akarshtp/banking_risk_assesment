import yaml
import os
from src.hitl.store import HITLStore
from src.core.logger_config import app_logger

class HITLManager:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "hitl_rules.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.rules = yaml.safe_load(f).get("rules", [])
        else:
            self.rules = []
            
    def check_and_trigger(self, intermediate_results: dict) -> str:
        """
        Evaluate intermediate results against HITL rules.
        If a rule is triggered, creates a pending task and returns the task ID.
        Returns None if no rules are triggered.
        """
        # Very simple evaluation logic for demonstration
        # In a real system, you'd use a rules engine or AST evaluation
        
        # Extract potential variables from intermediate results
        # e.g., if the agent calculated a loan amount, it might be in the text or tool results
        
        # Since we don't have a structured AST parser here, we'll do basic heuristic checks
        # on the string representation of the intermediate results.
        import re
        content_str = str(intermediate_results).lower()
        
        # Extract numbers to check for high value loans
        amounts = re.findall(r'(?:rs\.?|inr|₹|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', content_str)
        high_value_found = False
        for amt in amounts:
            try:
                val = float(amt.replace(',', ''))
                if val >= 5000000:
                    high_value_found = True
                    break
            except ValueError:
                pass

        for rule in self.rules:
            if rule["id"] == "high_value_loan" and high_value_found:
                app_logger.warning("HITL rule 'high_value_loan' triggered based on amount parsing.")
                return HITLStore.create_task(intermediate_results, rule)
                
            if rule["id"] == "low_credit_score" and ("score: 5" in content_str or "score is 5" in content_str):
                app_logger.warning("HITL rule 'low_credit_score' triggered.")
                return HITLStore.create_task(intermediate_results, rule)

        return None
        
hitl_manager = HITLManager()
