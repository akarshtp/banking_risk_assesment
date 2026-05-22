import os
import yaml

class RBACManager:
    def __init__(self):
        self.roles = {}
        self._load_roles()
        
    def _load_roles(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "roles.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
                self.roles = data.get("roles", {})
                
    def get_role_filter(self, role_name: str) -> dict:
        """
        Generate a ChromaDB compatible metadata filter based on user role.
        """
        if role_name not in self.roles:
            # Default restrictive fallback if role is unknown
            return {"confidentiality": "public"}
            
        role_config = self.roles[role_name]
        
        # If the role has access to everything, return no filter
        if "*" in role_config.get("allowed_doc_types", []):
            return {}
            
        # Construct filter. In a real system, this might be a complex $and/$or expression
        filter_dict = {}
        allowed_conf = role_config.get("restricted_metadata", {}).get("confidentiality", [])
        if allowed_conf:
            if len(allowed_conf) == 1:
                filter_dict["confidentiality"] = allowed_conf[0]
            else:
                filter_dict["confidentiality"] = {"$in": allowed_conf}
                
        return filter_dict

rbac_manager = RBACManager()
