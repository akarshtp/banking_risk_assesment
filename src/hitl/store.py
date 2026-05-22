import json
import os
import uuid
from datetime import datetime
from src.core.logger_config import app_logger

# Simple file-based store for pending tasks
HITL_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hitl_store.json")

def _load_store():
    if not os.path.exists(HITL_DB_FILE):
        os.makedirs(os.path.dirname(HITL_DB_FILE), exist_ok=True)
        return {}
    try:
        with open(HITL_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def _save_store(data):
    with open(HITL_DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

class HITLStore:
    @staticmethod
    def create_task(agent_context: dict, rule_triggered: dict) -> str:
        task_id = str(uuid.uuid4())
        store = _load_store()
        
        task = {
            "task_id": task_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "rule": rule_triggered,
            "agent_context": str(agent_context),  # Convert to string to prevent JSON serialization errors with Langchain objects
            "decision": None,
            "comments": None,
            "reviewer": None
        }
        
        store[task_id] = task
        _save_store(store)
        app_logger.info(f"Created HITL task {task_id} for rule {rule_triggered.get('id')}")
        return task_id
        
    @staticmethod
    def get_pending_tasks() -> list:
        store = _load_store()
        return [t for t in store.values() if t["status"] == "pending"]
        
    @staticmethod
    def resolve_task(task_id: str, decision: str, comments: str, reviewer: str):
        store = _load_store()
        if task_id not in store:
            raise ValueError(f"Task {task_id} not found")
            
        store[task_id]["status"] = "resolved"
        store[task_id]["decision"] = decision
        store[task_id]["comments"] = comments
        store[task_id]["reviewer"] = reviewer
        store[task_id]["resolved_at"] = datetime.utcnow().isoformat()
        
        _save_store(store)
        app_logger.info(f"Resolved HITL task {task_id} with decision: {decision}")
        return store[task_id]
