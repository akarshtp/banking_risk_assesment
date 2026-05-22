import os
import yaml
from langchain_core.prompts import ChatPromptTemplate
from src.core.logger_config import app_logger

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")

class PromptManager:
    def __init__(self):
        self.templates = {}
        self.load_all_prompts()
        
    def load_all_prompts(self):
        if not os.path.exists(PROMPTS_DIR):
            os.makedirs(PROMPTS_DIR, exist_ok=True)
            return
            
        for filename in os.listdir(PROMPTS_DIR):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                name = filename.rsplit(".", 1)[0]
                path = os.path.join(PROMPTS_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        self.templates[name] = data
                except Exception as e:
                    app_logger.error(f"Failed to load prompt {filename}: {e}")
                    
    def get_prompt_data(self, name: str) -> dict:
        """Get the raw YAML prompt metadata."""
        # Simple hot-reload: check file modified time or just reload
        path = os.path.join(PROMPTS_DIR, f"{name}.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.templates[name] = data
        return self.templates.get(name)

    def get_chat_prompt(self, name: str, is_system: bool = True) -> ChatPromptTemplate:
        """Retrieve a ChatPromptTemplate constructed from the YAML definition."""
        data = self.get_prompt_data(name)
        if not data or "template" not in data:
            raise ValueError(f"Prompt '{name}' not found or invalid.")
            
        role = "system" if is_system else "human"
        
        # We can construct full chat prompts or simple role prompts based on needs
        if name == "guardrail_prompt":
            return ChatPromptTemplate.from_messages([
                ("system", data["template"]),
                ("human", "{input}"),
            ])
            
        return ChatPromptTemplate.from_messages([
            ("system", data["template"]),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

prompt_manager = PromptManager()
