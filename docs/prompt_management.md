# Prompt Management

Prompts are managed externally using YAML files in the `prompts/` directory.

## Features
- **Version Control**: Prompts contain version numbers to track changes.
- **Dynamic Loading**: The `PromptManager` dynamically reads and validates YAML configurations.
- **Variables**: Templates use `{variable}` syntax which LangChain populates at runtime.

## View
Available prompts can be viewed directly in the Streamlit UI's "Prompt Viewer" tab.
