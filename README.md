# Loan Underwriting Assistant: Comprehensive Project Handover

An enterprise-grade, AI-powered loan risk assessment system built with **LangChain**, **Claude (Anthropic)**, **FastAPI**, and **Streamlit**.

This application analyzes complex loan applications by combining traditional rule-based banking logic with advanced Agentic AI workflows. It is designed to be fully decoupled, auditable, and secure.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the main Streamlit Chat interface here ]*

---

## 🚀 Core Underwriting Capabilities

At its foundation, the AI uses structured Pydantic models to assess risk across three primary financial pillars:

1. **DTI (Debt-to-Income) Calculator:** Automatically computes front-end and back-end DTI ratios, determines maximum recommended loan amounts, and evaluates eligibility based on strict banking thresholds.
2. **Document Verification Engine:** Validates the format and authenticity of PAN cards, Aadhaar cards, PaySlips, and ITRs. It flags discrepancies and generates confidence scores to prevent fraud.
3. **Credit Score Analyzer:** Simulates a localized credit scoring model. It analyzes payment history, credit age, active loans, and defaults to assign applicants into actionable risk buckets (Low, Medium, High, Reject).

> *[ SCREENSHOT PLACEHOLDER: Add an image of the AI generating a structured LoanDecision output in the chat here ]*

---

## 🧠 Advanced Agent Architecture

The project utilizes a modern LangChain Expression Language (LCEL) architecture:

*   **ToolCallingAgent & AgentExecutor:** The brain of the system automatically determines which tools to invoke. It features a robust 3-attempt exponential backoff retry mechanism to survive API rate limits.
*   **Decoupled Prompt Management (`prompts.yaml`):** All system instructions and Guardrails are separated from the Python logic. A dedicated **Prompt Viewer Tab** allows administrators to inspect active prompts without touching the codebase.
*   **Semantic Few-Shot Routing:** Using `SemanticSimilarityExampleSelector` and ChromaDB, the agent dynamically injects the 3 most relevant examples into its context window based on the user's specific query.
*   **Windowed Memory:** Retains the last 10 conversation turns per session, allowing for deep, multi-turn interviews with applicants.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the Prompt Viewer Tab here ]*

---

## 🔐 Secure RAG & Role-Based Access Control (RBAC)

The system is augmented with a powerful Retrieval-Augmented Generation (RAG) pipeline backed by vector databases (Pinecone/Chroma/FAISS). It ingests internal banking manuals and compliance guidelines.

**Zero-Leakage RBAC:**
To ensure data security, the RAG pipeline enforces strict metadata filtering. When a user selects their role in the sidebar (e.g., Junior Analyst, Senior Underwriter, Credit Head), the backend uses `get_role_filter()` to automatically restrict vector searches. For example, a Junior Analyst cannot retrieve `confidential` documents, guaranteeing zero data leakage.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the Sidebar showing the User Role dropdown and the colored 'Access Level' warning box here ]*

---

## 🔌 Model Context Protocol (MCP) Integration

The project has completely decoupled its advanced data-fetching tools using Anthropic's **Model Context Protocol (MCP)**. Instead of hardcoding logic, the agent connects dynamically to four external FastMCP Python servers via standard I/O streams:

1. **Property Valuation Server:** Mocks retrieving real estate pricing, commercial zoning laws, and environmental risk tiers (flood/seismic zones).
2. **Regulatory Feed Server:** Mocks fetching the absolute latest central bank (RBI) updates, such as repo rate changes and LTV caps.
3. **Credit Bureau Server:** An external mock API for fetching PAN histories.
4. **Income Verification Server:** An external mock API for verifying employment data.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the 🔌 MCP Servers Tab showing the active connections here ]*

---

## ⏸️ Human-In-The-Loop (HITL) Workflows

The AI is not autonomous when it comes to high-risk decisions. Using rules defined in `config/hitl_rules.yaml`, the system intercepts and pauses highly sensitive requests before execution. 

For example, if an applicant requests a commercial loan exceeding ₹50,000,000, the Agent immediately halts and places the request into a **Pending Review** queue. A human underwriter must navigate to the **HITL Approvals Tab** to manually review the rationale, provide comments, and click "Approve" before the AI is allowed to finalize the decision.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the ✅ HITL Approvals Tab showing a pending task here ]*

---

## 📊 Comprehensive Evaluation & Data Drift Harness

To guarantee the reliability of the AI, the project features a rigorous, automated testing suite built directly into the UI's **Eval Dashboard**.

**1. LLM-as-a-Judge Evaluation:**
The system evaluates a Golden Dataset of complex banking queries. For each query, it runs the pipeline and then spins up a *second* AI to act as a judge, generating a detailed scorecard:
*   **Intrinsic Retrieval:** Hit@3, MRR (Mean Reciprocal Rank), Context Precision, Context Recall.
*   **Extrinsic Generation:** Factual Correctness (1-5), Completeness (1-5), Citation Quality (detecting hallucinations), and Compliance with banking policies.

**2. Data Drift Detection:**
The backend utilizes KL-Divergence mathematics to compare incoming application credit profiles against historical baselines, alerting administrators if the system begins experiencing statistical data drift.

> *[ SCREENSHOT PLACEHOLDER: Add an image of the 📊 Eval Dashboard showing the Detailed Metrics grid and Composite Score here ]*

---

## 📂 Complete Project Structure

```text
Akarsh_T_P_Banking_Risk_Assessment/
├── src/
│   ├── api/
│   │   └── server.py             # FastAPI backend orchestrator & endpoints
│   ├── frontend/
│   │   └── app.py                # Streamlit UI (Chat, HITL, Prompts, Eval, MCP tabs)
│   ├── agent/
│   │   ├── chain.py              # LCEL AgentExecutor with retry & fallback logic
│   │   ├── tools.py              # Native LangChain tools (DTI, Doc Verify)
│   │   ├── prompts.py            # Few-shot example semantic selector
│   │   └── memory_manager.py     # Session-based conversational memory
│   ├── core/
│   │   ├── schemas.py            # Pydantic schemas (LoanDecision, EvalResult, etc.)
│   │   └── logger_config.py      # Structured JSON application logging
│   ├── mcp/
│   │   ├── client.py             # MCP Client that wraps FastMCP servers
│   │   ├── registry.py           # YAML loader for MCP tool servers
│   │   └── servers/              # External FastMCP python servers
│   │       ├── credit_bureau_server.py
│   │       ├── income_verification_server.py
│   │       ├── property_valuation_server.py
│   │       └── regulatory_feed_server.py
│   ├── hitl/
│   │   ├── manager.py            # Intercepts prompts against YAML HITL rules
│   │   └── store.py              # Local JSON store for pending HITL tasks
│   ├── prompt_manager/
│   │   └── loader.py             # Loads decoupled prompt templates from YAML
│   ├── rbac/
│   │   └── filter.py             # Generates vector DB metadata filters based on role
│   ├── rag/
│   │   ├── ingestion.py          # Document vectorization
│   │   └── retrieval.py          # Vector similarity search
│   └── agents/
│       └── dispatcher.py         # Multi-agent dispatching logic
├── config/
│   ├── hitl_rules.yaml           # Triggers for Human-in-the-loop review
│   ├── mcp_servers.yaml          # Registry of active MCP servers
│   ├── prompts.yaml              # Decoupled agent system prompts
│   └── roles.yaml                # RBAC permission tiers
├── eval/
│   ├── custom_metrics.py         # Citation and Compliance tracking metrics
│   ├── drift.py                  # KL-Divergence data drift monitoring
│   ├── intrinsic.py              # Ragas-style intrinsic retrieval metrics
│   ├── llm_judge.py              # Independent Extrinsic evaluator logic
│   ├── run_eval.py               # The main automated Evaluation Harness
│   └── golden_set.json           # Ground-truth test queries
├── tests/
│   ├── test_report.py            # Automated API integration test runner
│   └── test_regex.py             # Regex extraction unit tests
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup & Run Instructions

### Step 1: Environment Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: API Keys

Create a `.env` file in the root directory and add your API keys:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 4: Start the Backend Server

The FastAPI backend automatically spins up the AI agent, evaluation suite, and connects dynamically to all four MCP servers.

```bash
# Terminal 1
uvicorn src.api.server:app --reload --port 8000
```
> **Note:** Wait approximately 2 minutes for the HuggingFace embeddings and Vector databases to complete their "cold boot". The backend is fully ready when the terminal logs: `Application startup complete.`

### Step 5: Launch the Streamlit Dashboard

```bash
# Terminal 2
streamlit run src/frontend/app.py
```
> The User Interface will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

You can run the standalone automated test script to verify core banking logic and agent routing without using the UI.

1. Ensure the Backend (Terminal 1) is running.
2. In a new terminal, execute:
   ```bash
   python tests/test_report.py
   ```
3. A detailed Markdown report summarizing the results, latency, and tool traces will be generated inside the `test_reports/` directory.
