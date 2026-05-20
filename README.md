#  Loan Underwriting Assistant

An AI-powered loan risk assessment chatbot built with **LangChain**, **Claude (Anthropic)**, **FastAPI**, and **Streamlit**.

The assistant analyzes loan applications using 3 specialized tools — credit scoring, document verification, and DTI calculation — and returns structured underwriting decisions.

---

## Project Structure

```
loan_underwriter_project/
├── src/
│   ├── api/
│   │   └── server.py         # FastAPI backend server (/chat, /reset, /health)
│   ├── frontend/
│   │   └── app.py            # Streamlit chat frontend (calls the API)
│   ├── agent/
│   │   ├── chain.py          # Core LangChain agent 
│   │   ├── tools.py          # 4 custom tools (Credit Score, Doc Verify, DTI, Retrieval)
│   │   ├── prompts.py        # Few-shot prompt templates with semantic selection
│   │   └── memory_manager.py # Multi-turn conversation memory (10 turns)
│   ├── core/
│   │   ├── schemas.py        # Pydantic models for all inputs and outputs
│   │   └── logger_config.py  # Structured JSON logging setup
│   └── rag/
│       ├── config.py         # RAG configuration
│       ├── ingestion.py      # Document ingestion pipeline
│       ├── retrieval.py      # Document retrieval pipeline
│       ├── vector_store.py   # Vector DB initialization
│       └── evaluator.py      # Evaluation suite logic
├── eval/
│   ├── intrinsic.py          # Intrinsic RAG metrics
│   ├── llm_judge.py          # LLM as a Judge logic
│   ├── run_eval.py           # Evaluation runner script
│   └── golden_set.json       # Golden dataset for evaluation
├── tests/
│   └── test_report.py        # Test runner — sends 10 queries (reduced from 20), generates report
├── data/                     # Vector databases and sample data
├── requirements.txt          # Python dependencies
├── .env                      # API key 
├── README.md             
├── logs/                     # Auto-created at runtime
│   ├── app.log               # System events log (JSON)
│   └── interactions.log      # All chat interactions log (JSON)
└── test_reports/             # Auto-created when tests run
    ├── test_results_<timestamp>.json   # Timestamped raw results
    ├── test_report_<timestamp>.md      # Timestamped readable report
    ├── test_results_latest.json        # Most recent run (overwritten)
    └── test_report_latest.md           # Most recent run (overwritten)
```

---

##  What Each File Does

| File | Purpose |
|---|---|
| **`api.py`** | FastAPI server with 3 endpoints. `POST /chat` sends user messages to the agent. `POST /reset` clears session memory. `GET /health` returns system status. This is the backend that the frontend talks to. |
| **`app.py`** | Streamlit chat UI. Shows message history, structured decision cards, tools used per response, and a sidebar with session controls, health status, and example queries. Communicates with `api.py` via HTTP. |
| **`chain.py`** | The brain of the project. Runs the guardrail check (on-topic filter), builds the LangChain agent with tools and memory, handles retry logic (3 attempts with backoff), falls back to a lighter model on failure, and parses structured output. |
| **`tools.py`** | Defines 3 LangChain tools: (1) **Credit Score Analyzer** — computes a simulated CIBIL/FICO score from 5 weighted factors and classifies into risk buckets. (2) **Document Verification Engine** — validates PAN, Aadhaar, PaySlip, ITR formats and flags mismatches. (3) **DTI Calculator** — computes debt-to-income ratio and maximum recommended loan amount. |
| **`prompts.py`** | Contains 8 few-shot examples covering DTI, credit, document, and combined scenarios. Uses `SemanticSimilarityExampleSelector` with HuggingFace embeddings + Chroma to pick the 3 most relevant examples per query at runtime. Builds the full prompt template. |
| **`schemas.py`** | Pydantic models for every data structure: `LoanDecision`, `CreditScoreResult`, `DocumentVerificationResult`, `DTIResult`, `ChatRequest`, `ChatResponse`, `HealthResponse`. Includes validators and enums for strict type checking. |
| **`memory_manager.py`** | Manages per-session conversation memory using LangChain's `ConversationBufferWindowMemory` with `k=10`. Each session ID gets its own isolated memory. Supports reset and listing active sessions. |
| **`logger_config.py`** | Sets up two structured JSON loggers that run automatically in the background. `logs/app.log` captures system events (startup, errors, tool calls). `logs/interactions.log` records every user ↔ assistant chat exchange with session ID, tools used, and response data. This is the always-on audit trail for Objective 9. |
| **`test_report.py`** | Automated test runner. Sends 10 predefined queries (reduced from 20 to prevent API timeouts) to the API covering FAQ, guardrails, tools, and multi-turn conversations. Records pass/fail status, tools used, and response times. Generates both a JSON results file and a Markdown report inside the `test_reports/` folder. |

---

##  Setup & Run



### Step 1: Create a virtual environment (recommended)

```bash
python -m venv venv

# On Mac/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add your API key

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### Step 5: Start the FastAPI backend

```bash
uvicorn src.api.server:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     🚀 Loan Underwriting API starting up...
```

### Step 6: Start the Streamlit frontend (new terminal)

```bash
streamlit run src/frontend/app.py
```

Opens in browser at `http://localhost:8501`.

---

##  Running the Test Report

The test report sends 10 queries (reduced from the original 20 to prevent API timeouts and rate limits) to the API and records the results automatically. Note that since the sample size is reduced, the percentage pass rate will shift more dramatically with a single failure (1 failure = 90% instead of 95%).

### Prerequisites

The FastAPI backend **must be running** before you run the tests:

```bash
# Terminal 1 — keep this running
uvicorn src.api.server:app --reload --port 8000
```

### Run the tests

```bash
# Terminal 2
python tests/test_report.py
```

### What you'll see in the terminal

```
🏥 Checking API health...
   Status: healthy
   Model:  claude-sonnet-4-20250514
   Tools:  ['credit_score_analyzer', 'document_verification_engine', 'dti_calculator']

======================================================================
🧪 LOAN UNDERWRITING ASSISTANT — TEST REPORT
   Date: 2025-01-15 14:30:22
   API:  http://localhost:8000
   Total Queries: 20
======================================================================

──────────────────────────────────────────────────
📂 Category: FAQ
──────────────────────────────────────────────────
  🔄 Query 1: What is a debt-to-income ratio and why does it matter for loans?...
  ✅ PASS | Time: 3.21s | Tools: []

  🔄 Query 2: What credit score is considered good for a home loan?...
  ✅ PASS | Time: 2.89s | Tools: []

... (continues for all 20 queries) ...

======================================================================
📊 FINAL RESULTS: 19/20 passed (95.0%)
======================================================================

📁 Saving reports to: test_reports/
  💾 JSON saved: test_reports/test_results_20250115_143022.json
  💾 JSON saved: test_reports/test_results_latest.json
  📄 Markdown saved: test_reports/test_report_20250115_143022.md
  📄 Markdown saved: test_reports/test_report_latest.md

✅ All done!
```

### Where to find the results

After running, check the `test_reports/` folder:

```
test_reports/
├── test_results_20250115_143022.json   ← Permanent record with timestamp
├── test_report_20250115_143022.md      ← Permanent record with timestamp
├── test_results_latest.json            ← Quick access (overwritten each run)
└── test_report_latest.md              ← Quick access (overwritten each run)
```

### How to read the results

**Option 1: Open the Markdown report** (recommended for reading)

```bash
# Open in any Markdown viewer, or just read it in terminal
cat test_reports/test_report_latest.md
```

The Markdown report contains:
- **Summary table** — total queries, pass/fail count, pass rate, average response time
- **Category breakdown** — results grouped by FAQ, Guardrail, Credit Score, Document Verification, DTI Calculator, Multi-Turn
- **Detailed results** — each query with expected behavior, actual response, tools used, and pass/fail status

**Option 2: Open the JSON file** (for programmatic access)

```bash
cat test_reports/test_results_latest.json
```

The JSON file contains:
- `report_metadata` — date, total queries, passed, failed, pass rate
- `results` — array of 20 objects, each with query, response, tools_used, passed, failure_reason

**Option 3: View in VS Code**

```bash
code test_reports/test_report_latest.md
```

VS Code will render the Markdown with formatted tables and headings.

### The test queries cover (originally 20 queries, now limited to 10 for testing stability)

| Queries | Category | What It Tests |
|---|---|---|
| 1–5 | FAQ | General banking questions answered without tools |
| 6–8 | Guardrail | Off-topic queries (recipes, sports, coding) get blocked |
| 9–11 | Credit Score Analyzer | Low, Medium, and High/Reject risk profiles |
| 12–14 | Document Verification | Valid PAN, valid Aadhaar, invalid/forged documents |
| 15–17 | DTI Calculator | Eligible, not eligible, and max loan computation |
| 18–20 | Multi-Turn | 3 sequential messages in same session testing memory recall |

---



---

##  Objectives Covered

| # | Objective | Where |
|---|---|---|
| 1 | LangChain-based architecture with clean chain composition | `chain.py` — agent + pipe operator chains |
| 2 | Conversational memory (10+ turns) | `memory_manager.py` — `ConversationBufferWindowMemory(k=10)` |
| 3 | 3 Custom tools | `tools.py` — Credit Score, Doc Verification, DTI Calculator |
| 4 | Few-shot prompts with semantic example selection | `prompts.py` — `SemanticSimilarityExampleSelector` + Chroma |
| 5 | Structured output parsing with Pydantic | `schemas.py` — validated models; `chain.py` — `parse_structured_output()` |
| 6 | FastAPI backend (`/chat`, `/reset`, `/health`) | `api.py` |
| 7 | Streamlit frontend with chat interface | `app.py` |
| 8 | Error handling with retry logic and fallback chains | `chain.py` — `tenacity` retry + fallback LLM |
| 9 | Structured JSON logging | `logger_config.py` — logs to `logs/app.log` and `logs/interactions.log` |

---
