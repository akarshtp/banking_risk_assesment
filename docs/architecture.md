# System Architecture

## Overview
The Loan Risk Assessment system is built on a modern AI stack using **FastAPI** for the backend, **Streamlit** for the frontend, and **LangChain / LangGraph** for orchestration.

## Components
1. **API Layer (`src/api/server.py`)**: Exposes REST endpoints for Chat, RAG ingestion, HITL approvals, MCP tool invocations, and Evaluations.
2. **Agent Orchestration (`src/agent/chain.py`)**: Uses LangChain Expression Language (LCEL) to chain prompts, LLMs, and tools.
3. **Knowledge Base (`src/rag/`)**: Integrates with Vector DBs (Chroma/FAISS) for Retrieval-Augmented Generation.
4. **Tool Servers (`src/mcp/`)**: Uses the Model Context Protocol (MCP) to interact with mock external systems like Credit Bureaus.
5. **Human-In-The-Loop (`src/hitl/`)**: Persistent store for tasks that trigger manual review based on risk thresholds.

## Deployment
Docker-compose is used to containerize the API and UI. See `deployment.md` for more details.
