# Deployment Guide

This document outlines how to deploy the Loan Underwriting Assistant via Docker. By containerizing the application, you eliminate the need to install local Python dependencies and ensure that "it works on my machine" translates to "it works everywhere."

## Prerequisites
- **Docker Engine** (v20.10+)
- **Docker Compose** (v2.0+)
- **Make** (optional, but convenient)

## 1. Environment Setup
Copy the example environment file (if available) or create a new `.env` file in the project root:

```bash
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional overrides (the defaults are usually fine)
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_STORE_TYPE=chroma
```

## 2. Standard Deployment (Development / Production)

The main deployment orchestrates the **API** (FastAPI) and the **UI** (Streamlit).

### Using Make (Recommended)
We've included a `Makefile` to simplify common operations:

```bash
# Build the Docker images
make build

# Start the services in the background
make up

# Tail the logs
make logs
```

### Using standard Docker Compose
If you prefer not to use `Make`:

```bash
# Build and start services in detached mode
docker compose up -d --build

# View logs
docker compose logs -f
```

### Verification
- **API Status**: Visit `http://localhost:8000/health`
- **Frontend UI**: Open `http://localhost:8501` in your browser

---

## 3. Persistent Volumes
The `docker-compose.yml` mounts three local directories into the containers to ensure data persists across restarts:
- `./data` ➜ `/app/data`: Stores the vector databases (e.g., ChromaDB) and any temporary ingested files.
- `./logs` ➜ `/app/logs`: Stores application and interaction JSON logs.
- `./test_reports` ➜ `/app/test_reports`: Stores generated Markdown and JSON evaluation reports.

If you ever need to completely wipe the system state, you can clear the `./data/chroma_db` directory locally, and the next startup will reinitialize an empty index.

---

## 4. Running the RAG Evaluation Suite
The project includes a separate container specifically tuned for running one-shot LLM-as-a-judge evaluations against the Golden Dataset. This ensures your RAG pipeline's metrics don't silently degrade.

To run an evaluation pass:

```bash
make eval
# OR
docker compose -f docker-compose.eval.yml up --build
```
This spawns the `loan_underwriter_eval` container, executes the full test suite, saves the reports to `./test_reports`, and then gracefully exits.

---

## 5. Teardown & Cleanup

To stop the services safely:

```bash
make down
# OR
docker compose down
```

To perform a complete wipe of containers, images, and temporary files:

```bash
make clean
```
*(Note: This does not delete your `chroma_db` index, it only cleans up Docker artifacts and tmp files).*
