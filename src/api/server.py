import os
import time
import shutil
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.core.schemas import (
    ChatRequest, ChatResponse, HealthResponse, 
    IngestResponse, JobStatusResponse, 
    RetrievalRequest, RetrievalResponse, SourceListResponse, EvalResult
)
from src.agent.chain import get_underwriter_response
from src.agent.memory_manager import memory_manager
from src.agent.tools import TOOL_NAMES
from src.core.logger_config import app_logger

# --- WEEK 3 IMPORTS ---
from src.rag.config import VECTOR_STORE_TYPE, DATA_DIR
from src.rag.ingestion import create_job, process_document_job, INGESTION_JOBS
from src.rag.retrieval import retrieve_documents
from src.rag.evaluator import run_evaluation_suite
from src.rag.vector_store import get_vector_store
from src.mcp.client import mcp_manager
from src.hitl.store import HITLStore
from src.prompt_manager.loader import prompt_manager
from src.rbac.filter import rbac_manager
from src.agents.dispatcher import agent_dispatcher
from eval.run_eval import main as run_eval_main
from eval.drift import drift_detector
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# Ensure temp data directory exists for uploads
os.makedirs(os.path.join(DATA_DIR, "tmp"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan — startup/shutdown logging
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("🚀 Loan Underwriting API starting up...")
    app_logger.info(f"Available tools: {TOOL_NAMES}")
    app_logger.info(f"Active Vector Store: {VECTOR_STORE_TYPE.upper()}")
    yield
    app_logger.info("🛑 Loan Underwriting API shutting down...")


app = FastAPI(
    title="Loan Underwriting Assistant API",
    description="AI-powered loan risk assessment with Agent + RAG architecture",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# WEEK 2 ENDPOINTS (Core Underwriting)
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    app_logger.info(f"[/chat] session={request.session_id} | message={request.message[:100]}")

    try:
        result = await get_underwriter_response(
            user_text=request.message,
            session_id=request.session_id,
            role_name=request.role_name
        )

        elapsed = time.time() - start_time
        app_logger.info(
            f"[/chat] session={request.session_id} | "
            f"tools={result['tools_used']} | "
            f"time={elapsed:.2f}s"
        )

        return ChatResponse(
            session_id=request.session_id,
            response=result["response"],
            structured_output=result.get("structured_output"),
            tools_used=result.get("tools_used", []),
            citations=result.get("citations", []) # WEEK 3 addition
        )

    except Exception as e:
        app_logger.error(f"[/chat] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@app.post("/reset")
async def reset_endpoint(session_id: str = "default"):
    app_logger.info(f"[/reset] Resetting session: {session_id}")
    was_reset = memory_manager.reset_session(session_id)
    return {
        "status": "success" if was_reset else "not_found",
        "message": f"Session '{session_id}' memory cleared." if was_reset else f"Session '{session_id}' had no active memory.",
        "active_sessions": memory_manager.active_sessions,
    }


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    return HealthResponse(
        status="healthy",
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet"),
        vector_store=VECTOR_STORE_TYPE,
        memory_sessions=memory_manager.active_sessions,
        tools_available=TOOL_NAMES,
    )

# =============================================================================
# WEEK 3 ENDPOINTS (RAG & Knowledge Base)
# =============================================================================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a document and queues it for background RAG ingestion."""
    try:
        # Save file to temporary directory
        temp_path = os.path.join(DATA_DIR, "tmp", f"{uuid.uuid4()}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create a tracking job
        job_id = create_job()
        
        # Fire and forget the ingestion process
        background_tasks.add_task(process_document_job, job_id, temp_path, file.filename)
        
        return IngestResponse(
            job_id=job_id,
            message=f"Upload successful. Ingestion started for {file.filename}."
        )
    except Exception as e:
        app_logger.error(f"Failed to initiate upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
async def get_ingest_status(job_id: str):
    """Poll the status of an ingestion job."""
    if job_id not in INGESTION_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
        
    job = INGESTION_JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        state=job["state"],
        progress=job["progress"],
        message=job["message"],
        error=job.get("error")
    )


@app.post("/retrieve", response_model=RetrievalResponse)
async def pure_retrieval(request: RetrievalRequest):
    """Pure retrieval endpoint (no LLM generation) for testing/debugging."""
    try:
        results = retrieve_documents(
            query=request.query, 
            top_k=request.top_k, 
            session_id=request.session_id
        )
        return RetrievalResponse(query=request.query, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



EVAL_JOBS = {}

@app.post("/evaluate/start")
async def start_evaluate(background_tasks: BackgroundTasks):
    """Starts the standalone evaluation suite in the background."""
    job_id = str(uuid.uuid4())
    EVAL_JOBS[job_id] = {"state": "running", "progress": 0.0, "message": "Starting evaluation...", "result": None, "error": None}
    
    async def run_eval_job():
        try:
            from eval.run_eval import main as run_harness
            import json
            
            def update_progress(current, total, msg):
                EVAL_JOBS[job_id]["progress"] = current / total
                EVAL_JOBS[job_id]["message"] = msg
                
            # Await directly in the same event loop to prevent MCP session deadlocks
            await run_harness(progress_callback=update_progress)
            
            results_path = os.path.join(DATA_DIR, "..", "test_reports", "eval_details_latest.json")
            with open(results_path, "r") as f:
                eval_data = json.load(f)
                
            EVAL_JOBS[job_id]["state"] = "completed"
            EVAL_JOBS[job_id]["result"] = eval_data
            EVAL_JOBS[job_id]["progress"] = 1.0
            EVAL_JOBS[job_id]["message"] = "Evaluation complete."
        except Exception as e:
            app_logger.error(f"Evaluation runner encountered an error: {e}")
            EVAL_JOBS[job_id]["state"] = "failed"
            EVAL_JOBS[job_id]["error"] = str(e)
            EVAL_JOBS[job_id]["message"] = "Failed."
            
    background_tasks.add_task(run_eval_job)
    return {"job_id": job_id}

@app.get("/evaluate/status/{job_id}")
async def eval_status(job_id: str):
    """Poll the status of an evaluation job."""
    if job_id not in EVAL_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return EVAL_JOBS[job_id]


@app.get("/sources", response_model=SourceListResponse)
async def list_sources():
    """Returns basic stats about the indexed knowledge base."""
    # Note: Fetching full doc lists varies wildly between Pinecone, Chroma, and FAISS. 
    # For swappability, we return a high-level summary.
    try:
        # Just verifying DB connection works
        vector_store = get_vector_store()
        return SourceListResponse(sources=[], total_chunks=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WEEK 4 ENDPOINTS (MCP & HITL & RBAC)
# =============================================================================

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all registered MCP tool servers with their capabilities."""
    try:
        tools = await mcp_manager.get_langchain_tools()
        return {"status": "success", "tools": [{"name": t.name, "description": t.description} for t in tools]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MCPInvokeRequest(BaseModel):
    server_name: str
    tool_name: str
    args: dict

@app.post("/mcp/invoke")
async def invoke_mcp_tool(request: MCPInvokeRequest):
    """Invoke a specific MCP tool by name with provided parameters."""
    try:
        result = await mcp_manager.invoke_tool(request.server_name, request.tool_name, request.args)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hitl/pending")
async def list_pending_hitl_tasks():
    """List all pending HITL approval requests."""
    try:
        tasks = HITLStore.get_pending_tasks()
        return {"status": "success", "pending_tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class HITLReviewRequest(BaseModel):
    decision: str
    comments: str
    reviewer: str = "admin"

@app.post("/hitl/review/{task_id}")
async def review_hitl_task(task_id: str, request: HITLReviewRequest):
    """Approve or reject a pending HITL task."""
    try:
        if request.decision not in ["approve", "reject"]:
            raise ValueError("Decision must be 'approve' or 'reject'")
            
        task = HITLStore.resolve_task(task_id, request.decision, request.comments, request.reviewer)
        return {"status": "success", "task": task}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts")
async def list_prompts():
    """List all prompt templates with their metadata."""
    try:
        prompts_meta = {}
        for name, data in prompt_manager.templates.items():
            prompts_meta[name] = {
                "version": data.get("version"),
                "author": data.get("author"),
                "description": data.get("description")
            }
        return {"status": "success", "prompts": prompts_meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/roles")
async def list_roles():
    """List available roles and their document access permissions."""
    try:
        return {"status": "success", "roles": rbac_manager.roles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/context")
async def get_auth_context(role: str = "junior_analyst"):
    """Return the current user's role and accessible document types (Simulated)."""
    try:
        filter_dict = rbac_manager.get_role_filter(role)
        return {"status": "success", "role": role, "active_filters": filter_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval/regression")
async def run_regression_suite():
    """Run regression suite against the golden set."""
    try:
        results = await run_eval_main()
        return {"status": "success", "metrics": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/dispatch")
async def dispatch_agents(request: ChatRequest):
    """Dispatch query to specialized sub-agents."""
    try:
        results = await agent_dispatcher.dispatch(request.message, context={"session_id": request.session_id})
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval/drift")
async def evaluate_drift():
    """Evaluate data drift for credit scores (Simulated)."""
    try:
        # Mocking data for demonstration
        baseline = [650, 700, 720, 680, 710, 690, 750, 800, 620, 640]
        # Updated 'recent' to be nearly identical to 'baseline' to simulate a healthy, non-drifting system
        recent = [655, 700, 715, 680, 710, 695, 745, 800, 625, 640]
        
        results = drift_detector.analyze_credit_score_drift(baseline, recent)
        return {"status": "success", "drift_analysis": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
