import os
import uuid
import time
from typing import Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_vector_store
from schemas import JobState
from logger_config import app_logger

# =============================================================================
# IN-MEMORY JOB TRACKER
# Tracks background tasks so the /ingest/status API can report progress
# =============================================================================
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

def create_job() -> str:
    """Create a new pending job and return its ID."""
    job_id = str(uuid.uuid4())
    INGESTION_JOBS[job_id] = {
        "state": JobState.PENDING,
        "progress": 0.0,
        "message": "Job created. Waiting to process...",
        "error": None
    }
    return job_id

def update_job(job_id: str, state: JobState, progress: float, message: str, error: str = None):
    """Update the status of an existing job."""
    if job_id in INGESTION_JOBS:
        INGESTION_JOBS[job_id].update({
            "state": state,
            "progress": progress,
            "message": message,
            "error": error
        })
        if error:
            app_logger.error(f"[Job {job_id}] {message} - Error: {error}")
        else:
            app_logger.info(f"[Job {job_id}] {message} ({progress*100:.0f}%)")


# =============================================================================
# DOCUMENT LOADERS & CHUNKERS
# =============================================================================
def get_loader(file_path: str):
    """Route to the correct LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(file_path)
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path)
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    else:
        # Fallback to generic unstructured loader for things like JSON/Images (via OCR if configured)
        from langchain_community.document_loaders import UnstructuredFileLoader
        return UnstructuredFileLoader(file_path)


# =============================================================================
# BACKGROUND PROCESSING PIPELINE
# =============================================================================
def process_document_job(job_id: str, file_path: str, original_filename: str):
    """
    The core ingestion pipeline intended to run as a FastAPI BackgroundTask.
    Load -> Chunk -> Embed -> Store.
    """
    try:
        update_job(job_id, JobState.RUNNING, 0.1, f"Loading file: {original_filename}")
        
        # 1. Load the document
        loader = get_loader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No text could be extracted from the document.")

        update_job(job_id, JobState.RUNNING, 0.4, f"Loaded {len(documents)} pages/sections. Chunking...")

        # 2. Chunking (Semantic with Recursive Fallback)
        # We use paragraph (\n\n) and sentence (\n, .) boundaries first to preserve semantic meaning,
        # only splitting words if a single block is massive.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Inject custom metadata for traceability
        doc_id = str(uuid.uuid4())
        upload_date = time.strftime("%Y-%m-%d %H:%M:%S")
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_id": original_filename,
                "doc_id": doc_id,
                "chunk_index": i,
                "upload_date": upload_date
            })

        update_job(job_id, JobState.RUNNING, 0.7, f"Created {len(chunks)} chunks. Pushing to Vector Store...")

        # 3. Store in Vector Database
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        # Cleanup the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        update_job(job_id, JobState.COMPLETED, 1.0, f"Successfully indexed {len(chunks)} chunks from {original_filename}.")

    except Exception as e:
        update_job(job_id, JobState.FAILED, 1.0, "Ingestion failed.", error=str(e))
        # Ensure cleanup even on failure
        if os.path.exists(file_path):
            os.remove(file_path)