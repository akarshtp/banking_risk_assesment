import os
import uuid
import time
from typing import Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredHTMLLoader, TextLoader,
    UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Ensure you run: pip install langchain-experimental
from langchain_experimental.text_splitter import SemanticChunker 

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_STRATEGY
from rag.vector_store import get_embeddings, get_vector_store
from schemas import JobState
from logger_config import app_logger

INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    INGESTION_JOBS[job_id] = {
        "state": JobState.PENDING,
        "progress": 0.0,
        "message": "Job created. Waiting to process...",
        "error": None
    }
    return job_id

def update_job(job_id: str, state: JobState, progress: float, message: str, error: str = None):
    if job_id in INGESTION_JOBS:
        INGESTION_JOBS[job_id].update({"state": state, "progress": progress, "message": message, "error": error})

def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(file_path)
    elif ext == ".csv": return CSVLoader(file_path)
    elif ext in [".html", ".htm"]: return UnstructuredHTMLLoader(file_path)
    elif ext == ".md": return UnstructuredMarkdownLoader(file_path)
    elif ext in [".doc", ".docx"]: return UnstructuredWordDocumentLoader(file_path)
    else: return TextLoader(file_path)

def process_document_job(job_id: str, file_path: str, original_filename: str):
    """Core ingestion pipeline managing dual splitting strategies."""
    try:
        update_job(job_id, JobState.RUNNING, 0.1, f"Loading file: {original_filename}")
        loader = get_loader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No text could be extracted from the document.")

        update_job(job_id, JobState.RUNNING, 0.4, "Document loaded. Executing chunking strategy...")

        # DUAL CHUNKING METHOD SELECTION
        if CHUNK_STRATEGY == "semantic":
            app_logger.info("Executing Semantic Chunking pipeline...")
            embeddings_engine = get_embeddings()
            text_splitter = SemanticChunker(
                embeddings_engine, 
                breakpoint_threshold_type="percentile"
            )
            chunks = text_splitter.split_documents(documents)
        else:
            app_logger.info(f"Executing Recursive Character splitting (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

        # Inject unified citation tracking metadata
        doc_id = str(uuid.uuid4())
        upload_date = time.strftime("%Y-%m-%d %H:%M:%S")
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_id": original_filename,
                "doc_id": doc_id,
                "chunk_index": i,
                "upload_date": upload_date
            })

        update_job(job_id, JobState.RUNNING, 0.7, f"Generated {len(chunks)} chunks. Syncing to Database...")
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        if os.path.exists(file_path):
            os.remove(file_path)

        update_job(job_id, JobState.COMPLETED, 1.0, f"Successfully indexed {len(chunks)} chunks using '{CHUNK_STRATEGY}' strategy.")

    except Exception as e:
        update_job(job_id, JobState.FAILED, 1.0, "Ingestion process failed.", error=str(e))
        if os.path.exists(file_path):
            os.remove(file_path)