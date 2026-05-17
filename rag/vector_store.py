import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding & Model Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector Store Routing ---
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "banking-risk-rag")

# Local DB Paths (for Chroma and FAISS)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- Chunking Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200