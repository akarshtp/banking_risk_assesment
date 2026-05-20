import os
from dotenv import load_dotenv

load_dotenv()

# --- Embedding & Model Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# NEW: Embedding Adapter Switch ('openai', 'huggingface', 'cohere')
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --- Vector Store Routing ---
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "banking-risk-rag")

# Local DB Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- CRITICAL UPDATED CHUNKING CONFIGURATION ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# NEW: Chunking Strategy Selector ('recursive', 'semantic')
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive").lower()