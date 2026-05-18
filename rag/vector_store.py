import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag.config import (
    VECTOR_STORE_TYPE,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    CHROMA_PERSIST_DIR,
    FAISS_INDEX_PATH
)
from logger_config import app_logger

def get_embeddings():
    """Initialize OpenAI embeddings."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def get_vector_store():
    """
    Factory function to return the configured Vector Store.
    Swaps seamlessly between Pinecone, FAISS, and Chroma based on .env config.
    """
    embeddings = get_embeddings()

    if VECTOR_STORE_TYPE == "pinecone":
        app_logger.info(f"Initializing Pinecone vector store (Index: {PINECONE_INDEX_NAME})")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set for Pinecone.")
            
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists, create if not (AWS us-east-1 serverless)
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            app_logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # Standard dimension for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be provisioned
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
                
        return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    elif VECTOR_STORE_TYPE == "faiss":
        app_logger.info(f"Initializing FAISS vector store at {FAISS_INDEX_PATH}")
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        
        # FAISS needs to load an existing index or be initialized with dummy data
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Initialize an empty FAISS index and save it
            app_logger.info("Creating new FAISS index.")
            empty_store = FAISS.from_texts(["_init_"], embeddings)
            empty_store.save_local(FAISS_INDEX_PATH)
            return empty_store

    else:
        # Default to Chroma
        app_logger.info(f"Initializing Chroma vector store at {CHROMA_PERSIST_DIR}")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        return Chroma(
            collection_name="banking_risk",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )