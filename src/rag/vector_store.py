import os
import time
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Ensure you run: pip install langchain-cohere
from langchain_cohere import CohereEmbeddings 
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.rag.config import (
    VECTOR_STORE_TYPE,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    COHERE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    CHROMA_PERSIST_DIR,
    FAISS_INDEX_PATH
)
from src.core.logger_config import app_logger

def get_embeddings():
    """
    EMBEDDING ADAPTER: Swaps seamlessly between OpenAI, 
    HuggingFace, and Cohere providers based on .env configuration.
    """
    if EMBEDDING_PROVIDER == "huggingface":
        app_logger.info("Using HuggingFace Embedding Provider (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    elif EMBEDDING_PROVIDER == "cohere":
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not configured in environment.")
        app_logger.info("Using Cohere Embedding Provider (embed-english-v3)")
        return CohereEmbeddings(model="embed-english-v3", cohere_api_key=COHERE_API_KEY)
    else:
        # Default to OpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured in environment.")
        app_logger.info(f"Using OpenAI Embedding Provider ({EMBEDDING_MODEL})")
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def get_vector_store():
    """Factory function returning the active Vector Store instance."""
    embeddings = get_embeddings()

    if VECTOR_STORE_TYPE == "pinecone":
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set for Pinecone.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            app_logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            # Determine dimensions based on adapter selection
            if EMBEDDING_PROVIDER == "huggingface":
                dimension = 384
            elif EMBEDDING_PROVIDER == "cohere":
                dimension = 1024
            else:
                dimension = 1536 # OpenAI
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
        return PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    elif VECTOR_STORE_TYPE == "faiss":
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            empty_store = FAISS.from_texts(["_init_"], embeddings)
            empty_store.save_local(FAISS_INDEX_PATH)
            return empty_store
    else:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        return Chroma(
            collection_name="banking_risk",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
