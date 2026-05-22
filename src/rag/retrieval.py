import os
import hashlib
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from src.rag.config import EMBEDDING_MODEL
from src.rag.vector_store import get_vector_store
from src.core.logger_config import app_logger
from src.rbac.filter import rbac_manager

# =============================================================================
# MODELS & RE-RANKER INITIALIZATION
# =============================================================================
# We use MS-MARCO for fast, highly accurate cross-encoder re-ranking
try:
    app_logger.info("Loading CrossEncoder re-ranker...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    app_logger.error(f"Failed to load CrossEncoder: {e}")
    reranker = None

# =============================================================================
# IN-MEMORY LRU CACHE (Session-based)
# =============================================================================
# Structure: { "session_id": { "query_hash": [results] } }
RETRIEVAL_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

def get_query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()

# =============================================================================
# MULTI-QUERY TRANSFORMATION
# =============================================================================
def generate_multi_queries(original_query: str) -> List[str]:
    """Uses the LLM to generate 2 alternative phrasings of the query for better recall."""
    try:
        provider = os.getenv("PRIMARY_PROVIDER", "anthropic").lower()
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=100,
                temperature=0.2
            )
        else:
            llm = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=100,
                temperature=0.2
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Your task is to generate 2 different alternative phrasings of the user's query to help retrieve relevant documents from a vector database. Return ONLY a comma-separated list of the 2 new queries."),
            ("human", "{query}")
        ])
        
        chain = prompt | llm | CommaSeparatedListOutputParser()
        alternatives = chain.invoke({"query": original_query})
        
        # Combine original + alternatives, removing duplicates
        all_queries = list(set([original_query] + [q.strip() for q in alternatives if q.strip()]))
        return all_queries
    except Exception as e:
        app_logger.warning(f"Multi-query generation failed, falling back to original query: {e}")
        return [original_query]


# =============================================================================
# CORE RETRIEVAL PIPELINE
# =============================================================================
def retrieve_documents(query: str, top_k: int = 5, session_id: str = None, role_name: str = "junior_analyst") -> List[Dict[str, Any]]:
    """
    Executes the full RAG retrieval pipeline:
    Cache Check -> Multi-Query -> Vector Search (w/ RBAC filter) -> Cross-Encoder Re-rank
    """
    # 1. Check Cache
    if session_id:
        q_hash = get_query_hash(query)
        if session_id not in RETRIEVAL_CACHE:
            RETRIEVAL_CACHE[session_id] = {}
        if q_hash in RETRIEVAL_CACHE[session_id]:
            app_logger.info(f"[/retrieve] Cache hit for session {session_id}")
            return RETRIEVAL_CACHE[session_id][q_hash][:top_k]

    vector_store = get_vector_store()
    
    # 2. Multi-Query Expansion
    search_queries = generate_multi_queries(query)
    app_logger.info(f"[/retrieve] Expanded to {len(search_queries)} queries: {search_queries}")
    
    # 3. Broad Dense Retrieval
    candidate_docs_map = {}
    metadata_filter = rbac_manager.get_role_filter(role_name)
    # Fetch top_k * 2 candidates for EACH variation to give the re-ranker a large pool
    for q in search_queries:
        docs = vector_store.similarity_search(q, k=top_k * 2, filter=metadata_filter if metadata_filter else None)
        for doc in docs:
            # Deduplicate by content to avoid scoring the same chunk twice
            candidate_docs_map[doc.page_content] = doc

    candidate_docs = list(candidate_docs_map.values())
    
    if not candidate_docs:
        app_logger.info("[/retrieve] No documents found.")
        return []

    # 4. Cross-Encoder Re-ranking
    results = []
    if reranker:
        app_logger.info(f"[/retrieve] Re-ranking {len(candidate_docs)} candidates...")
        # CrossEncoder expects pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = reranker.predict(pairs)
        
        # Combine docs with their new relevance scores
        doc_scores = list(zip(candidate_docs, scores))
        # Sort descending by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter and map top_k
        for doc, score in doc_scores[:top_k]:
            # NEW: Ignore completely irrelevant documents (negative logits)
            if score < -5.0:
                continue
                
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # Re-ranker logit score
            })
    else:
        # Fallback if Re-ranker failed to load
        app_logger.warning("[/retrieve] Re-ranker unavailable, using raw dense scores.")
        for doc in candidate_docs[:top_k]:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 
            })

    # 5. Update Cache
    if session_id:
        RETRIEVAL_CACHE[session_id][q_hash] = results
        # Prevent cache bloat (LRU-ish behavior, cap at 20 queries per session)
        if len(RETRIEVAL_CACHE[session_id]) > 20:
            first_key = list(RETRIEVAL_CACHE[session_id].keys())[0]
            del RETRIEVAL_CACHE[session_id][first_key]

    return results
