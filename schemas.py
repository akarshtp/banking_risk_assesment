from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


# =============================================================================
# WEEK 2: CORE UNDERWRITING SCHEMAS (Unchanged)
# =============================================================================

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    REJECT = "Reject"

class DecisionType(str, Enum):
    APPROVED = "Approved"
    DENIED = "Denied"
    MANUAL_REVIEW = "Manual Review"

class LoanDecision(BaseModel):
    """Primary structured output for underwriting decisions."""
    decision: DecisionType = Field(description="Final loan decision")
    risk_score: RiskLevel = Field(description="Assessed risk level")
    reasoning: List[str] = Field(description="Step-by-step analysis reasoning")
    explanation: str = Field(description="Human-readable summary of the decision")

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Reasoning steps cannot be empty")
        return v

class CreditScoreResult(BaseModel):
    credit_score: int = Field(ge=300, le=900, description="Computed credit score (300-900)")
    risk_bucket: RiskLevel = Field(description="Risk classification bucket")
    factors: List[str] = Field(description="Key factors affecting the score")
    recommendation: str = Field(description="Brief recommendation based on score")

    @field_validator("credit_score")
    @classmethod
    def score_in_range(cls, v):
        if not 300 <= v <= 900:
            raise ValueError("Credit score must be between 300 and 900")
        return v

class DocumentVerificationResult(BaseModel):
    document_type: str = Field(description="Type of document verified (PAN/Aadhaar/PaySlip/ITR)")
    is_valid: bool = Field(description="Whether the document passed verification")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Verification confidence 0-1")
    flags: List[str] = Field(default_factory=list, description="Any mismatch or forgery flags")
    extracted_fields: dict = Field(default_factory=dict, description="OCR-extracted key fields")

class DTIResult(BaseModel):
    monthly_income: float = Field(ge=0, description="Declared monthly income")
    total_monthly_debt: float = Field(ge=0, description="Total monthly debt obligations")
    dti_ratio: float = Field(ge=0, description="Computed DTI ratio as percentage")
    eligibility: str = Field(description="Eligibility verdict")
    max_recommended_loan: float = Field(ge=0, description="Maximum recommended loan amount")
    breakdown: List[str] = Field(description="Calculation breakdown steps")


# =============================================================================
# WEEK 3: RAG & KNOWLEDGE BASE SCHEMAS (New)
# =============================================================================

class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Citation(BaseModel):
    """Represents a traceable reference to a source document chunk."""
    source_id: str = Field(description="Identifier or filename of the source document")
    snippet: str = Field(description="The exact text chunk used by the LLM")
    page_or_section: Optional[str] = Field(default=None, description="Page number or section header")
    relevance_score: Optional[float] = Field(default=None, description="Hybrid search score of the chunk")

class DocumentChunk(BaseModel):
    """Represents a retrieved chunk of text from the vector store."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class SourceDoc(BaseModel):
    """Represents an indexed document in the knowledge base."""
    doc_id: str
    filename: str
    chunk_count: int
    upload_date: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvalResult(BaseModel):
    """Metrics from the evaluation harness."""
    context_precision: float = Field(description="RagAS context precision score")
    context_recall: float = Field(description="RagAS context recall score")
    answer_relevance: float = Field(description="RagAS answer relevance score")
    faithfulness: float = Field(description="RagAS faithfulness score (no hallucinations)")
    end_to_end_quality: float = Field(description="Overall composite score")
    details: Dict[str, Any] = Field(default_factory=dict, description="Per-query evaluation breakdown")


# =============================================================================
# API REQUEST / RESPONSE SCHEMAS
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Session identifier for memory")
    message: str = Field(min_length=1, description="User message")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    structured_output: Optional[LoanDecision] = None
    tools_used: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list, description="Citations from the RAG tool")

class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store: str  # NEW: Tracks which DB is active
    memory_sessions: int
    tools_available: List[str]

class IngestResponse(BaseModel):
    job_id: str = Field(description="Unique tracking ID for the background ingestion job")
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    state: JobState
    progress: float = Field(ge=0.0, le=1.0, description="Completion percentage 0.0 to 1.0")
    message: str
    error: Optional[str] = None

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for LRU cache lookup")

class RetrievalResponse(BaseModel):
    query: str
    results: List[DocumentChunk]

class SourceListResponse(BaseModel):
    sources: List[SourceDoc]
    total_chunks: int