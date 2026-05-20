from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES: Diverse banking/underwriting scenarios
# Each example has an 'input' (user query) and 'output' (ideal assistant response).
# The semantic selector will pick the closest matches at runtime.
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    # --- DTI / Income-Debt Scenarios ---
    {
        "input": "I earn $5000 monthly and have $1500 in debt payments. Can I get a loan?",
        "output": (
            "Let me analyze your application. I'll use the DTI Calculator to assess your eligibility.\n"
            "With $5,000 income and $1,500 debt, your DTI ratio is 30%. This falls in the Medium risk category.\n"
            "Decision: Conditionally Approved — you may qualify for a standard loan product with documentation."
        ),
    },
    {
        "input": "My salary is $8000 and I pay $800 rent and $400 EMI. What's the maximum loan I can get?",
        "output": (
            "I'll calculate your DTI and maximum borrowing capacity.\n"
            "Total monthly obligations: $1,200. DTI ratio: 15% — Low risk, excellent profile.\n"
            "With capacity for additional EMI up to 45% DTI, your maximum recommended loan at 10% for 5 years "
            "is approximately $141,000. Decision: Pre-Approved for premium loan products."
        ),
    },
    {
        "input": "I have $3000 income with $2000 in existing debts. Am I eligible?",
        "output": (
            "Your DTI ratio is 66.7% which significantly exceeds the 45% threshold.\n"
            "Risk Level: Reject. I'd recommend reducing existing obligations before applying.\n"
            "Decision: Denied — DTI ratio too high for any standard loan product."
        ),
    },
    # --- Credit Score Scenarios ---
    {
        "input": "I've never missed a payment and have 8 years of credit history. Check my credit score.",
        "output": (
            "I'll analyze your credit profile using the Credit Score Analyzer.\n"
            "With 100% on-time payments and 8 years of history, your estimated score is 780+ (Low Risk).\n"
            "Factors: Excellent payment history, established credit age.\n"
            "Decision: Strong candidate for pre-approved offers at competitive rates."
        ),
    },
    {
        "input": "I have 3 active loans and 70% credit card utilization. What's my risk level?",
        "output": (
            "Running credit analysis: 3 active loans is moderate, but 70% utilization is concerning.\n"
            "Estimated score: ~580 (High Risk bucket).\n"
            "Recommendation: Reduce credit card balances below 30% before applying for new credit."
        ),
    },
    # --- Document Verification Scenarios ---
    {
        "input": "I need to verify my PAN card ABCDE1234F for the loan application.",
        "output": (
            "I'll run document verification on your PAN card.\n"
            "PAN: ABCDE1234F — Format is valid (matches ABCDE1234F pattern).\n"
            "Verification confidence: 95%. No flags detected. Document cleared for processing."
        ),
    },
    {
        "input": "Verify my Aadhaar number 1234 5678 9012 and pay slip showing $6000 salary.",
        "output": (
            "Running dual document verification:\n"
            "1. Aadhaar XXXX-XXXX-9012: Format valid, 12 digits confirmed.\n"
            "2. Pay Slip: OCR extracted salary matches declared $6,000 within tolerance.\n"
            "Both documents verified successfully. Proceeding with application."
        ),
    },
    # --- Combined / Complex Scenarios ---
    {
        "input": "I earn $10000, have $2000 EMI, credit score around 720. Want a $200K home loan.",
        "output": (
            "Comprehensive assessment:\n"
            "1. DTI: $2,000/$10,000 = 20% — Low Risk ✅\n"
            "2. Credit Score 720: Medium-Low risk bucket, eligible for standard rates.\n"
            "3. Max capacity at 45% DTI: ~$200K over 20yr @ 8.5% — tight but feasible.\n"
            "Decision: Conditionally Approved. Recommend submitting full documentation for final review."
        ),
    },
    # --- WEEK 3: RAG / Knowledge Retrieval Scenario ---
    {
        "input": "What is the bank's policy on minimum credit score for commercial real estate loans?",
        "output": (
            "I will check the bank's internal policy documents for commercial real estate lending guidelines using the knowledge retrieval tool.\n"
            "Based on the commercial lending manual, the minimum acceptable credit score for a commercial real estate loan is 680, provided the business has been operational for at least 3 years.\n"
            "Decision: Manual Review — policy check completed, waiting on applicant data."
        )
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC EXAMPLE SELECTOR
# Embeds all examples and picks top-k closest to the user's actual query.
# Uses HuggingFace sentence-transformers (runs locally, no extra API key).
# ─────────────────────────────────────────────────────────────────────────────
def build_example_selector(k: int = 3):
    """
    Build a SemanticSimilarityExampleSelector using Chroma + HuggingFace embeddings.
    
    Args:
        k: Number of few-shot examples to select per query.
    
    Returns:
        Configured SemanticSimilarityExampleSelector instance.
    """
    # Semantic selection — embedding model runs locally
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Lightweight, fast, good quality
        model_kwargs={"device": "cpu"},
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=FEW_SHOT_EXAMPLES,
        embeddings=embeddings,
        vectorstore_cls=Chroma,
        k=k,  # Select top-k most similar examples
        input_keys=["input"],  # Only use 'input' field for similarity (avoids list-type errors)
    )
    return example_selector


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE BUILDER
# Composes the full prompt: system instructions + few-shot examples + memory + tools
# ─────────────────────────────────────────────────────────────────────────────

# System prompt for the underwriting agent
SYSTEM_PROMPT = """You are a Senior Credit Risk Underwriter AI Agent with access to specialized banking tools.

YOUR RESPONSIBILITIES:
1. Analyze loan applications by gathering income, debt, and credit information.
2. Use your tools to perform precise calculations — NEVER guess numbers.
3. Provide clear, structured decisions: Approved / Denied / Manual Review.
4. Flag any document issues or risk concerns.
5. Search internal knowledge bases for policy clarifications when the user asks general banking questions.

TOOL USAGE GUIDELINES:
- Use `credit_score_analyzer` when evaluating creditworthiness or risk buckets.
- Use `document_verification_engine` when the user provides KYC documents (PAN, Aadhaar, PaySlip, ITR).
- Use `dti_calculator` when income and debt figures are provided.
- Use `knowledge_retrieval` (RAG) when the user asks general questions about loan limits, policies, definitions, or bank rules.
- You may call multiple tools in sequence for comprehensive assessments (e.g., look up a policy, then calculate DTI).

RESPONSE FORMAT:
Always structure your final response with:
- **Decision**: Approved / Denied / Manual Review
- **Risk Level**: Low / Medium / High / Reject
- **Key Findings**: Bullet points of analysis
- **Recommendation**: Clear next steps

If the user hasn't provided enough information, ask specific follow-up questions.
Be professional, thorough, and empathetic."""


def build_prompt(example_selector=None):
    """
    Build the full ChatPromptTemplate with few-shot examples and memory placeholders.
    
    Args:
        example_selector: SemanticSimilarityExampleSelector (if None, no few-shot).
    
    Returns:
        ChatPromptTemplate ready for the agent.
    """
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    if example_selector:
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            input_variables=["input"],
        )

        # Full prompt with: system → few-shot → memory → user input → agent scratchpad
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            few_shot_prompt,                           # Injected few-shot examples
            MessagesPlaceholder("chat_history"),       # Memory placeholder
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),   # Agent tool calls go here
        ])
    else:
        # Fallback: no few-shot examples
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

    return full_prompt
