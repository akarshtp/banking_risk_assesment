import json
import random
from typing import Optional
from langchain_core.tools import tool

from schemas import CreditScoreResult, DocumentVerificationResult, DTIResult

# --- WEEK 3 IMPORTS ---
from rag.retrieval import retrieve_documents
from logger_config import app_logger


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: Credit Score Analyzer
# Fetches and interprets credit bureau data (CIBIL/FICO), computes composite
# risk scores, and classifies applicants into risk buckets.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def credit_score_analyzer(
    name: str,
    pan_number: Optional[str] = None,
    existing_loans: int = 0,
    credit_card_utilization: float = 0.0,
    payment_history_pct: float = 100.0,
    credit_age_years: float = 1.0,
) -> str:
    """
    Analyzes applicant credit score based on credit bureau parameters.
    
    Args:
        name: Applicant's full name.
        pan_number: PAN card number for CIBIL lookup (optional, simulated).
        existing_loans: Number of currently active loans.
        credit_card_utilization: Credit card utilization ratio (0-100%).
        payment_history_pct: Percentage of on-time payments (0-100%).
        credit_age_years: Length of credit history in years.
    
    Returns:
        JSON string with credit score, risk bucket, factors, and recommendation.
    """
    # --- Simulated CIBIL/FICO Score Computation ---
    # Base score starts at 600; modifiers applied per factor
    base_score = 600

    # Factor 1: Payment history (35% weight — most important)
    payment_modifier = (payment_history_pct - 70) * 3.5  # Range: -245 to +105
    
    # Factor 2: Credit utilization (30% weight)
    utilization_modifier = (50 - credit_card_utilization) * 2.5  # Lower is better
    
    # Factor 3: Credit age (15% weight)
    age_modifier = min(credit_age_years * 15, 100)  # Capped at 100
    
    # Factor 4: Number of active loans (10% weight — too many is risky)
    loan_modifier = max(50 - (existing_loans * 20), -100)
    
    # Factor 5: Credit mix / randomized bureau data (10% weight)
    mix_modifier = random.randint(-20, 40)

    raw_score = base_score + payment_modifier + utilization_modifier + age_modifier + loan_modifier + mix_modifier
    credit_score = int(max(300, min(900, raw_score)))  # Clamp 300-900

    # --- Risk Bucket Classification ---
    factors = []
    if credit_score >= 750:
        risk_bucket = "Low"
        recommendation = "Excellent credit profile. Pre-approved for premium loan products."
    elif credit_score >= 650:
        risk_bucket = "Medium"
        recommendation = "Fair credit profile. Standard loan products available with documentation."
    elif credit_score >= 550:
        risk_bucket = "High"
        recommendation = "Below-average credit. Consider secured loan options or co-applicant."
    else:
        risk_bucket = "Reject"
        recommendation = "Poor credit profile. Loan application not recommended at this time."

    # --- Build factor explanations ---
    if payment_history_pct < 80:
        factors.append(f"⚠ Poor payment history ({payment_history_pct}% on-time)")
    else:
        factors.append(f"✅ Good payment history ({payment_history_pct}% on-time)")

    if credit_card_utilization > 50:
        factors.append(f"⚠ High credit utilization ({credit_card_utilization}%)")
    else:
        factors.append(f"✅ Healthy credit utilization ({credit_card_utilization}%)")

    if credit_age_years < 2:
        factors.append(f"⚠ Short credit history ({credit_age_years} years)")
    else:
        factors.append(f"✅ Established credit history ({credit_age_years} years)")

    if existing_loans > 3:
        factors.append(f"⚠ Multiple active loans ({existing_loans})")
    else:
        factors.append(f"✅ Manageable active loans ({existing_loans})")

    result = CreditScoreResult(
        credit_score=credit_score,
        risk_bucket=risk_bucket,
        factors=factors,
        recommendation=recommendation,
    )
    return result.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: Document Verification Engine
# Validates uploaded KYC documents (PAN, Aadhaar, pay slips, ITR) against
# OCR-extracted fields and flags mismatches or forgery indicators.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def document_verification_engine(
    document_type: str,
    document_number: str,
    applicant_name: str,
    declared_income: Optional[float] = None,
    employer_name: Optional[str] = None,
) -> str:
    """
    Validates KYC documents by simulating OCR extraction and cross-verification.
    
    Args:
        document_type: Type of document — one of 'PAN', 'Aadhaar', 'PaySlip', 'ITR'.
        document_number: The document identification number.
        applicant_name: Name of the applicant for cross-check.
        declared_income: Income declared by applicant (for PaySlip/ITR verification).
        employer_name: Employer name (for PaySlip verification).
    
    Returns:
        JSON string with verification status, confidence, flags, and extracted fields.
    """
    flags = []
    extracted_fields = {}
    confidence = 0.95  # Default high confidence

    doc_type_upper = document_type.upper().replace(" ", "")

    # --- PAN Card Verification ---
    if doc_type_upper == "PAN":
        # PAN format: ABCDE1234F (5 letters, 4 digits, 1 letter)
        is_valid_format = (
            len(document_number) == 10
            and document_number[:5].isalpha()
            and document_number[5:9].isdigit()
            and document_number[9].isalpha()
        )
        extracted_fields = {
            "pan_number": document_number.upper(),
            "name_on_pan": applicant_name.upper(),
            "format_valid": is_valid_format,
        }
        if not is_valid_format:
            flags.append("INVALID_PAN_FORMAT: Does not match ABCDE1234F pattern")
            confidence = 0.3
        # Simulate name-mismatch check
        if len(applicant_name) < 3:
            flags.append("NAME_TOO_SHORT: Possible data entry error")
            confidence -= 0.15

    # --- Aadhaar Verification ---
    elif doc_type_upper in ("AADHAAR", "AADHAR"):
        is_valid_format = len(document_number.replace(" ", "")) == 12 and document_number.replace(" ", "").isdigit()
        extracted_fields = {
            "aadhaar_number": document_number[-4:].rjust(12, "X"),  # Masked
            "name_on_aadhaar": applicant_name,
            "format_valid": is_valid_format,
        }
        if not is_valid_format:
            flags.append("INVALID_AADHAAR_FORMAT: Must be 12 digits")
            confidence = 0.25
        # Simulate duplicate check
        if document_number.replace(" ", "") in ("123456789012", "000000000000"):
            flags.append("FORGERY_INDICATOR: Known test/fake Aadhaar number")
            confidence = 0.1

    # --- Pay Slip Verification ---
    elif doc_type_upper in ("PAYSLIP", "PAY_SLIP", "SALARY_SLIP"):
        ocr_income = declared_income * random.uniform(0.92, 1.08) if declared_income else 0
        income_mismatch = abs(ocr_income - (declared_income or 0)) / max(declared_income or 1, 1) > 0.1
        extracted_fields = {
            "ocr_extracted_income": round(ocr_income, 2),
            "declared_income": declared_income,
            "employer": employer_name or "Not Provided",
            "month": "Latest",
        }
        if income_mismatch:
            flags.append(
                f"INCOME_MISMATCH: OCR extracted ${ocr_income:.0f} vs declared ${declared_income:.0f}"
            )
            confidence -= 0.3
        if not employer_name:
            flags.append("MISSING_EMPLOYER: Employer name not provided for cross-check")
            confidence -= 0.1

    # --- ITR Verification ---
    elif doc_type_upper == "ITR":
        # Simulate ITR annual income extraction
        annual_income = (declared_income or 0) * 12
        ocr_annual = annual_income * random.uniform(0.88, 1.12)
        extracted_fields = {
            "assessment_year": "2024-25",
            "ocr_annual_income": round(ocr_annual, 2),
            "declared_annual_income": round(annual_income, 2),
            "pan_linked": document_number.upper(),
        }
        variance = abs(ocr_annual - annual_income) / max(annual_income, 1)
        if variance > 0.15:
            flags.append(f"INCOME_VARIANCE: {variance*100:.1f}% deviation from declared income")
            confidence -= 0.35
    else:
        flags.append(f"UNKNOWN_DOCUMENT_TYPE: '{document_type}' is not supported")
        confidence = 0.0
        extracted_fields = {"error": "Unsupported document type"}

    is_valid = len(flags) == 0 and confidence > 0.5
    confidence = max(0.0, min(1.0, confidence))

    result = DocumentVerificationResult(
        document_type=document_type,
        is_valid=is_valid,
        confidence_score=round(confidence, 2),
        flags=flags,
        extracted_fields=extracted_fields,
    )
    return result.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: Debt-to-Income (DTI) Calculator
# Computes DTI ratio from declared income and existing liabilities; returns
# an eligibility verdict with maximum recommended loan amount.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def dti_calculator(
    monthly_income: float,
    monthly_rent: float = 0.0,
    monthly_emi: float = 0.0,
    credit_card_payments: float = 0.0,
    other_obligations: float = 0.0,
    requested_loan_amount: Optional[float] = None,
    loan_tenure_months: int = 60,
    interest_rate: float = 10.0,
) -> str:
    """
    Calculates the Debt-to-Income ratio and determines loan eligibility.
    
    Args:
        monthly_income: Applicant's gross monthly income.
        monthly_rent: Monthly rent or housing payment.
        monthly_emi: Total existing EMI payments across all loans.
        credit_card_payments: Monthly minimum credit card payments.
        other_obligations: Any other monthly financial obligations.
        requested_loan_amount: Amount of loan being requested (optional).
        loan_tenure_months: Loan tenure in months (default 60).
        interest_rate: Annual interest rate percentage (default 10%).
    
    Returns:
        JSON string with DTI ratio, eligibility, max loan, and breakdown.
    """
    # --- Step 1: Total existing monthly debt ---
    total_debt = monthly_rent + monthly_emi + credit_card_payments + other_obligations
    breakdown = [
        f"Monthly Income: ${monthly_income:,.2f}",
        f"Monthly Rent/Housing: ${monthly_rent:,.2f}",
        f"Existing EMIs: ${monthly_emi:,.2f}",
        f"Credit Card Payments: ${credit_card_payments:,.2f}",
        f"Other Obligations: ${other_obligations:,.2f}",
        f"Total Monthly Debt: ${total_debt:,.2f}",
    ]

    # --- Step 2: Calculate DTI ratio ---
    if monthly_income <= 0:
        return json.dumps({
            "error": "Monthly income must be greater than zero"
        })

    dti_ratio = (total_debt / monthly_income) * 100
    breakdown.append(f"DTI Ratio: {total_debt:,.2f} / {monthly_income:,.2f} = {dti_ratio:.2f}%")

    # --- Step 3: Determine eligibility ---
    if dti_ratio < 30:
        eligibility = "ELIGIBLE — Low risk. Comfortable debt level."
        max_dti_for_new_loan = 45  # Can go up to 45% DTI with new loan
    elif dti_ratio < 45:
        eligibility = "CONDITIONALLY ELIGIBLE — Medium risk. Limited additional borrowing capacity."
        max_dti_for_new_loan = 50  # Stretch to 50% maximum
    elif dti_ratio < 55:
        eligibility = "HIGH RISK — Debt levels are elevated. Manual review required."
        max_dti_for_new_loan = 55
    else:
        eligibility = "NOT ELIGIBLE — DTI ratio exceeds acceptable limits."
        max_dti_for_new_loan = dti_ratio  # Already over limit

    # --- Step 4: Compute maximum recommended loan amount ---
    # Available monthly capacity for new EMI
    available_monthly = (monthly_income * (max_dti_for_new_loan / 100)) - total_debt
    available_monthly = max(available_monthly, 0)

    # EMI formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    monthly_rate = (interest_rate / 100) / 12
    if monthly_rate > 0 and available_monthly > 0:
        factor = ((1 + monthly_rate) ** loan_tenure_months)
        max_loan = available_monthly * (factor - 1) / (monthly_rate * factor)
    else:
        max_loan = 0

    breakdown.append(f"Available monthly capacity for new EMI: ${available_monthly:,.2f}")
    breakdown.append(
        f"Max recommended loan ({loan_tenure_months}mo @ {interest_rate}%): ${max_loan:,.0f}"
    )

    # --- Step 5: Check requested amount if provided ---
    if requested_loan_amount is not None:
        if requested_loan_amount <= max_loan:
            breakdown.append(
                f"✅ Requested ${requested_loan_amount:,.0f} is within limit"
            )
        else:
            breakdown.append(
                f"⚠ Requested ${requested_loan_amount:,.0f} EXCEEDS recommended max of ${max_loan:,.0f}"
            )
            eligibility += f" Requested amount exceeds capacity by ${requested_loan_amount - max_loan:,.0f}."

    result = DTIResult(
        monthly_income=monthly_income,
        total_monthly_debt=total_debt,
        dti_ratio=round(dti_ratio, 2),
        eligibility=eligibility,
        max_recommended_loan=round(max_loan, 2),
        breakdown=breakdown,
    )
    return result.model_dump_json()

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 (WEEK 3): Knowledge Retrieval
# Connects the agent to the RAG pipeline to answer policy/FAQ questions based
# on uploaded documents.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def knowledge_retrieval(query: str) -> str:
    """
    Search the internal banking knowledge base for policies, guidelines, definitions, and rules.
    Use this tool when the user asks general questions about loan rules, maximum limits,
    required documents, or definitions that are not specific to a single applicant's numbers.

    Args:
        query: The search query to look up in the knowledge base.
    
    Returns:
        A formatted string of relevant document snippets with metadata for citation.
    """
    app_logger.info(f"[Tool: knowledge_retrieval] Searching for: {query}")
    try:
        # Retrieve the top 4 chunks using the hybrid + cross-encoder pipeline
        results = retrieve_documents(query, top_k=4)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Format the results so the LLM can easily read and cite them
        formatted_docs = []
        for i, res in enumerate(results):
            source_id = res["metadata"].get("source_id", f"Doc_{i+1}")
            page = res["metadata"].get("page", "N/A")
            score = res.get("score", 0.0)
            
            # The structure "[CITABLE SOURCE: ...]" helps the LLM recognize where the data came from
            doc_str = (
                f"[CITABLE SOURCE: {source_id} | Page: {page} | Score: {score:.2f}]\n"
                f"{res['content']}\n"
            )
            formatted_docs.append(doc_str)
            
        return "\n".join(formatted_docs)
        
    except Exception as e:
        app_logger.error(f"Knowledge retrieval tool failed: {e}")
        return f"Error accessing knowledge base: {str(e)}"

# --- Export all tools as a list for the agent ---
# Add the new RAG tool to the list of available tools
ALL_TOOLS = [credit_score_analyzer, document_verification_engine, dti_calculator, knowledge_retrieval]
TOOL_NAMES = [t.name for t in ALL_TOOLS]