from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Income Verification Server")

@mcp.tool()
def verify_employment_income(pan: str, employer_name: str) -> dict:
    """
    Verify employment status and monthly income from tax records.
    
    Args:
        pan: The user's Permanent Account Number (PAN)
        employer_name: The declared employer name
    """
    # Mock verification logic
    is_verified = not pan.startswith("X")
    base_income = 85000 if is_verified else 0
    
    # Adjust income based on PAN for testing different scenarios
    if pan.startswith("A"):
        base_income = 150000
    elif pan.startswith("B"):
        base_income = 45000
        
    return {
        "employer_match": is_verified,
        "verified_monthly_income_inr": base_income,
        "tax_filing_status": "Up to date" if is_verified else "Pending",
        "confidence_score": 0.95 if is_verified else 0.1
    }

if __name__ == "__main__":
    mcp.run()
