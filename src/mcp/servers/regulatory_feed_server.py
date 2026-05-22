from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta

# Create an MCP server for regulatory updates
mcp = FastMCP("Regulatory Feed Server")

@mcp.tool()
def get_latest_regulations(category: str) -> dict:
    """
    Fetch the latest regulatory updates and policy changes from the central bank.
    
    Args:
        category: The category of regulations to fetch (e.g., 'interest_rates', 'commercial_real_estate', 'compliance').
    """
    cat_lower = category.lower()
    
    # Mock data lookup based on category
    if "interest" in cat_lower or "rate" in cat_lower:
        return {
            "category": "Interest Rates",
            "latest_circular": "RBI/2026-27/42",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "The central bank has increased the repo rate by 25 basis points to curb inflation. All floating rate commercial loans must be indexed to the new benchmark effective immediately.",
            "impact_level": "High"
        }
    elif "commercial" in cat_lower or "real_estate" in cat_lower or "cre" in cat_lower:
        return {
            "category": "Commercial Real Estate (CRE)",
            "latest_circular": "RBI/2026-27/15",
            "date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
            "summary": "Maximum Loan-to-Value (LTV) ratio for Tier 1 city commercial real estate has been capped at 65% to reduce systemic risk exposure.",
            "impact_level": "Critical"
        }
    else:
        # Default mock response
        return {
            "category": "General Compliance",
            "latest_circular": "RBI/2026-27/01",
            "date": (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d"),
            "summary": "Standard KYC and AML guidelines remain unchanged. Ensure Udyam registration is verified for all MSME entities.",
            "impact_level": "Low"
        }

if __name__ == "__main__":
    mcp.run()
