from mcp.server.fastmcp import FastMCP

# Create an MCP server for credit bureau tools
mcp = FastMCP("Credit Bureau Server")

@mcp.tool()
def fetch_credit_score(pan: str) -> dict:
    """
    Fetch the credit score and risk profile for a given PAN from the mock credit bureau.
    
    Args:
        pan: The user's Permanent Account Number (PAN)
    """
    # Mock data lookup based on PAN patterns
    if pan.startswith("A"):
        return {"score": 810, "profile": "Excellent", "history_length": "10 years", "active_loans": 1}
    elif pan.startswith("B"):
        return {"score": 620, "profile": "Fair", "history_length": "2 years", "active_loans": 3}
    elif pan.startswith("C"):
        return {"score": 450, "profile": "Poor", "history_length": "5 years", "active_loans": 5}
    else:
        # Default mock response
        return {"score": 720, "profile": "Good", "history_length": "4 years", "active_loans": 0}

@mcp.tool()
def check_defaulter_list(pan: str) -> dict:
    """
    Check if the PAN is listed in the national defaulter registry.
    
    Args:
        pan: The user's PAN
    """
    is_defaulter = pan.startswith("C")
    return {
        "pan": pan,
        "is_defaulter": is_defaulter,
        "last_checked": "2026-05-21T00:00:00Z"
    }

if __name__ == "__main__":
    mcp.run()
