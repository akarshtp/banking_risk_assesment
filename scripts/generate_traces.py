import httpx
import time
import os

API_URL = "http://localhost:8000/chat"

# Sample queries designed to hit MCP tools, trigger HITL rules, and test Role filtering.
TEST_QUERIES = [
    "What is the credit score of applicant John Doe?", # Should hit MCP credit bureau tool
    "Can you verify the income for Jane Smith from her recent tax filings?", # Should hit MCP income verification tool
    "I need a loan of ₹60,000,000 for a commercial property.", # Should trigger HITL (high_value_loan)
    "My credit score is 580, can I get a personal loan?", # Should trigger HITL (low_credit_score)
    "What are the internal guidelines for approving exceptions?", # Should hit Role-based RAG
] * 10 # Repeat 10 times to get 50 queries

def main():
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Warning: LANGCHAIN_API_KEY is not set. Traces will not be sent to LangSmith.")
        
    print(f"Generating traces for {len(TEST_QUERIES)} queries...")
    
    # Simulate queries with different roles
    roles = ["junior_analyst", "senior_underwriter", "credit_head"]
    
    for i, query in enumerate(TEST_QUERIES):
        role = roles[i % len(roles)]
        print(f"[{i+1}/{len(TEST_QUERIES)}] Role: {role} | Query: {query}")
        
        try:
            response = httpx.post(API_URL, json={
                "message": query,
                "session_id": f"trace_test_{i}",
                "role_name": role
            }, timeout=30.0)
            
            if response.status_code == 200:
                print("  -> Success")
            else:
                print(f"  -> Failed: {response.status_code}")
                
        except Exception as e:
            print(f"  -> Error: {e}")
            
        time.sleep(1) # Small delay to avoid overwhelming the API
        
    print("Trace generation complete!")

if __name__ == "__main__":
    main()
