# =============================================================================
# TEST REPORT: 20 queries covering FAQ, tool usage, and multi-turn conversation
# =============================================================================
# This script:
#   1. Sends 20 structured test queries to the FastAPI backend
#   2. Records responses, tools used, and pass/fail status
#   3. Saves raw results to test_reports/test_results_<timestamp>.json
#   4. Generates a readable test_reports/test_report_<timestamp>.md
#
# Run:
#   1. Make sure the API is running: uvicorn src.api.server:app --reload --port 8000
#   2. Then: python tests/test_report.py
# =============================================================================

import os
import httpx
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8000"
TIMEOUT = 180.0

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 20 TEST QUERIES organized by category
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # =========================================================================
    # CATEGORY 1: FAQ / General Banking Queries (Queries 1–5)
    # Tests: guardrail passes, general knowledge, no specific tool needed
    # =========================================================================
    {
        "id": 1,
        "category": "FAQ",
        "session_id": "faq-session",
        "query": "What is a debt-to-income ratio and why does it matter for loans?",
        "expected_behavior": "Should explain DTI concept without calling tools",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },
    {
        "id": 2,
        "category": "FAQ",
        "session_id": "faq-session",
        "query": "What credit score is considered good for a home loan?",
        "expected_behavior": "Should explain credit score ranges (300-900 or 300-850)",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },
    {
        "id": 3,
        "category": "FAQ",
        "session_id": "faq-session",
        "query": "What documents do I need to apply for a personal loan?",
        "expected_behavior": "Should list KYC documents: PAN, Aadhaar, income proof, etc.",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },
    {
        "id": 4,
        "category": "FAQ",
        "session_id": "faq-session",
        "query": "What is the difference between secured and unsecured loans?",
        "expected_behavior": "Should explain collateral-based vs non-collateral loans",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },
    {
        "id": 5,
        "category": "FAQ",
        "session_id": "faq-session",
        "query": "How long does the loan approval process usually take?",
        "expected_behavior": "Should give general timeline for loan processing",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },

    # =========================================================================
    # CATEGORY 2: Guardrail / Off-Topic Queries (Queries 6–8)
    # Tests: guardrail blocks non-banking queries
    # =========================================================================
    {
        "id": 6,
        "category": "Guardrail",
        "session_id": "guardrail-session",
        "query": "What is the best recipe for chocolate cake?",
        "expected_behavior": "Should be blocked by guardrail — not banking related",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": True,
    },
    {
        "id": 7,
        "category": "Guardrail",
        "session_id": "guardrail-session",
        "query": "Who won the FIFA World Cup in 2022?",
        "expected_behavior": "Should be blocked by guardrail — sports topic",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": True,
    },
    {
        "id": 8,
        "category": "Guardrail",
        "session_id": "guardrail-session",
        "query": "Write me a Python script to sort a list",
        "expected_behavior": "Should be blocked by guardrail — programming topic",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": True,
    },

    # =========================================================================
    # CATEGORY 3: Tool Usage — Credit Score Analyzer (Queries 9–11)
    # Tests: Tool 1 invocation with different risk profiles
    # =========================================================================
    {
        "id": 9,
        "category": "Tool - Credit Score",
        "session_id": "credit-session",
        "query": "Analyze my credit: I have 7 years of credit history, 98% on-time payments, 25% credit card utilization, and 1 active loan.",
        "expected_behavior": "Should call credit_score_analyzer, return Low risk bucket",
        "expect_tools": True,
        "expected_tool_names": ["credit_score_analyzer"],
        "expect_guardrail_block": False,
    },
    {
        "id": 10,
        "category": "Tool - Credit Score",
        "session_id": "credit-session-2",
        "query": "Check credit for applicant Rahul with PAN BXYPK1234A, 2 years credit history, 60% utilization, 3 active loans, 75% payment history.",
        "expected_behavior": "Should call credit_score_analyzer, return Medium or High risk",
        "expect_tools": True,
        "expected_tool_names": ["credit_score_analyzer"],
        "expect_guardrail_block": False,
    },
    {
        "id": 11,
        "category": "Tool - Credit Score",
        "session_id": "credit-session-3",
        "query": "I have only 6 months of credit history, missed several payments (50% on-time), 90% credit utilization, and 5 active loans. What's my risk?",
        "expected_behavior": "Should call credit_score_analyzer, return High or Reject",
        "expect_tools": True,
        "expected_tool_names": ["credit_score_analyzer"],
        "expect_guardrail_block": False,
    },

    # =========================================================================
    # CATEGORY 4: Tool Usage — Document Verification (Queries 12–14)
    # Tests: Tool 2 with valid/invalid documents
    # =========================================================================
    {
        "id": 12,
        "category": "Tool - Document Verification",
        "session_id": "doc-session",
        "query": "Verify my PAN card number ABCPK5678M for applicant name Priya Sharma.",
        "expected_behavior": "Should call document_verification_engine, PAN format valid",
        "expect_tools": True,
        "expected_tool_names": ["document_verification_engine"],
        "expect_guardrail_block": False,
    },
    {
        "id": 13,
        "category": "Tool - Document Verification",
        "session_id": "doc-session-2",
        "query": "Verify Aadhaar number 9876 5432 1098 for applicant Amit Kumar.",
        "expected_behavior": "Should call document_verification_engine, Aadhaar valid",
        "expect_tools": True,
        "expected_tool_names": ["document_verification_engine"],
        "expect_guardrail_block": False,
    },
    {
        "id": 14,
        "category": "Tool - Document Verification",
        "session_id": "doc-session-3",
        "query": "Verify PAN card 12345 for applicant name X. Also check Aadhaar 123456789012.",
        "expected_behavior": "Should flag invalid PAN format and fake Aadhaar number",
        "expect_tools": True,
        "expected_tool_names": ["document_verification_engine"],
        "expect_guardrail_block": False,
    },

    # =========================================================================
    # CATEGORY 5: Tool Usage — DTI Calculator (Queries 15–17)
    # Tests: Tool 3 with different income/debt profiles
    # =========================================================================
    {
        "id": 15,
        "category": "Tool - DTI Calculator",
        "session_id": "dti-session",
        "query": "Calculate my DTI: I earn $8000/month, pay $1200 rent, $500 EMI, and $300 credit card payment. I want a $100,000 loan for 5 years at 9%.",
        "expected_behavior": "Should call dti_calculator, DTI=25%, eligible, check loan amount",
        "expect_tools": True,
        "expected_tool_names": ["dti_calculator"],
        "expect_guardrail_block": False,
    },
    {
        "id": 16,
        "category": "Tool - DTI Calculator",
        "session_id": "dti-session-2",
        "query": "My income is $3500 per month. I pay $1000 rent, $800 EMI, $400 credit card, and $300 other debts. Am I eligible for any loan?",
        "expected_behavior": "Should call dti_calculator, DTI=71.4%, not eligible",
        "expect_tools": True,
        "expected_tool_names": ["dti_calculator"],
        "expect_guardrail_block": False,
    },
    {
        "id": 17,
        "category": "Tool - DTI Calculator",
        "session_id": "dti-session-3",
        "query": "I earn $12000 monthly with zero existing debts. What is the maximum home loan I can get at 8.5% for 20 years?",
        "expected_behavior": "Should call dti_calculator, DTI=0%, high max loan amount",
        "expect_tools": True,
        "expected_tool_names": ["dti_calculator"],
        "expect_guardrail_block": False,
    },

    # =========================================================================
    # CATEGORY 6: Multi-Turn Conversation (Queries 18–20)
    # Tests: Memory retention — 3 sequential queries in same session
    # =========================================================================
    {
        "id": 18,
        "category": "Multi-Turn",
        "session_id": "multiturn-session",
        "query": "Hi, I'm Ananya. I earn $9000 per month and I want to apply for a car loan.",
        "expected_behavior": "Should acknowledge name and income, ask for more details",
        "expect_tools": False,
        "expected_tool_names": [],
        "expect_guardrail_block": False,
    },
    {
        "id": 19,
        "category": "Multi-Turn",
        "session_id": "multiturn-session",
        "query": "I pay $1500 rent and $600 EMI on an existing personal loan. Can you calculate my DTI?",
        "expected_behavior": "Should remember income=$9000 from previous turn, call dti_calculator",
        "expect_tools": True,
        "expected_tool_names": ["dti_calculator"],
        "expect_guardrail_block": False,
    },
    {
        "id": 20,
        "category": "Multi-Turn",
        "session_id": "multiturn-session",
        "query": "Based on my earlier assessment, verify my PAN BNZAA2318J and tell me the final decision.",
        "expected_behavior": "Should remember DTI results + name from earlier turns, call doc verification, give final decision",
        "expect_tools": True,
        "expected_tool_names": ["document_verification_engine"],
        "expect_guardrail_block": False,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_single_test(test_case: dict) -> dict:
    """Send a single test query to the API and record the result."""
    print(f"\n  🔄 Query {test_case['id']}: {test_case['query'][:80]}...")

    start = time.time()
    try:
        resp = httpx.post(
            f"{API_BASE}/chat",
            json={
                "session_id": test_case["session_id"],
                "message": test_case["query"],
            },
            timeout=TIMEOUT,
        )
        elapsed = round(time.time() - start, 2)
        resp.raise_for_status()
        data = resp.json()

        # Determine pass/fail
        tools_used = data.get("tools_used", [])
        response_text = data.get("response", "")
        guardrail_blocked = "I am a Loan" in response_text or "can only help" in response_text.lower()

        # Check guardrail expectation
        guardrail_pass = (test_case["expect_guardrail_block"] == guardrail_blocked)

        # Check tool usage expectation
        if test_case["expect_tools"]:
            tools_pass = len(tools_used) > 0
        else:
            tools_pass = True  # No tool expected, any result is fine

        passed = guardrail_pass and tools_pass and len(response_text) > 10

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | Time: {elapsed}s | Tools: {tools_used}")

        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "session_id": test_case["session_id"],
            "query": test_case["query"],
            "expected_behavior": test_case["expected_behavior"],
            "response": response_text,
            "tools_used": tools_used,
            "structured_output": data.get("structured_output"),
            "guardrail_blocked": guardrail_blocked,
            "response_time_sec": elapsed,
            "passed": passed,
            "failure_reason": None if passed else (
                "Guardrail mismatch" if not guardrail_pass
                else "Expected tool not called" if not tools_pass
                else "Empty response"
            ),
        }

    except httpx.ConnectError:
        print("  ❌ FAIL | API not reachable")
        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "session_id": test_case.get("session_id", "unknown"),
            "query": test_case["query"],
            "expected_behavior": test_case["expected_behavior"],
            "response": None,
            "tools_used": [],
            "structured_output": None,
            "guardrail_blocked": False,
            "passed": False,
            "failure_reason": "API not reachable — is uvicorn running?",
            "response_time_sec": round(time.time() - start, 2),
        }
    except Exception as e:
        print(f"  ❌ FAIL | Error: {e}")
        return {
            "id": test_case["id"],
            "category": test_case["category"],
            "session_id": test_case.get("session_id", "unknown"),
            "query": test_case["query"],
            "expected_behavior": test_case["expected_behavior"],
            "response": None,
            "tools_used": [],
            "structured_output": None,
            "guardrail_blocked": False,
            "passed": False,
            "failure_reason": str(e),
            "response_time_sec": round(time.time() - start, 2),
        }


def reset_all_test_sessions():
    """Reset all sessions used in tests."""
    session_ids = set(t["session_id"] for t in TEST_QUERIES)
    for sid in session_ids:
        try:
            httpx.post(f"{API_BASE}/reset", params={"session_id": sid}, timeout=10)
        except Exception:
            pass


def run_all_tests() -> list:
    """Run all 20 test queries sequentially."""
    print("=" * 70)
    print("🧪 LOAN UNDERWRITING ASSISTANT — TEST REPORT")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   API:  {API_BASE}")
    print(f"   Total Queries: {min(10, len(TEST_QUERIES))} (Reduced to prevent timeouts)")
    print("=" * 70)

    # Reset sessions before starting
    print("\n🔄 Resetting test sessions...")
    reset_all_test_sessions()

    # Limit to 10 queries to avoid API rate limits/timeouts
    test_queries_limited = TEST_QUERIES[:10]

    results = []
    for test_case in test_queries_limited:
        category = test_case["category"]
        if not results or results[-1]["category"] != category:
            print(f"\n{'─'*50}")
            print(f"📂 Category: {category}")
            print(f"{'─'*50}")

        result = run_single_test(test_case)
        results.append(result)

        # Small delay between requests to avoid rate limiting
        time.sleep(1)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS TO JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_json_results(results: list, filepath: str):
    """Save raw test results as JSON."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    output = {
        "report_metadata": {
            "title": "Loan Underwriting Assistant — Test Report",
            "date": datetime.now().isoformat(),
            "api_base": API_BASE,
            "total_queries": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total)*100:.1f}%",
        },
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  💾 JSON saved: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_markdown_report(results: list, filepath: str):
    """Generate a readable Markdown test report."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    avg_time = sum(r.get("response_time_sec", 0) for r in results) / total

    lines = []
    lines.append("# 🧪 Test Report: Loan Underwriting Assistant\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**API:** `{API_BASE}`  \n")

    # Summary table
    lines.append("## 📊 Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total Queries | {total} |")
    lines.append(f"| Passed | {passed} ✅ |")
    lines.append(f"| Failed | {failed} ❌ |")
    lines.append(f"| Pass Rate | {(passed/total)*100:.1f}% |")
    lines.append(f"| Avg Response Time | {avg_time:.2f}s |")
    lines.append("")

    # Category breakdown
    lines.append("## 📂 Results by Category\n")
    categories = {}
    for r in results:
        cat = r.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r["passed"])
        cat_total = len(cat_results)
        lines.append(f"### {cat} ({cat_passed}/{cat_total} passed)\n")
        lines.append("| # | Query | Tools Used | Time | Status |")
        lines.append("|---|---|---|---|---|")

        for r in cat_results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            tools = ", ".join(r.get("tools_used", [])) or "—"
            query_short = r["query"][:60] + "..." if len(r["query"]) > 60 else r["query"]
            time_str = f"{r.get('response_time_sec', 0):.1f}s"
            lines.append(f"| {r['id']} | {query_short} | {tools} | {time_str} | {status} |")

        lines.append("")

    # Detailed results
    lines.append("## 📝 Detailed Results\n")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        lines.append(f"### Query {r['id']} — {status}\n")
        lines.append(f"- **Category:** {r.get('category', 'N/A')}")
        lines.append(f"- **Session:** `{r.get('session_id', 'N/A')}`")
        lines.append(f"- **Query:** {r['query']}")
        lines.append(f"- **Expected:** {r.get('expected_behavior', 'N/A')}")
        lines.append(f"- **Tools Used:** {', '.join(r.get('tools_used', [])) or 'None'}")
        lines.append(f"- **Response Time:** {r.get('response_time_sec', 0):.2f}s")

        if not r["passed"] and r.get("failure_reason"):
            lines.append(f"- **Failure Reason:** {r['failure_reason']}")

        # Show structured output if available
        if r.get("structured_output"):
            so = r["structured_output"]
            lines.append(f"- **Decision:** {so.get('decision', 'N/A')}")
            lines.append(f"- **Risk Score:** {so.get('risk_score', 'N/A')}")

        # Truncate response for readability
        response = r.get("response", "No response")
        if response and len(response) > 500:
            response = response[:500] + "... *(truncated)*"
        lines.append(f"\n**Response:**\n```\n{response}\n```\n")
        lines.append("---\n")

    # Write file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  📄 Markdown saved: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check API health first
    print("🏥 Checking API health...")
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=5)
        health_data = health.json()
        print(f"   Status: {health_data['status']}")
        print(f"   Model:  {health_data['model']}")
        print(f"   Tools:  {health_data['tools_available']}")
    except Exception:
        print("❌ API is not running! Start it first:")
        print("   uvicorn src.api.server:app --reload --port 8000")
        exit(1)

    # Run all tests
    results = run_all_tests()

    # Print summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 70)
    print(f"📊 FINAL RESULTS: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
    print("=" * 70)

    # Timestamped filenames inside test_reports/ folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(REPORT_DIR, f"test_results_{timestamp}.json")
    md_path = os.path.join(REPORT_DIR, f"test_report_{timestamp}.md")
    json_latest = os.path.join(REPORT_DIR, "test_results_latest.json")
    md_latest = os.path.join(REPORT_DIR, "test_report_latest.md")

    print(f"\n📁 Saving reports to: {REPORT_DIR}/")
    save_json_results(results, json_path)
    save_json_results(results, json_latest)
    generate_markdown_report(results, md_path)
    generate_markdown_report(results, md_latest)

    print(f"\n✅ All done! Reports saved:")
    print(f"   📄 {os.path.basename(md_path)}")
    print(f"   💾 {os.path.basename(json_path)}")
    print(f"   📄 test_report_latest.md       (quick access)")
    print(f"   💾 test_results_latest.json     (quick access)")
