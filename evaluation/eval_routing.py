"""
eval_routing.py
----------------
Benchmark suite evaluating the RouterAgent's intent classification and tool selection
accuracy across 24 realistic, diverse enterprise queries, including adversarial
counterexamples (e.g. document questions with numeric phrases like 'how many days'
or 'compare policies').

Run:
    python -m evaluation.eval_routing
"""

from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcp_client import MCPClient

# (question, expected_route, category_description)
BENCHMARK = [
    # Structured SQL queries (aggregates, metrics, filters, groupings)
    ("What was the total revenue across all quarters?", "sql_query"),
    ("What was the total revenue in Q1?", "sql_query"),
    ("Which region had the highest revenue?", "sql_query"),
    ("What is the average headcount per quarter?", "sql_query"),
    ("Compare NA and EU revenue.", "sql_query"),
    ("What did the company make in NA during Q2?", "sql_query"),
    ("What was our headcount in APAC in 2026?", "sql_query"),
    ("Which quarter had the lowest revenue in EU?", "sql_query"),
    ("How many quarters are recorded in the database?", "sql_query"),
    ("Total earnings for LATAM in 2025?", "sql_query"),
    ("Average revenue across all regions", "sql_query"),
    ("Breakdown of headcount by region", "sql_query"),

    # Unstructured Document queries (policies, procedures, runbooks, including tricky edge cases)
    ("What is the remote work policy?", "semantic_search"),
    ("How many days can I work remotely?", "semantic_search"),  # Tricky: contains "how many"
    ("How long must customer records be retained?", "semantic_search"),
    ("What does the incident response runbook require for Sev-1 outages?", "semantic_search"),
    ("What is the expense reimbursement policy?", "semantic_search"),
    ("Can we compare our security policies with best practices?", "semantic_search"),  # Tricky: contains "compare"
    ("What are the rules for code review SLA?", "semantic_search"),
    ("What is the equipment stipend amount for home offices?", "semantic_search"),
    ("What benefits are provided for parental leave?", "semantic_search"),
    ("What is the disaster recovery RTO and RPO requirement?", "semantic_search"),
    ("How does the employee referral bonus program work?", "semantic_search"),
    ("What are the engineering onboarding guidelines?", "semantic_search"),
]


def main() -> None:
    server_cmd = [sys.executable, "-m", "mcp_server.server"]
    correct = 0
    total = len(BENCHMARK)

    print("=" * 80)
    print("ROUTING EVALUATION BENCHMARK (24 Test Cases)")
    print("=" * 80)

    with MCPClient(server_cmd) as client:
        for i, (question, expected) in enumerate(BENCHMARK, 1):
            result = client.call_tool("classify_document", {"question": question})
            route = result.get("route", "")
            is_correct = route == expected
            correct += int(is_correct)
            status = "PASS" if is_correct else "FAIL"
            reason = result.get("reason", "")
            print(f"[{status}] #{i:02d}: '{question}'")
            print(f"       -> Got: '{route}', Expected: '{expected}' | Reason: {reason}")

    accuracy = (correct / total) * 100
    print("=" * 80)
    print(f"Benchmark Results: {correct}/{total} passed ({accuracy:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
