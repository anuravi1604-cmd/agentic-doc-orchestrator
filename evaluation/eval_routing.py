"""
eval_routing.py
----------------
A small benchmark for the RouterAgent's tool-selection accuracy, in the same
spirit as the DeepEval-based benchmark used in ContextIQ -- except here the
metric of interest is "did the agent pick the right tool", which is the
crux of the agent-orchestration requirement, not just answer quality.

Run:
    python -m evaluation.eval_routing
"""

from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcp_client import MCPClient

# (question, expected_route)
BENCHMARK = [
    ("What was the total revenue across all quarters?", "sql_query"),
    ("What is the remote work policy?", "semantic_search"),
    ("Which region had the highest revenue?", "sql_query"),
    ("How long must customer records be retained?", "semantic_search"),
    ("What is the average headcount per quarter?", "sql_query"),
    ("What does the incident response runbook require for Sev-1 outages?", "semantic_search"),
    ("Compare NA and EU revenue.", "sql_query"),
    ("What is the expense reimbursement policy?", "semantic_search"),
]


def main() -> None:
    server_cmd = [sys.executable, "-m", "mcp_server.server"]
    correct = 0
    with MCPClient(server_cmd) as client:
        for question, expected in BENCHMARK:
            result = client.call_tool("classify_document", {"question": question})
            route = result["route"]
            is_correct = route == expected
            correct += int(is_correct)
            status = "PASS" if is_correct else "FAIL"
            print(f"[{status}] '{question}' -> got '{route}', expected '{expected}'")

    accuracy = correct / len(BENCHMARK)
    print(f"\nRouting accuracy: {correct}/{len(BENCHMARK)} ({accuracy:.1%})")


if __name__ == "__main__":
    main()
