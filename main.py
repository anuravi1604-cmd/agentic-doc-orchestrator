"""
main.py
-------
CLI entrypoint. Spawns the MCP server as a subprocess, connects an MCP
client to it, and runs a question through the multi-agent orchestrator.

Usage:
    python main.py "What was the total revenue across all quarters?"
    python main.py "What is the remote work policy?"
    python main.py                # runs a small built-in demo set
"""

from __future__ import annotations
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.mcp_client import MCPClient
from agents.orchestrator import DocumentAgentOrchestrator

DEMO_QUESTIONS = [
    "What was the total revenue across all quarters?",
    "What is the remote work policy?",
    "Which region had the highest revenue?",
    "How long must customer records be retained?",
]


def run(question: str) -> None:
    server_cmd = [sys.executable, "-m", "mcp_server.server"]
    with MCPClient(server_cmd) as client:
        tools = client.list_tools()
        print(f"[MCP] Connected. {len(tools)} tool(s) available: "
              f"{', '.join(t['name'] for t in tools)}\n")

        orchestrator = DocumentAgentOrchestrator(client)
        result = orchestrator.answer(question)

        print(f"Q: {question}")
        print("\nAgent trace:")
        for step in result.trace:
            print(f"  [{step.agent}] {step.action}: {step.detail}")
        print(f"\nRoute chosen: {result.route}")
        print(f"Answer: {result.answer}\n")
        print("-" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        for q in DEMO_QUESTIONS:
            run(q)
