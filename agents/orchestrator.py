"""
orchestrator.py
----------------
A small multi-agent system built on top of the MCP client:

    RouterAgent       -> decides which specialist should handle the question
                         (calls the `classify_document` MCP tool)
    RetrievalAgent    -> handles descriptive/unstructured questions
                         (calls the `semantic_search` MCP tool)
    SQLAgent          -> handles numeric/structured questions
                         (calls the `sql_query` MCP tool, after asking an LLM
                          -- or a deterministic fallback -- to draft the SQL)
    SynthesisAgent    -> turns raw tool output into a final natural-language
                         answer

Each agent's only interface to the outside world is the MCPClient -- no
agent calls another agent's Python function directly, and no agent talks to
a tool implementation directly. This mirrors a real agent-orchestration
setup where agents are decoupled by the protocol boundary, so any agent
could be swapped for a different implementation (or a different LLM)
without touching the others.

If ANTHROPIC_API_KEY is set and the `anthropic` package is installed, the
SQLAgent and SynthesisAgent use Claude to draft SQL / write the final
answer. Otherwise both fall back to deterministic logic so the whole
pipeline still runs end-to-end offline, with no API key required, which is
useful for grading, demos, or CI.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.mcp_client import MCPClient

try:
    import anthropic  # type: ignore

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


@dataclass
class TraceStep:
    agent: str
    action: str
    detail: str


@dataclass
class OrchestratorResult:
    answer: str
    route: str
    trace: List[TraceStep] = field(default_factory=list)


class LLMBackend:
    """Wraps an optional real LLM call; falls back to deterministic logic."""

    def __init__(self):
        self.enabled = _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.enabled:
            self._client = anthropic.Anthropic()

    def draft_sql(self, question: str, schema_hint: str) -> str:
        if self.enabled:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Schema: {schema_hint}\nWrite a single read-only SQLite "
                        f"SELECT statement (no explanation, just SQL) to answer: "
                        f"{question}"
                    ),
                }],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            return _strip_sql_fences(text)
        return _fallback_draft_sql(question)

    def synthesize(self, question: str, tool_output: Dict[str, Any], route: str) -> str:
        if self.enabled:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {question}\nTool used: {route}\n"
                        f"Tool output: {tool_output}\n"
                        f"Write a concise, direct answer using only this data."
                    ),
                }],
            )
            return "".join(b.text for b in msg.content if b.type == "text")
        return _fallback_synthesize(question, tool_output, route)


def _strip_sql_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```sql\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    return text


def _fallback_draft_sql(question: str) -> str:
    """Deterministic SQL templates for the demo schema, used when no LLM is configured."""
    q = question.lower()
    if "total" in q and "revenue" in q:
        return "SELECT SUM(revenue) AS total_revenue FROM quarterly_revenue;"
    if "average" in q and "headcount" in q:
        return "SELECT AVG(headcount) AS avg_headcount FROM quarterly_revenue;"
    if "highest" in q or "top" in q:
        return "SELECT quarter, region, revenue FROM quarterly_revenue ORDER BY revenue DESC LIMIT 1;"
    if "region" in q:
        return "SELECT region, SUM(revenue) AS total_revenue FROM quarterly_revenue GROUP BY region;"
    return "SELECT * FROM quarterly_revenue;"


def _fallback_synthesize(question: str, tool_output: Dict[str, Any], route: str) -> str:
    if route == "sql_query":
        rows, cols = tool_output.get("rows", []), tool_output.get("columns", [])
        if not rows:
            return "No matching rows were found for that query."
        formatted = "; ".join(
            ", ".join(f"{c}={v}" for c, v in zip(cols, row)) for row in rows
        )
        return f"Based on the structured data: {formatted}"
    else:
        results = tool_output.get("results", [])
        if not results:
            return "No relevant document passages were found."
        top = results[0]
        return f"Based on the most relevant document ({top['doc_id']}): {top['text']}"


class DocumentAgentOrchestrator:
    """Coordinates RouterAgent -> {RetrievalAgent | SQLAgent} -> SynthesisAgent."""

    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
        self.llm = LLMBackend()

    def answer(self, question: str) -> OrchestratorResult:
        trace: List[TraceStep] = []

        # --- RouterAgent ---
        route_result = self.client.call_tool("classify_document", {"question": question})
        route = route_result["route"]
        trace.append(TraceStep(
            agent="RouterAgent", action="classify_document",
            detail=f"routed to '{route}' -- {route_result['reason']}",
        ))

        # --- Specialist agent hand-off ---
        if route == "sql_query":
            sql = self.llm.draft_sql(question, schema_hint="quarterly_revenue(quarter, region, revenue, headcount)")
            trace.append(TraceStep(agent="SQLAgent", action="draft_sql", detail=sql))
            tool_output = self.client.call_tool("sql_query", {"sql": sql})
            trace.append(TraceStep(
                agent="SQLAgent", action="sql_query",
                detail=f"{len(tool_output.get('rows', []))} row(s) returned",
            ))
        else:
            tool_output = self.client.call_tool("semantic_search", {"query": question, "top_k": 3})
            trace.append(TraceStep(
                agent="RetrievalAgent", action="semantic_search",
                detail=f"{len(tool_output.get('results', []))} passage(s) retrieved",
            ))

        # --- SynthesisAgent ---
        answer_text = self.llm.synthesize(question, tool_output, route)
        trace.append(TraceStep(agent="SynthesisAgent", action="synthesize", detail="final answer composed"))

        return OrchestratorResult(answer=answer_text, route=route, trace=trace)
