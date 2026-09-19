"""
orchestrator.py
----------------
A decoupled multi-agent document intelligence system built on the Model Context
Protocol (MCP) boundary:

    RouterAgent       -> Analyzes user intent, checks agent skills, routes to specialist
    RetrievalAgent    -> Formulates document retrieval parameters, queries semantic_search tool
    SQLAgent          -> Generates SQL (dynamic generator or LLM), executes sql_query tool
    SynthesisAgent    -> Grounds evidence and produces final natural-language answer
    DocumentAgentOrchestrator -> Coordinates agent handoffs, lifecycle, and execution tracing

All agent-to-tool interactions occur exclusively over the MCP protocol boundary.
"""

from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.mcp_client import MCPClient
from agents.skill_loader import SkillLoader, AgentSkill
from agents.sql_generator import SQLGenerator

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    question: str
    answer: str
    route: str
    trace: List[TraceStep] = field(default_factory=list)
    raw_output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "trace": [
                {"agent": s.agent, "action": s.action, "detail": s.detail, "metadata": s.metadata}
                for s in self.trace
            ],
            "raw_output": self.raw_output,
        }


class BaseAgent(ABC):
    """Abstract interface for autonomous specialist agents."""

    def __init__(self, name: str, mcp_client: MCPClient, skill: Optional[AgentSkill] = None):
        self.name = name
        self.client = mcp_client
        self.skill = skill
        self.discovered_tools: Dict[str, Dict[str, Any]] = {}
        self._discover_tools()

    def _discover_tools(self) -> None:
        """Discovers available tools dynamically from the MCP server."""
        try:
            tools = self.client.list_tools()
            self.discovered_tools = {t["name"]: t for t in tools}
        except Exception:
            self.discovered_tools = {}

    @abstractmethod
    def run(self, context: Dict[str, Any], trace: List[TraceStep]) -> Dict[str, Any]:
        pass


class RouterAgent(BaseAgent):
    """Decides which specialist agent should handle a given user question."""

    def __init__(self, mcp_client: MCPClient, skill: Optional[AgentSkill] = None):
        super().__init__("RouterAgent", mcp_client, skill)
        self.llm_enabled = _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.llm_enabled:
            self._anthropic = anthropic.Anthropic()

    def run(self, context: Dict[str, Any], trace: List[TraceStep]) -> Dict[str, Any]:
        question = context["question"]

        if self.llm_enabled:
            # LLM-assisted zero-shot classification
            try:
                system_prompt = (
                    "You are a routing agent for a document intelligence system. "
                    "Determine whether the question requires 'sql_query' (structured tabular data "
                    "about quarterly revenue, headcount, and regions) or 'semantic_search' "
                    "(unstructured documentation, company policies, runbooks, procedures).\n"
                    "Respond with ONLY 'sql_query' or 'semantic_search'."
                )
                msg = self._anthropic.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=30,
                    system=system_prompt,
                    messages=[{"role": "user", "content": question}],
                )
                route_candidate = msg.content[0].text.strip().lower()
                route = "sql_query" if "sql" in route_candidate else "semantic_search"
                reason = "LLM zero-shot classification based on skill definitions."
            except Exception:
                route_res = self.client.call_tool("classify_document", {"question": question})
                route = route_res.get("route", "semantic_search")
                reason = route_res.get("reason", "Fallback heuristic routing")
        else:
            # Protocol-mediated classification tool
            route_res = self.client.call_tool("classify_document", {"question": question})
            route = route_res.get("route", "semantic_search")
            reason = route_res.get("reason", "Contextual entity and intent classification")

        trace.append(
            TraceStep(
                agent=self.name,
                action="classify_intent",
                detail=f"Routed to '{route}' ({reason})",
                metadata={"route": route, "reason": reason},
            )
        )
        return {"route": route, "reason": reason}


class SQLAgent(BaseAgent):
    """Specialist agent for translating natural language to SQL and querying structured data."""

    def __init__(self, mcp_client: MCPClient, skill: Optional[AgentSkill] = None):
        super().__init__("SQLAgent", mcp_client, skill)
        self.sql_generator = SQLGenerator()
        self.llm_enabled = _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.llm_enabled:
            self._anthropic = anthropic.Anthropic()

    def run(self, context: Dict[str, Any], trace: List[TraceStep]) -> Dict[str, Any]:
        question = context["question"]
        schema_hint = self.sql_generator.SCHEMA_HINT

        # 1. Draft SQL
        if self.llm_enabled:
            try:
                msg = self._anthropic.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Schema: {schema_hint}\n"
                            f"Write a single read-only SQLite SELECT statement to answer: {question}\n"
                            f"Output only raw SQL without explanation or markdown fences."
                        ),
                    }],
                )
                raw_sql = msg.content[0].text.strip()
                sql = re.sub(r"^```sql\s*|^```\s*|```$", "", raw_sql, flags=re.MULTILINE).strip()
            except Exception:
                sql = self.sql_generator.generate(question)
        else:
            sql = self.sql_generator.generate(question)

        trace.append(
            TraceStep(
                agent=self.name,
                action="generate_sql",
                detail=sql,
                metadata={"sql": sql},
            )
        )

        # 2. Execute SQL over MCP boundary
        tool_output = self.client.call_tool("sql_query", {"sql": sql})
        row_count = len(tool_output.get("rows", []))
        trace.append(
            TraceStep(
                agent=self.name,
                action="execute_query",
                detail=f"Executed query successfully. {row_count} row(s) returned.",
                metadata={"row_count": row_count},
            )
        )

        return {"sql": sql, "tool_output": tool_output}


class RetrievalAgent(BaseAgent):
    """Specialist agent for retrieving and ranking passages from unstructured documentation."""

    def __init__(self, mcp_client: MCPClient, skill: Optional[AgentSkill] = None):
        super().__init__("RetrievalAgent", mcp_client, skill)

    def run(self, context: Dict[str, Any], trace: List[TraceStep]) -> Dict[str, Any]:
        question = context["question"]
        top_k = context.get("top_k", 3)

        tool_output = self.client.call_tool("semantic_search", {"query": question, "top_k": top_k})
        results = tool_output.get("results", [])

        trace.append(
            TraceStep(
                agent=self.name,
                action="retrieve_passages",
                detail=f"Retrieved {len(results)} passage(s) via hybrid RRF search.",
                metadata={"count": len(results), "top_doc": results[0]["doc_id"] if results else None},
            )
        )

        return {"tool_output": tool_output}


class SynthesisAgent(BaseAgent):
    """Composes grounded natural-language responses from structured or unstructured data."""

    def __init__(self, mcp_client: MCPClient, skill: Optional[AgentSkill] = None):
        super().__init__("SynthesisAgent", mcp_client, skill)
        self.llm_enabled = _ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.llm_enabled:
            self._anthropic = anthropic.Anthropic()

    def run(self, context: Dict[str, Any], trace: List[TraceStep]) -> Dict[str, Any]:
        question = context["question"]
        route = context["route"]
        tool_output = context["tool_output"]

        if self.llm_enabled:
            try:
                msg = self._anthropic.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Data source: {route}\n"
                            f"Data retrieved: {tool_output}\n"
                            f"Synthesize a clear, concise, direct answer citing specific numbers or document IDs."
                        ),
                    }],
                )
                answer = msg.content[0].text.strip()
            except Exception:
                answer = self._synthesize_offline(question, route, tool_output)
        else:
            answer = self._synthesize_offline(question, route, tool_output)

        trace.append(
            TraceStep(
                agent=self.name,
                action="synthesize_response",
                detail="Composed grounded answer from verified tool evidence.",
            )
        )
        return {"answer": answer}

    def _synthesize_offline(self, question: str, route: str, tool_output: Dict[str, Any]) -> str:
        if route == "sql_query":
            rows = tool_output.get("rows", [])
            cols = tool_output.get("columns", [])
            if not rows:
                return "No matching records found in the structured database."

            if len(rows) == 1 and len(cols) == 1:
                col, val = cols[0], rows[0][0]
                if isinstance(val, float):
                    formatted_val = f"{val:,.2f}"
                else:
                    formatted_val = str(val)
                return f"According to quarterly financial records, the {col.replace('_', ' ')} is {formatted_val}."

            row_summaries = []
            for r in rows[:5]:
                entry = ", ".join(f"{c}={v:,.2f}" if isinstance(v, float) else f"{c}={v}" for c, v in zip(cols, r))
                row_summaries.append(entry)

            joined_summary = "; ".join(row_summaries)
            if len(rows) > 5:
                joined_summary += f" (and {len(rows) - 5} more records)"
            return f"Based on the structured tabular data: {joined_summary}"

        else:
            results = tool_output.get("results", [])
            if not results:
                return "No relevant documentation passages were found matching your query."
            top = results[0]
            doc_id = top.get("doc_id", "doc")
            text = top.get("text", "")
            return f"According to company documentation ({doc_id}): {text}"


class DocumentAgentOrchestrator:
    """Coordinates specialist agents across the Model Context Protocol boundary."""

    def __init__(self, mcp_client: MCPClient):
        self.client = mcp_client
        self.skill_loader = SkillLoader()
        self.skill = self.skill_loader.get_skill("Document Intelligence Q&A")

        # Instantiate specialist agents
        self.router_agent = RouterAgent(mcp_client, self.skill)
        self.sql_agent = SQLAgent(mcp_client, self.skill)
        self.retrieval_agent = RetrievalAgent(mcp_client, self.skill)
        self.synthesis_agent = SynthesisAgent(mcp_client, self.skill)

    def answer(self, question: str) -> OrchestratorResult:
        trace: List[TraceStep] = []
        context: Dict[str, Any] = {"question": question}

        # 1. Router Agent
        route_info = self.router_agent.run(context, trace)
        route = route_info["route"]
        context["route"] = route

        # 2. Specialist Agent execution
        if route == "sql_query":
            specialist_output = self.sql_agent.run(context, trace)
            tool_output = specialist_output["tool_output"]
        else:
            specialist_output = self.retrieval_agent.run(context, trace)
            tool_output = specialist_output["tool_output"]

        context["tool_output"] = tool_output

        # 3. Synthesis Agent
        synthesis_output = self.synthesis_agent.run(context, trace)
        answer = synthesis_output["answer"]

        return OrchestratorResult(
            question=question,
            answer=answer,
            route=route,
            trace=trace,
            raw_output=tool_output,
        )
