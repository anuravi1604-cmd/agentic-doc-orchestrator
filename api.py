"""
api.py
------
Production FastAPI service exposing the Agentic Document Orchestrator:
- POST /ask     : Run a question through the multi-agent orchestrator
- GET  /tools   : List all tools registered on the MCP server
- GET  /health  : Health check and MCP server status
- GET  /eval    : Run the routing accuracy benchmark and return metrics
- GET  /docs    : Interactive OpenAPI / Swagger UI
"""

from __future__ import annotations
import sys
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.mcp_client import MCPClient
from agents.orchestrator import DocumentAgentOrchestrator
from evaluation.eval_routing import BENCHMARK


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language question to ask the multi-agent orchestrator",
        json_schema_extra={"example": "What was the total revenue in Q1 2026?"},
    )
    top_k: Optional[int] = Field(
        default=3,
        description="Number of passages to retrieve for document queries",
        ge=1,
        le=10,
    )


class TraceStepResponse(BaseModel):
    agent: str
    action: str
    detail: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    trace: List[TraceStepResponse]
    raw_output: Dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    mcp_server: str
    tools_available: int
    tools: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: spawn MCP server as subprocess and initialize orchestrator
    server_cmd = [sys.executable, "-m", "mcp_server.server"]
    client = MCPClient(server_cmd)
    orchestrator = DocumentAgentOrchestrator(client)

    app.state.client = client
    app.state.orchestrator = orchestrator

    yield

    # Shutdown: gracefully close the MCP client and terminate server process
    client.close()


app = FastAPI(
    title="Agentic Document Orchestrator API",
    description="Multi-agent document intelligence system with MCP tool boundary, dynamic SQL parsing, and hybrid retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    client: MCPClient = app.state.client
    tools = client.list_tools()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        mcp_server="doc-intelligence-mcp (JSON-RPC 2.0 stdio)",
        tools_available=len(tools),
        tools=[t["name"] for t in tools],
    )


@app.get("/tools", response_model=List[ToolResponse], tags=["MCP Tools"])
def list_tools():
    """Lists all tools discovered dynamically from the MCP server."""
    client: MCPClient = app.state.client
    return client.list_tools()


@app.post("/ask", response_model=QueryResponse, tags=["Orchestrator"])
def ask_question(request: QueryRequest):
    """
    Submits a question to the multi-agent orchestrator:
    1. RouterAgent classifies intent
    2. SQLAgent or RetrievalAgent executes specialist tool over MCP
    3. SynthesisAgent synthesizes final answer grounded in tool evidence
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    orchestrator: DocumentAgentOrchestrator = app.state.orchestrator
    try:
        result = orchestrator.answer(request.question)
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            route=result.route,
            trace=[
                TraceStepResponse(agent=s.agent, action=s.action, detail=s.detail, metadata=s.metadata)
                for s in result.trace
            ],
            raw_output=result.raw_output,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failure: {str(e)}")


@app.get("/eval", tags=["Evaluation"])
def run_evaluation():
    """Runs the benchmark evaluation across the dataset and returns accuracy metrics."""
    client: MCPClient = app.state.client
    correct = 0
    details = []

    for question, expected in BENCHMARK:
        res = client.call_tool("classify_document", {"question": question})
        got = res.get("route", "")
        is_correct = got == expected
        correct += int(is_correct)
        details.append({
            "question": question,
            "expected": expected,
            "got": got,
            "passed": is_correct,
            "reason": res.get("reason", ""),
        })

    accuracy = correct / len(BENCHMARK) if BENCHMARK else 0.0
    return {
        "total_test_cases": len(BENCHMARK),
        "correct": correct,
        "accuracy_pct": round(accuracy * 100, 2),
        "results": details,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
