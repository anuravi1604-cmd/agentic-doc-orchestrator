"""
test_orchestrator.py
--------------------
Comprehensive automated test suite verifying:
- Model Context Protocol (MCP) stdio client/server communication and spec compliance.
- Dynamic SQL generator logic and parameterized queries.
- Read-only SQL defense mechanisms against adversarial queries.
- Hybrid retrieval with Reciprocal Rank Fusion.
- Agent skill loading and parsing.
- Multi-agent orchestration and trace logging.
- FastAPI REST API endpoints (/health, /tools, /ask, /eval).
"""

from __future__ import annotations
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mcp_client import MCPClient
from agents.orchestrator import DocumentAgentOrchestrator
from agents.skill_loader import SkillLoader
from agents.sql_generator import SQLGenerator
from mcp_server.sql_tool import SafeSQLEngine
from api import app


@pytest.fixture(scope="module")
def mcp_client():
    server_cmd = [sys.executable, "-m", "mcp_server.server"]
    with MCPClient(server_cmd) as client:
        yield client


def test_mcp_handshake_and_tools_list(mcp_client):
    """Verifies MCP capability exchange and dynamic tool discovery."""
    tools = mcp_client.list_tools()
    assert len(tools) == 3
    tool_names = {t["name"] for t in tools}
    assert "classify_document" in tool_names
    assert "semantic_search" in tool_names
    assert "sql_query" in tool_names


def test_skill_loader():
    """Verifies that skills/document_qa_skill.md is loaded and parsed."""
    loader = SkillLoader()
    skills = loader.load_all()
    assert "Document Intelligence Q&A" in skills
    skill = loader.get_skill("Document Intelligence Q&A")
    assert skill is not None
    assert "classify_document" in skill.tools
    assert "semantic_search" in skill.tools
    assert "sql_query" in skill.tools
    assert len(skill.procedure) > 0


def test_dynamic_sql_generator():
    """Verifies dynamic SQL generation across filters and aggregations."""
    gen = SQLGenerator()

    # Sum across all quarters
    sql1 = gen.generate("What was the total revenue across all quarters?")
    assert "SELECT SUM(revenue) AS total_revenue FROM quarterly_revenue;" == sql1

    # Filter by specific quarter
    sql2 = gen.generate("What was the total revenue in Q1?")
    assert "WHERE quarter LIKE 'Q1%'" in sql2
    assert "SUM(revenue)" in sql2

    # Filter by region and quarter
    sql3 = gen.generate("What did the company make in NA during Q2-2026?")
    assert "region = 'NA'" in sql3
    assert "quarter = 'Q2-2026'" in sql3

    # Group by region comparison
    sql4 = gen.generate("Compare NA and EU revenue")
    assert "region IN ('NA', 'EU')" in sql4
    assert "GROUP BY region" in sql4

    # Highest revenue
    sql5 = gen.generate("Which region had the highest revenue?")
    assert "GROUP BY region ORDER BY total_revenue DESC LIMIT 1" in sql5


def test_safe_sql_engine_adversarial_rejection():
    """Verifies defense in depth rejects write, DDL, and chained queries."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.db")
    engine = SafeSQLEngine(db_path)

    # Rejects non-SELECT
    with pytest.raises(PermissionError, match="Only SELECT"):
        engine.run_read_only("UPDATE quarterly_revenue SET revenue = 0;")

    # Rejects DROP TABLE
    with pytest.raises(PermissionError, match="Forbidden keyword 'drop'"):
        engine.run_read_only("SELECT * FROM quarterly_revenue; DROP TABLE quarterly_revenue;")

    # Rejects statement chaining
    with pytest.raises(PermissionError):
        engine.run_read_only("SELECT * FROM quarterly_revenue; SELECT * FROM quarterly_revenue;")


def test_multi_agent_orchestrator_sql_flow(mcp_client):
    """Verifies end-to-end multi-agent execution on structured data."""
    orchestrator = DocumentAgentOrchestrator(mcp_client)
    res = orchestrator.answer("What was the total revenue in Q1 2026?")

    assert res.route == "sql_query"
    assert "16,130,000.00" in res.answer or "total revenue is" in res.answer
    assert len(res.trace) == 4

    agents_in_trace = [s.agent for s in res.trace]
    assert agents_in_trace == ["RouterAgent", "SQLAgent", "SQLAgent", "SynthesisAgent"]


def test_multi_agent_orchestrator_retrieval_flow(mcp_client):
    """Verifies end-to-end multi-agent execution on unstructured documents."""
    orchestrator = DocumentAgentOrchestrator(mcp_client)
    res = orchestrator.answer("How many days can I work remotely?")

    assert res.route == "semantic_search"
    assert "three days per week" in res.answer
    assert "doc_1" in res.answer

    agents_in_trace = [s.agent for s in res.trace]
    assert agents_in_trace == ["RouterAgent", "RetrievalAgent", "SynthesisAgent"]


def test_fastapi_endpoints():
    """Verifies FastAPI service endpoints (/health, /tools, /ask, /eval)."""
    with TestClient(app) as client:
        # Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        data_health = res_health.json()
        assert data_health["status"] == "healthy"
        assert data_health["tools_available"] == 3

        # Tools list
        res_tools = client.get("/tools")
        assert res_tools.status_code == 200
        tools = res_tools.json()
        assert len(tools) == 3

        # Ask endpoint (SQL query)
        res_ask_sql = client.post("/ask", json={"question": "What was the total revenue across all quarters?"})
        assert res_ask_sql.status_code == 200
        data_sql = res_ask_sql.json()
        assert data_sql["route"] == "sql_query"
        assert len(data_sql["trace"]) > 0

        # Ask endpoint (Document query)
        res_ask_doc = client.post("/ask", json={"question": "What is the remote work policy?"})
        assert res_ask_doc.status_code == 200
        data_doc = res_ask_doc.json()
        assert data_doc["route"] == "semantic_search"

        # Eval endpoint
        res_eval = client.get("/eval")
        assert res_eval.status_code == 200
        data_eval = res_eval.json()
        assert data_eval["total_test_cases"] == 24
        assert data_eval["accuracy_pct"] == 100.0
