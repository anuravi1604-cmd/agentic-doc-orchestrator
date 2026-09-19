"""
server.py
---------
MCP Server exposing document intelligence and SQL tools over JSON-RPC 2.0 stdio
transport according to the Model Context Protocol 2024-11-05 specification.
"""

from __future__ import annotations
import sys
import os
import sqlite3
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.protocol import (
    ToolSpec,
    MCPError,
    format_tool_content,
    make_response,
    make_error,
    dumps,
    PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from mcp_server.retrieval_tool import HybridRetriever
from mcp_server.sql_tool import SafeSQLEngine


class DocumentIntelligenceMCPServer:
    """Registers tools and dispatches JSON-RPC requests conforming to MCP spec."""

    def __init__(self, docs_path: str, sqlite_path: str):
        self.retriever = HybridRetriever.from_jsonl(docs_path)
        self.sql_engine = SafeSQLEngine(sqlite_path)
        self.tools: Dict[str, ToolSpec] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        self.tools["semantic_search"] = ToolSpec(
            name="semantic_search",
            description=(
                "Search unstructured document text using hybrid lexical (TF-IDF) + "
                "sub-word character n-gram scoring with Reciprocal Rank Fusion. "
                "Use for questions about policies, runbooks, engineering standards, "
                "and company documentation."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "default": 3, "description": "Number of passages to return"},
                },
                "required": ["query"],
            },
            handler=self._handle_semantic_search,
        )

        self.tools["sql_query"] = ToolSpec(
            name="sql_query",
            description=(
                "Run a validated read-only SQL query against structured tabular data "
                "(quarterly_revenue table with columns: quarter, region, revenue, headcount). "
                "Use for exact numerical questions, financial aggregations, and regional statistics."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single read-only SELECT statement"},
                },
                "required": ["sql"],
            },
            handler=self._handle_sql_query,
        )

        self.tools["classify_document"] = ToolSpec(
            name="classify_document",
            description=(
                "Classify whether a user question should be answered via semantic_search "
                "(unstructured documents) or sql_query (structured tabular data), using "
                "entity recognition and intent extraction."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User question to classify"},
                },
                "required": ["question"],
            },
            handler=self._handle_classify,
        )

    def _handle_semantic_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not query:
            raise MCPError(INVALID_PARAMS, "`query` parameter is required")
        top_k = int(args.get("top_k", 3))
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "results": [hit.to_dict() for hit in hits],
            "count": len(hits),
        }

    def _handle_sql_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = args.get("sql")
        if not sql:
            raise MCPError(INVALID_PARAMS, "`sql` parameter is required")
        try:
            rows, columns = self.sql_engine.run_read_only(sql)
        except PermissionError as e:
            raise MCPError(INVALID_PARAMS, str(e))
        except sqlite3.Error as e:
            raise MCPError(INTERNAL_ERROR, f"SQL error: {e}")
        return {"columns": columns, "rows": rows, "count": len(rows)}

    def _handle_classify(self, args: Dict[str, Any]) -> Dict[str, Any]:
        question = args.get("question", "")
        if not question:
            raise MCPError(INVALID_PARAMS, "`question` parameter is required")
        return self.sql_engine.classify_question(question)

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        # Notification handling (notifications do not have an id and expect no response)
        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return make_response(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {
                        "name": "doc-intelligence-mcp",
                        "version": "1.0.0",
                    },
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                },
            )

        if method == "tools/list":
            return make_response(
                req_id,
                {"tools": [t.to_public_dict() for t in self.tools.values()]},
            )

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            tool = self.tools.get(name)
            if tool is None:
                return make_error(req_id, METHOD_NOT_FOUND, f"Unknown tool: '{name}'")
            try:
                raw_result = tool.handler(arguments)
                # Format response as standard MCP content block
                tool_content = format_tool_content(raw_result, is_error=False)
                return make_response(req_id, tool_content)
            except MCPError as e:
                tool_content = format_tool_content({"error": e.message, "code": e.code}, is_error=True)
                return make_response(req_id, tool_content)
            except Exception as e:
                tool_content = format_tool_content({"error": str(e)}, is_error=True)
                return make_response(req_id, tool_content)

        return make_error(req_id, METHOD_NOT_FOUND, f"Unknown method: '{method}'")

    def serve_forever(self) -> None:
        """Reads newline-delimited JSON-RPC messages from stdin and writes responses to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = __import__("json").loads(line)
            except ValueError:
                sys.stdout.write(dumps(make_error(None, -32700, "Parse error")) + "\n")
                sys.stdout.flush()
                continue

            resp = self.handle_request(req)
            if resp is not None:
                sys.stdout.write(dumps(resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(base, "..", "data", "sample_docs.jsonl")
    sqlite_path = os.path.join(base, "..", "data", "sample.db")
    server = DocumentIntelligenceMCPServer(docs_path, sqlite_path)
    server.serve_forever()
