"""
server.py
---------
An MCP server exposing three document-intelligence tools, extending the
retrieval/Text-to-SQL work from ContextIQ into a proper tool-calling surface
that any MCP client (an agent loop, Claude Code, Claude Desktop, etc.) can
discover and call.

Tools exposed:
    1. semantic_search  - hybrid TF-IDF + embedding-style search over a small
                           in-memory document store (stands in for the
                           FAISS/BM25 hybrid retrieval used in ContextIQ).
    2. sql_query         - safe, read-only natural-language-to-SQL over a
                           sample SQLite table (mirrors ContextIQ's agentic
                           Text-to-SQL routing).
    3. classify_document - routes an incoming document to either the
                           semantic_search index or the sql_query engine
                           based on its content/type, the same auto-routing
                           idea used in ContextIQ's OCR/text-PDF split.

The server communicates over stdio using JSON-RPC 2.0 messages, one per
line (newline-delimited JSON), which is the same transport MCP uses for
local ("stdio") servers.
"""

from __future__ import annotations
import sys
import sqlite3
import os
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.protocol import (
    ToolSpec,
    MCPError,
    make_response,
    make_error,
    dumps,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from mcp_server.retrieval_tool import HybridRetriever
from mcp_server.sql_tool import SafeSQLEngine


class DocumentIntelligenceMCPServer:
    """Registers tools and dispatches JSON-RPC requests to them."""

    def __init__(self, docs_path: str, sqlite_path: str):
        self.retriever = HybridRetriever.from_jsonl(docs_path)
        self.sql_engine = SafeSQLEngine(sqlite_path)
        self.tools: Dict[str, ToolSpec] = {}
        self._register_tools()

    # ---- Tool registration -------------------------------------------------

    def _register_tools(self) -> None:
        self.tools["semantic_search"] = ToolSpec(
            name="semantic_search",
            description=(
                "Search unstructured document text using hybrid lexical + "
                "semantic scoring. Use for conceptual / descriptive questions "
                "about document content (policies, narrative text, explanations)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
            handler=self._handle_semantic_search,
        )

        self.tools["sql_query"] = ToolSpec(
            name="sql_query",
            description=(
                "Run a read-only SQL query against structured tabular data "
                "(e.g. financial figures, counts, aggregates). Use for exact "
                "numerical questions -- totals, averages, comparisons, filters."
            ),
            input_schema={
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            handler=self._handle_sql_query,
        )

        self.tools["classify_document"] = ToolSpec(
            name="classify_document",
            description=(
                "Classify whether a user question should be answered via "
                "semantic_search (unstructured text) or sql_query (structured "
                "data), mirroring the auto-routing used for OCR vs. text-PDF "
                "and CSV/Excel vs. document ingestion."
            ),
            input_schema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
            handler=self._handle_classify,
        )

    # ---- Tool handlers ------------------------------------------------------

    def _handle_semantic_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not query:
            raise MCPError(INVALID_PARAMS, "`query` is required")
        top_k = int(args.get("top_k", 3))
        hits = self.retriever.search(query, top_k=top_k)
        return {
            "results": [
                {"doc_id": h.doc_id, "score": round(h.score, 4), "text": h.text}
                for h in hits
            ]
        }

    def _handle_sql_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sql = args.get("sql")
        if not sql:
            raise MCPError(INVALID_PARAMS, "`sql` is required")
        try:
            rows, columns = self.sql_engine.run_read_only(sql)
        except PermissionError as e:
            raise MCPError(INVALID_PARAMS, str(e))
        except sqlite3.Error as e:
            raise MCPError(INTERNAL_ERROR, f"SQL error: {e}")
        return {"columns": columns, "rows": rows}

    def _handle_classify(self, args: Dict[str, Any]) -> Dict[str, Any]:
        question = args.get("question", "")
        if not question:
            raise MCPError(INVALID_PARAMS, "`question` is required")
        route = self.sql_engine.looks_structured(question)
        return {
            "route": "sql_query" if route else "semantic_search",
            "reason": (
                "Question references numeric/aggregate terms found in the "
                "structured schema."
                if route
                else "Question is descriptive/conceptual; routed to text retrieval."
            ),
        }

    # ---- JSON-RPC dispatch ---------------------------------------------------

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {}) or {}

        if method == "initialize":
            return make_response(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "doc-intelligence-mcp", "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                },
            )

        if method == "tools/list":
            return make_response(
                req_id, {"tools": [t.to_public_dict() for t in self.tools.values()]}
            )

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            tool = self.tools.get(name)
            if tool is None:
                return make_error(req_id, METHOD_NOT_FOUND, f"Unknown tool: {name}")
            try:
                result = tool.handler(arguments)
            except MCPError as e:
                return make_error(req_id, e.code, e.message)
            except Exception as e:  # defensive: never let a tool crash the server
                return make_error(req_id, INTERNAL_ERROR, f"Tool failed: {e}")
            return make_response(req_id, {"content": result})

        return make_error(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    # ---- stdio transport loop -------------------------------------------------

    def serve_forever(self) -> None:
        """Read newline-delimited JSON-RPC requests from stdin, write responses to stdout."""
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
            sys.stdout.write(dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(base, "..", "data", "sample_docs.jsonl")
    sqlite_path = os.path.join(base, "..", "data", "sample.db")
    server = DocumentIntelligenceMCPServer(docs_path, sqlite_path)
    server.serve_forever()
