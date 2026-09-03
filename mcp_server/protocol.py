"""
protocol.py
-----------
Minimal, spec-faithful implementation of the Model Context Protocol (MCP)
message layer, built directly on JSON-RPC 2.0 (https://www.jsonrpc.org/specification),
which is what MCP uses under the hood.

We implement this by hand (instead of depending on the official `mcp` SDK)
so the project has zero external runtime dependencies and can run fully
offline -- but the wire format below matches the real spec:

    Request:  {"jsonrpc": "2.0", "id": <int>, "method": <str>, "params": {...}}
    Response: {"jsonrpc": "2.0", "id": <int>, "result": {...}}
    Error:    {"jsonrpc": "2.0", "id": <int>, "error": {"code": int, "message": str}}

Methods implemented by our server (mirrors the real MCP lifecycle):
    initialize      -> capability negotiation / handshake
    tools/list      -> returns the tool registry (name, description, JSON schema)
    tools/call      -> invokes a tool by name with arguments, returns a result

This is the same shape a client (an LLM-driven agent, or Claude Desktop /
Claude Code itself) would use to discover and call tools over MCP.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


JSONRPC_VERSION = "2.0"


class MCPError(Exception):
    """Raised by tool handlers; converted into a JSON-RPC error response."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# Standard JSON-RPC error codes we make use of.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@dataclass
class ToolSpec:
    """Describes one MCP tool: its name, purpose, and input schema."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]] = field(repr=False)

    def to_public_dict(self) -> Dict[str, Any]:
        """What we hand back over tools/list -- never expose the handler."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def make_response(request_id: Optional[int], result: Any) -> Dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def make_error(request_id: Optional[int], code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)
