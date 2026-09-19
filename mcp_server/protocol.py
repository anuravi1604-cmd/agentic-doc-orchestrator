"""
protocol.py
-----------
Spec-compliant implementation of the Model Context Protocol (MCP) message layer
(2024-11-05 specification), built directly on JSON-RPC 2.0.

Wire format:
    Request:      {"jsonrpc": "2.0", "id": <int>, "method": <str>, "params": {...}}
    Response:     {"jsonrpc": "2.0", "id": <int>, "result": {...}}
    Notification: {"jsonrpc": "2.0", "method": <str>, "params": {...}}
    Error:        {"jsonrpc": "2.0", "id": <int>, "error": {"code": int, "message": str}}

Tools/Call Response according to MCP specification:
    {
      "content": [
        {
          "type": "text",
          "text": "<json or formatted string>"
        }
      ],
      "isError": false
    }
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    """Raised by tool handlers; converted into a JSON-RPC error response."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# Standard JSON-RPC error codes
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
    handler: Callable[[Dict[str, Any]], Any] = field(repr=False)

    def to_public_dict(self) -> Dict[str, Any]:
        """Exposed over tools/list according to MCP schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


def format_tool_content(data: Any, is_error: bool = False) -> Dict[str, Any]:
    """Wraps output in standard MCP tool call response format."""
    if isinstance(data, str):
        text_content = data
    else:
        text_content = json.dumps(data, ensure_ascii=False)

    return {
        "content": [
            {
                "type": "text",
                "text": text_content,
            }
        ],
        "isError": is_error,
    }


def make_response(request_id: Optional[Union[int, str]], result: Any) -> Dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def make_error(request_id: Optional[Union[int, str]], code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)
