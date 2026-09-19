"""
mcp_client.py
-------------
A spec-compliant Model Context Protocol (MCP) client communicating over stdio
using newline-delimited JSON-RPC 2.0 messages matching the MCP 2024-11-05 spec.
"""

from __future__ import annotations
import json
import subprocess
import sys
import itertools
from typing import Any, Dict, List, Optional, Union


class MCPClient:
    """Client for discovering and invoking tools over the MCP stdio protocol."""

    def __init__(self, server_cmd: List[str]):
        self._proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._id_counter = itertools.count(1)
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self._initialize()

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        req_id = next(self._id_counter)
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr_output = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"MCP server closed connection unexpectedly.\n{stderr_output}")

        try:
            response = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Malformed JSON from MCP server: {line}\nError: {e}")

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP Error ({err.get('code')}): {err.get('message')}")

        return response.get("result")

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Sends a JSON-RPC notification (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(notification) + "\n")
        self._proc.stdin.flush()

    def _initialize(self) -> None:
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "agentic-doc-orchestrator", "version": "1.0.0"},
                "capabilities": {},
            },
        )
        if isinstance(result, dict):
            self.server_info = result.get("serverInfo", {})
            self.capabilities = result.get("capabilities", {})

        # Send post-initialization notification per MCP specification
        self._send_notification("notifications/initialized")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns all tool specifications published by the MCP server."""
        result = self._send_request("tools/list")
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Invokes an MCP tool by name and parses the spec-compliant content block.
        MCP responses return: {'content': [{'type': 'text', 'text': '...'}], 'isError': bool}
        """
        result = self._send_request("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            return result

        content_blocks = result.get("content", [])
        is_error = result.get("isError", False)

        # Extract text from content block
        parsed_payload: Any = None
        if content_blocks and isinstance(content_blocks, list):
            first_block = content_blocks[0]
            if isinstance(first_block, dict) and "text" in first_block:
                raw_text = first_block["text"]
                try:
                    parsed_payload = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    parsed_payload = raw_text
            else:
                parsed_payload = first_block
        else:
            parsed_payload = result

        if is_error:
            error_msg = parsed_payload.get("error", str(parsed_payload)) if isinstance(parsed_payload, dict) else str(parsed_payload)
            raise RuntimeError(f"MCP Tool '{name}' returned error: {error_msg}")

        return parsed_payload

    def close(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
