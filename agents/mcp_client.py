"""
mcp_client.py
-------------
A minimal MCP client: spawns the MCP server as a child process and
communicates with it over stdio using newline-delimited JSON-RPC 2.0
messages -- the same transport real MCP clients (Claude Desktop, Claude
Code, etc.) use for local servers.

This is the piece any agent in this project uses to discover tools
(`list_tools`) and invoke them (`call_tool`), so the orchestration layer
never talks to the tool implementations directly -- only through the
protocol boundary, exactly as a production multi-agent system would.
"""

from __future__ import annotations
import json
import subprocess
import sys
import itertools
from typing import Any, Dict, List, Optional


class MCPClient:
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
        self._initialize()

    def _send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = next(self._id_counter)
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr_output = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"MCP server closed connection unexpectedly.\n{stderr_output}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"MCP error {response['error']['code']}: {response['error']['message']}")
        return response["result"]

    def _initialize(self) -> None:
        self._send("initialize", {"clientInfo": {"name": "agent-orchestrator", "version": "0.1.0"}})

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._send("tools/list")["tools"]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._send("tools/call", {"name": name, "arguments": arguments})["content"]

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait(timeout=5)

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
