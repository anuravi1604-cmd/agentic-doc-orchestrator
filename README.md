# Agentic Document Orchestrator

A multi-agent system for answering questions over a mix of unstructured
documents and structured tabular data, built on a real **Model Context
Protocol (MCP) server** and a small team of cooperating agents that
discover and call tools only through the MCP protocol boundary.

This project extends the retrieval + Text-to-SQL routing ideas from
[ContextIQ](https://github.com/anuravi1604-cmd) into an explicit
tool-calling, multi-agent architecture: a **Router Agent** decides which
specialist should handle a question, hands off to a **Retrieval Agent** or
**SQL Agent**, and a **Synthesis Agent** produces the final answer — with
every agent talking to tools exclusively over MCP, not direct function
calls.

## Why MCP, hand-rolled

Rather than depending on the official `mcp` SDK, the protocol layer here
(`mcp_server/protocol.py`) is implemented directly against the JSON-RPC 2.0
spec that MCP is built on — newline-delimited JSON-RPC messages over
stdio, with `initialize`, `tools/list`, and `tools/call` methods. This
keeps the project dependency-free and fully runnable offline, while still
matching the real wire protocol a client like Claude Desktop or Claude Code
would use to talk to any MCP server.

## Architecture

```
main.py
  └── spawns mcp_server/server.py as a subprocess (MCP server, stdio transport)
  └── agents/mcp_client.py talks to it over JSON-RPC 2.0

agents/orchestrator.py
  ├── RouterAgent      -> calls `classify_document` tool
  ├── RetrievalAgent   -> calls `semantic_search` tool
  ├── SQLAgent         -> drafts SQL, calls `sql_query` tool
  └── SynthesisAgent   -> composes the final answer

mcp_server/
  ├── protocol.py       -> JSON-RPC 2.0 message types (spec-compliant)
  ├── server.py         -> tool registry + dispatch loop
  ├── retrieval_tool.py -> hybrid TF-IDF + char-ngram search (BM25/BGE-style fusion)
  └── sql_tool.py       -> safe, read-only SQL execution + routing heuristic

skills/document_qa_skill.md -> an agent-skill definition (markdown
                                instructions an agent loads to know how
                                and when to use these tools)

evaluation/eval_routing.py -> benchmarks the Router Agent's tool-selection
                              accuracy (8/8 on the included test set)
```

## Tools exposed over MCP

| Tool | Purpose |
|---|---|
| `classify_document` | Routes a question to `sql_query` or `semantic_search` based on whether it's structured/numeric or descriptive |
| `semantic_search` | Hybrid lexical (word TF-IDF) + sub-word (char n-gram) search, fused the same way ContextIQ blends BM25 + dense embeddings |
| `sql_query` | Executes a single, validated **read-only** SELECT against a sample SQLite table — rejects writes, DDL, and chained statements |

## Running it

```bash
pip install -r requirements.txt

# Ask a specific question
python main.py "What was the total revenue across all quarters?"
python main.py "What is the remote work policy?"

# Or run the built-in demo set
python main.py

# Run the routing accuracy benchmark
python -m evaluation.eval_routing
```

No API key is required — the SQL-drafting and answer-synthesis steps fall
back to deterministic logic when `ANTHROPIC_API_KEY` isn't set, so the full
pipeline runs end-to-end offline. Set `ANTHROPIC_API_KEY` (and
`pip install anthropic`) to have the SQLAgent and SynthesisAgent use Claude
for SQL drafting and answer generation instead.

## Sample output

```
Q: What was the total revenue across all quarters?

Agent trace:
  [RouterAgent] classify_document: routed to 'sql_query' -- Question references numeric/aggregate terms
  [SQLAgent] draft_sql: SELECT SUM(revenue) AS total_revenue FROM quarterly_revenue;
  [SQLAgent] sql_query: 1 row(s) returned
  [SynthesisAgent] synthesize: final answer composed

Route chosen: sql_query
Answer: Based on the structured data: total_revenue=20400000.0
```

## Safety

The SQL tool is defensively hardened against an LLM (or a malicious prompt)
attempting to escalate to a write: it rejects any statement that isn't a
single `SELECT`, blocks write/DDL keywords anywhere in the query string
(not just the first token), rejects chained statements, and opens the
SQLite connection in strict read-only mode as a second line of defense.
Verified with adversarial inputs (`DROP TABLE`, chained `SELECT; DELETE`)
during development — both correctly rejected with the underlying table
left untouched.

## What this demonstrates

- Real MCP protocol implementation (JSON-RPC 2.0 over stdio), not a toy simulation
- Multi-agent orchestration with clear role separation and hand-offs
- Tool-calling / agent-skill patterns (see `skills/document_qa_skill.md`)
- Extending existing hybrid-retrieval and Text-to-SQL work into an explicit
  agentic architecture
- Offline-first design with an optional real-LLM path for production use
