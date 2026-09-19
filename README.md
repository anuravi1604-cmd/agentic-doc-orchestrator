# Agentic Document Orchestrator

A multi-agent document intelligence system built on the **Model Context Protocol (MCP 2024-11-05)**, a decoupled team of agents, dynamic Text-to-SQL generation, hybrid retrieval with Reciprocal Rank Fusion, a REST API powered by **FastAPI**, and **Docker** containerization.

---

## Key Features

- 🤖 **Decoupled Multi-Agent Architecture**: Dedicated `RouterAgent`, `SQLAgent`, `RetrievalAgent`, and `SynthesisAgent` with explicit interfaces and execution tracing.
- 🔌 **Spec-Compliant MCP Server**: Hand-rolled JSON-RPC 2.0 stdio server strictly adhering to the MCP 2024-11-05 specification with standard content blocks (`{"content": [{"type": "text", ...}]}`) and initialization handshake.
- 📜 **Dynamic Agent Skills**: Agents dynamically load and parse operational guidelines, procedures, and safety rules from `skills/document_qa_skill.md`.
- ⚡ **Dynamic NL-to-SQL Generator**: Parses natural language questions into valid SQL queries (aggregations, regional/quarterly filters, grouping, and ordering) across an expanded 32-row database spanning 2025–2026.
- 🔍 **Hybrid Document Retrieval**: Combines word-level TF-IDF (with sublinear frequency saturation approximating BM25) and sub-word character n-grams (handling acronyms, codes like `Sev-1`, and suffixes) merged via **Reciprocal Rank Fusion (RRF)** over 25 policy and engineering runbook documents.
- 🚀 **FastAPI Service**: Exposes REST endpoints (`/ask`, `/tools`, `/health`, `/eval`) with interactive Swagger UI (`/docs`).
- 🐳 **Docker & Docker Compose**: Sandboxed containerization exposing port `8000` with non-root security and automated container health checks.
- 🛡️ **Defensive SQL Security**: Enforces single-statement `SELECT`, blocks write/DDL keywords, and connects via SQLite read-only URI mode (`?mode=ro`).
- 🌐 **Offline-First with Optional LLM**: Operates 100% offline with zero external API dependencies. When `ANTHROPIC_API_KEY` is provided, agents automatically leverage Claude (`claude-sonnet-4-6`) for advanced zero-shot routing and synthesis.

---

## Architecture

```
                       User / Client Request
                     (CLI or FastAPI /ask)
                               │
                               ▼
                   DocumentAgentOrchestrator
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
       SkillLoader                          RouterAgent
 (loads document_qa_skill.md)              (classifies intent)
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
                     SQLAgent                                        RetrievalAgent
            (dynamic SQL generator)                               (hybrid query planner)
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                                              MCPClient
                                     (JSON-RPC 2.0 over stdio)
                                                  │
══════════════════════════════════════════════════╪══════════════════════════════════════════
                                    MCP Protocol Boundary
══════════════════════════════════════════════════╪══════════════════════════════════════════
                                                  │
                                                  ▼
                                    DocumentIntelligenceMCPServer
                                   (mcp_server/server.py subprocess)
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         ▼                        ▼                        ▼
                classify_document             sql_query             semantic_search
               (intent classifier)       (read-only SQLite)      (hybrid TF-IDF + RRF)
                                                  │
══════════════════════════════════════════════════╪══════════════════════════════════════════
                                                  │
                                                  ▼
                                            SynthesisAgent
                                (grounded natural-language answer)
```

---

## Directory Structure

```
├── Dockerfile                  # Production container image with health check
├── docker-compose.yml          # Multi-container orchestration (ports 8000:8000)
├── requirements.txt            # Project dependencies (FastAPI, Uvicorn, Scikit-Learn)
├── main.py                     # CLI entrypoint for interactive questions & demo runs
├── api.py                      # FastAPI REST application exposing /ask, /tools, /health, /eval
├── agents/
│   ├── orchestrator.py         # Real Agent implementations & orchestrator coordinator
│   ├── mcp_client.py           # Spec-compliant MCP stdio client
│   ├── sql_generator.py        # Dynamic Natural Language-to-SQL compiler
│   └── skill_loader.py         # Markdown agent skill parser and prompt injector
├── mcp_server/
│   ├── protocol.py             # JSON-RPC 2.0 & MCP 2024-11-05 spec data types
│   ├── server.py               # MCP server dispatch loop and tool registry
│   ├── retrieval_tool.py       # Hybrid TF-IDF + char n-gram with Reciprocal Rank Fusion
│   └── sql_tool.py             # Safe read-only SQLite execution and routing engine
├── skills/
│   └── document_qa_skill.md    # Agent skill definition with frontmatter & guidelines
├── data/
│   ├── sample.db               # SQLite database with 32 records (2025-2026, 4 regions)
│   └── sample_docs.jsonl       # 25 policy & engineering runbook documents
├── evaluation/
│   └── eval_routing.py         # 24-case benchmark testing intent routing & tricky edge cases
└── tests/
    └── test_orchestrator.py    # Automated test suite (pytest) covering agents, tools & API
```

---

## Tools Exposed Over MCP

| Tool | Parameters | Description |
|---|---|---|
| `classify_document` | `question: string` | Classifies query intent between structured data (`sql_query`) and text docs (`semantic_search`). |
| `semantic_search` | `query: string`, `top_k: int` | Hybrid lexical and sub-word passage retrieval with Reciprocal Rank Fusion (RRF). |
| `sql_query` | `sql: string` | Validates and executes a single read-only `SELECT` query against `quarterly_revenue`. |

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/anuravi1604-cmd/agentic-doc-orchestrator.git
cd agentic-doc-orchestrator

# Install dependencies
pip install -r requirements.txt
```

### 2. CLI Execution

```bash
# Run the built-in demo questions
python3 main.py

# Query structured tabular data
python3 main.py "What was the total revenue in Q1 2026?"
python3 main.py "Which region had the highest revenue?"

# Query unstructured policy documents
python3 main.py "What is the remote work policy?"
python3 main.py "How many days can I work remotely?"
```

### 3. FastAPI REST Service

Start the REST API server:

```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check**: `curl http://localhost:8000/health`
- **List discovered MCP tools**: `curl http://localhost:8000/tools`

#### Example API Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue in Q1 2026?"}'
```

Response:
```json
{
  "question": "What was the total revenue in Q1 2026?",
  "answer": "According to quarterly financial records, the total revenue is 16,130,000.00.",
  "route": "sql_query",
  "trace": [
    {
      "agent": "RouterAgent",
      "action": "classify_intent",
      "detail": "Routed to 'sql_query' (Question targets tabular numerical data (revenue, headcount, quarters, regions, database records).)"
    },
    {
      "agent": "SQLAgent",
      "action": "generate_sql",
      "detail": "SELECT SUM(revenue) AS total_revenue FROM quarterly_revenue WHERE quarter = 'Q1-2026';"
    },
    {
      "agent": "SQLAgent",
      "action": "execute_query",
      "detail": "Executed query successfully. 1 row(s) returned."
    },
    {
      "agent": "SynthesisAgent",
      "action": "synthesize_response",
      "detail": "Composed grounded answer from verified tool evidence."
    }
  ]
}
```

---

## 🐳 Docker & Docker Compose

### Run via Docker Compose

```bash
# Build and run the FastAPI service on port 8000
docker compose up --build

# Run in background (detached mode)
docker compose up -d
```

### Run via Docker CLI

```bash
# Build image
docker build -t agentic-doc-orchestrator .

# Run container exposing port 8000
docker run -p 8000:8000 --rm agentic-doc-orchestrator
```

To pass an optional Anthropic API key to the container:
```bash
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="your-api-key" --rm agentic-doc-orchestrator
```

---

## Evaluation & Testing

Run the automated test suite covering MCP communication, SQL safety, skill loading, and API endpoints:

```bash
python3 -m pytest tests/test_orchestrator.py -v
```

Run the 24-case intent routing benchmark (24 test cases):

```bash
python3 -m evaluation.eval_routing
```

Or trigger the benchmark via the REST API:
```bash
curl http://localhost:8000/eval
```

---

## Security

The SQL tool uses defensive hardening against unsafe or hallucinated LLM-generated SQL:
1. Rejects any statement not beginning with `SELECT`.
2. Scans the full query string for prohibited DDL/DML keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`, `PRAGMA`).
3. Rejects statement chaining (`SELECT ...; DROP ...`).
4. Establishes the SQLite connection using URI read-only mode (`file:...sample.db?mode=ro`).

Verified against adversarial attacks (`DROP TABLE quarterly_revenue`, chained `SELECT; DELETE`)—all attempts are rejected with the underlying table unmodified.
