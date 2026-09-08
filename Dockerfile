# Production-grade Dockerfile for Agentic Document Orchestrator (MCP Multi-Agent System)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Layer caching for pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Security best practice: non-root user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["python3", "main.py"]
