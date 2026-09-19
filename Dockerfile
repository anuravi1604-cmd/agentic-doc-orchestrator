# Production Dockerfile for Agentic Document Orchestrator (FastAPI + MCP Multi-Agent System)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Layer caching for pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Security best practice: create and run as non-root user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI REST service port
EXPOSE 8000

# Health check endpoint verification
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch FastAPI web application with Uvicorn by default
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
