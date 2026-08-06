FROM python:3.12-slim

# Node.js is required because the Agent SDK drives the Claude Code CLI under the hood
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir claude-agent-sdk fastapi uvicorn

WORKDIR /app
# Build context is the repo root, so the path includes the folder name
COPY claude-proxy/claude_proxy.py .

EXPOSE 8082
CMD ["uvicorn", "claude_proxy:app", "--host", "0.0.0.0", "--port", "8082"]
