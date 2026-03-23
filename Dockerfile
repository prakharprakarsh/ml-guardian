FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p artifacts data reports_output audit_logs

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run a single monitoring cycle
CMD ["python", "-m", "core.orchestrator"]
