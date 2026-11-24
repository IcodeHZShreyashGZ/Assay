# Multi-stage production container for Assay API and Demo UI
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY src/ src/
COPY demo/ demo/
COPY eval/ eval/
COPY data/ data/

ENV PYTHONPATH=/app
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1

EXPOSE 8000 8501

# Default command: run FastAPI service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
