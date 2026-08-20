FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required by some Python packages (e.g. psycopg, docling)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
