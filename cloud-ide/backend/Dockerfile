# Backend API Server for Orca Cloud IDE
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server_simple.py .

# Create necessary directories
RUN mkdir -p /tmp/orca-projects

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server_simple:app", "--host", "0.0.0.0", "--port", "8000"]
