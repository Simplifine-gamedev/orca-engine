# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Set environment variable for production
ENV FLASK_ENV=production

# Run the application with Gunicorn (Cloud Run provides PORT)
ENV PORT=8080
# Use shell form so $PORT is expanded at runtime by the shell
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 120 app:app"]