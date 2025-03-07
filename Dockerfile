# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app into the container
COPY fastapi_app/ .

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
