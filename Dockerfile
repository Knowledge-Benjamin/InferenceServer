# Use Python 3.10 slim image for the inference server
FROM python:3.10-slim

# Set environment variables to avoid python buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies (copy requirements first for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding model into the image (prevents download on every container boot)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]