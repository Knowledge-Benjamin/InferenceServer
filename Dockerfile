# Use Python 3.10 slim image for the inference server
FROM python:3.10-slim

# Set environment variables to avoid python buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies (copy requirements first for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Use Hugging Face cache volume for persistent model storage across restarts
ENV TRANSFORMERS_CACHE=/data/hf_cache
ENV HF_HOME=/data/hf_cache
VOLUME /data/hf_cache

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]