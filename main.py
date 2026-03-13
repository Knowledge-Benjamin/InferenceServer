"""
World-Class Embedding Inference Server
======================================

A high-performance, secure FastAPI server for generating embeddings using
sentence-transformers/all-mpnet-base-v2. Optimized for speed, scalability,
and deployment on Hugging Face Spaces.

Features:
- Batch processing for multiple texts
- GPU acceleration when available
- Input validation and error handling
- API key authentication
- Async endpoints for concurrency
- Structured logging with loguru
- Health and version endpoints
"""

import os
import asyncio
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from loguru import logger

# Model configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 32  # Adjust based on memory
API_KEY = os.getenv("INFERENCE_API_KEY", "default-key-change-in-production")  # Set in Spaces secrets
VERSION = "1.0.0"

# Global model instance
model: SentenceTransformer = None

# Security
security = HTTPBearer()

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_BATCH_SIZE,
                             description="List of texts to embed")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE}")
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Shutdown: Cleanup if needed
    logger.info("Shutting down server")

app = FastAPI(
    title="Embedding Inference Server",
    description="High-performance embedding service for NNI Truth Graph",
    version=VERSION,
    lifespan=lifespan
)

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest, _: Any = Depends(verify_api_key)):
    """
    Generate embeddings for a batch of texts.

    - **texts**: List of strings to embed (1 to 32 items)
    - Returns: List of 768-dimensional float vectors
    """
    try:
        logger.info(f"Processing batch of {len(request.texts)} texts")
        # Use a lambda to pass keyword arguments to model.encode in the thread pool
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.encode(request.texts, convert_to_tensor=False, show_progress_bar=False)
        )
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return EmbedResponse(embeddings=embeddings.tolist())
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Embedding failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME, "device": DEVICE, "version": VERSION}

@app.get("/version")
async def get_version():
    """Get server version"""
    return {"version": VERSION}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=7860)