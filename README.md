---
title: Embedding Inference Server
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
---

# Embedding Inference Server

A high-performance FastAPI server for generating sentence embeddings using `sentence-transformers/all-mpnet-base-v2`. Designed for the NNI Truth Graph pipeline, optimized for speed, security, and deployment on Hugging Face Spaces.

## Features

- **Batch Processing**: Handles up to 32 texts per request for efficiency
- **GPU Acceleration**: Automatically uses CUDA if available
- **Security**: API key authentication
- **Async**: Concurrent request handling with thread pool execution
- **Structured Logging**: Loguru for production logging
- **Health Checks**: `/health` and `/version` endpoints
- **Production Ready**: Error handling, validation, and middleware

## Endpoints

### POST /embed
Generate embeddings for texts.

**Request Body:**
```json
{
  "texts": ["First text to embed", "Second text"]
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

**Headers:** `Authorization: Bearer <INFERENCE_API_KEY>`

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "sentence-transformers/all-mpnet-base-v2",
  "device": "cuda",
  "version": "1.0.0"
}
```

### GET /version
Get server version.

## Deployment on Hugging Face Spaces

1. **Create a separate GitHub repository** for the Inference server (e.g., `InferenceServer`).
2. **Copy files** from `Inference server/` to the new repo root: `main.py`, `requirements.txt`, `README.md`, `Dockerfile`.
3. **Add GitHub Actions workflow** (see `.github/workflows/deploy_inference_hf.yml` in the main repo for reference).
4. **Create a new HF Space** with Docker support.
5. **Set secrets** in the new repo: `HF_TOKEN` (your Hugging Face token).
6. **Set environment variables** in the HF Space settings:
   - `INFERENCE_API_KEY`: A secure API key (generate randomly, e.g., `openssl rand -hex 32`).
7. **Push to main** in the new repo to trigger deployment.

The server will be available at `https://<username>-<space-name>.hf.space`.

## Usage in NNI Truth Graph

The main AI engine uses `inference_pool.py` to call this server. Set these environment variables in the main TruthEngine Space:

- `INFERENCE_SERVER_URL`: The URL of the deployed Inference server.
- `INFERENCE_API_KEY`: The API key for authentication.

## Local Development

```bash
pip install -r requirements.txt
export INFERENCE_API_KEY="your-key"
uvicorn main:app --reload
```

Test with:
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'
```

response = requests.post(
    "https://your-space-name.hf.space/embed",
    json={"texts": texts},
    headers={"Authorization": f"Bearer {API_KEY}"}
)
embeddings = response.json()["embeddings"]
```

## Performance Notes

- Model loads in ~20-30s on CPU, faster on GPU.
- Batch size capped at 32 to prevent OOM.
- Monitor Spaces logs for resource usage.