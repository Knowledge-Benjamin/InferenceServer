FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install requirements as root so they are globally accessible
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the non-root user required by Cloud Run & HF Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user

# Set environment paths to match the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/hub \
    HF_HOME=/home/user/.cache/huggingface

# Pre-download the model at build time strictly AS the non-root user.
# This guarantees it saves to `/home/user/.cache` and is instantly loaded on container boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Copy application files and transfer ownership to user
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]