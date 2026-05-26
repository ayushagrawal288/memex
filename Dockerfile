FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the ONNX model weights into the image (~67 MB) so the container
# starts without any network I/O.
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY . .

# WORKERS controls uvicorn worker count (default 4).
# Set WORKERS=1 in docker-compose.override.yml for hot-reload dev mode:
#   command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4}"]
