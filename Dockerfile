# Governed AI Service — production image.
#
# The image is an immutable (code + corpus + index + embedding model) bundle: everything needed to
# answer is baked in at build time, so the running container makes no downloads. The only
# outbound call it makes is to the Anthropic API, and only with PII already redacted.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch first; the default wheel pulls in ~2 GB of CUDA libraries we do not use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the local embedding model into the image.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')"

COPY app.py ./
COPY src/ src/
COPY data/ data/

# Build the vector index from the bundled corpus, then run as an unprivileged user.
RUN python -m src.ingest \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/var \
    && chown -R appuser /app/var /app/data/index

USER appuser

ENV HF_HUB_OFFLINE=1 \
    AUDIT_DB_PATH=/app/var/audit.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
