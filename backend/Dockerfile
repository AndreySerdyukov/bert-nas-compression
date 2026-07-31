# Serving image: FastAPI + CPU-only torch, with the model snapshots baked in.
FROM python:3.12-slim

WORKDIR /app

# Created before the COPYs so --chown=1000:1000 has a user to point at.
RUN useradd --create-home --uid 1000 appuser

# Dependencies in their own layer. The list is maintained by hand and has drifted from
# pyproject.toml in sibling projects before, so tests/test_packaging.py asserts the two match.
# The CPU wheel index matters: the default torch wheel drags in ~2 GB of CUDA that a
# CPU-only image can never use.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2.7" \
    "pydantic-settings>=2.3" \
    "torch>=2.2" \
    "transformers>=4.44" \
    "tokenizers>=0.19" \
    "huggingface-hub>=0.24" \
    "safetensors>=0.4" \
    "scikit-learn>=1.5" \
    "joblib>=1.4" \
    "numpy>=2.0" \
    "pandas>=2.2"

# Weights first, code second: the snapshots are ~1.1 GB and change once, the code changes
# constantly. Reversing these two lines re-copies a gigabyte on every edit.
COPY --chown=1000:1000 models ./models
COPY --chown=1000:1000 data ./data
COPY --chown=1000:1000 app ./app

USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Nothing may reach the network at serve time: an incomplete snapshot would otherwise make
    # transformers silently download a gigabyte inside create_app(), before the port is bound.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    # Latency is a published output of this app, so the thread count is fixed rather than left
    # to whatever the host happens to offer.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    APP_MODELS_DIR=/app/models \
    APP_DATA_DIR=/app/data

EXPOSE 8000

# start-period covers loading up to five BERTs from disk before the first probe counts.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
