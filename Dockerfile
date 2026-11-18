FROM python:3.11-slim

LABEL maintainer="Macadamia AI Team"
LABEL description="Macadamia Quality Grading AI System — Backend API + Streamlit UI"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (layer-cached) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────────
COPY config.yaml      .
COPY src/             ./src/
COPY deployment/      ./deployment/
COPY data/generate_sample_data.py ./data/generate_sample_data.py
COPY data/annotations/ ./data/annotations/

# ── Generate ONNX model weights at build time ──────────────────────────────────
RUN python src/models/model_builder.py

# ── Create necessary runtime directories ──────────────────────────────────────
RUN mkdir -p data/raw data/processed

# ── Expose ports: 8000 = FastAPI REST API, 8501 = Streamlit Dashboard ─────────
EXPOSE 8000 8501

# ── Entrypoint: start both API and UI via a simple launcher script ─────────────
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
