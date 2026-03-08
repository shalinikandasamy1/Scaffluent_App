# FireEye Dashboard — GPU-enabled Docker image
# Uses NVIDIA CUDA base for YOLO GPU inference + NiceGUI web dashboard
#
# VRAM notes (Tesla P4 = 8GB):
#   - YOLO model: ~200MB VRAM (tiny, always fits)
#   - VLM (Qwen2.5-VL-7B via Ollama): ~6GB VRAM — does NOT fit alongside YOLO on P4
#   - Default config: OpenRouter (cloud VLM) + YOLO GPU = ~200MB VRAM total = fine
#   - For local VLM: use RTX 3060 (12GB) or set FIREEYE_YOLO_DEVICE=cpu on P4

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Create and activate venv
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt nicegui==3.8.0

# Copy application code
COPY app/ app/
COPY prompts/ prompts/
COPY models/ models/
COPY ui_app.py .
COPY .env.example .env.example

# Copy test data for evaluation (optional, can be excluded in .dockerignore)
COPY test_data/ test_data/
COPY evaluate.py .

# Create directories for runtime data
RUN mkdir -p images/input images/output audit_logs

# Default port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090')" || exit 1

# Run the dashboard
CMD ["python", "ui_app.py"]
