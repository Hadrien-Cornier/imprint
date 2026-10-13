FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/uv && \
    /opt/uv/bin/pip install --no-cache-dir uv
ENV PATH=/opt/uv/bin:/workspace/imprint/.venv/bin:$PATH

WORKDIR /workspace/imprint
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PYTHONPATH=/workspace/imprint/src:/workspace/imprint/packages/openpi-client/src
CMD ["bash"]
