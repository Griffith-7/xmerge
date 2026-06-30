# ───────────────────────────────────────────────────────────────
# xmerge — Multi-stage Docker build
# ───────────────────────────────────────────────────────────────

# ---- Build stage ----
FROM python:3.12-slim AS build

WORKDIR /build
COPY . .

RUN pip install --upgrade pip && \
    pip install build && \
    python -m build --wheel


# ---- Runtime (CPU) ----
FROM python:3.12-slim AS cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/dist/*.whl /tmp/
RUN pip install --upgrade pip && \
    pip install /tmp/xmerge-*.whl --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers datasets numpy

ENTRYPOINT ["xmerge"]
CMD ["--help"]


# ---- Runtime (CUDA 12.1) ----
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS cuda

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/dist/*.whl /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install /tmp/xmerge-*.whl && \
    pip3 install transformers datasets numpy

ENTRYPOINT ["xmerge"]
CMD ["--help"]
