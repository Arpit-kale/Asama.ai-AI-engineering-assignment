# One image, two roles:
#   platform (default):  python -m app          -> :8000
#   demo agents:         python -m agents_sim   -> :9000
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY agents_sim ./agents_sim
COPY scripts ./scripts

# Unprivileged runtime user; SQLite lives on a volume mounted at /data.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser:appuser /data
USER appuser

ENV AN_ENV=production \
    AN_DB_PATH=/data/agent_network.db \
    HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1

CMD ["python", "-m", "app"]
