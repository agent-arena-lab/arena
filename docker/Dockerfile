# Harness image (CPU only – inference runs in llama-server on the host GPU).
# python:3.11-slim is multi-arch (linux/arm64 for Jetson, linux/amd64 for laptops).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[analysis,dev]" && useradd -m -u 1000 harness

COPY tests ./tests
# configs/, prompts/ and runs/ are bind-mounted at runtime (see docker-compose.yml)
USER harness
ENTRYPOINT ["coopsab"]
CMD ["--help"]
