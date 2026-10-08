FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app

# `src/` has to be present before the install: hatchling builds a wheel from
# it, and a wheel built without it installs no modules at all.
COPY pyproject.toml uv.lock README.md app.py ./
COPY src/ ./src/
COPY data/ ./data/
# `--frozen` keeps the image on the same versions the tests and CI used.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-dev --extra app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')"

# `--reload` is off, and `shiny run` needs the file path rather than the
# console script, which binds loopback by default.
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]
