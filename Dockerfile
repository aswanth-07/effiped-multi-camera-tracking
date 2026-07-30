# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS web
WORKDIR /build/apps/web
COPY apps/web/package*.json ./
RUN npm ci
COPY apps/web ./
COPY docs /build/docs
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EFFIPED_DEVICE=auto \
    EFFIPED_WEIGHTS_DIR=/weights \
    EFFIPED_RUNTIME_DIR=/runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY apps/api ./apps/api
COPY apps/__init__.py ./apps/__init__.py
COPY configs ./configs
COPY --from=web /build/apps/web/dist ./apps/web/dist
RUN pip install --no-cache-dir ".[runtime]"
VOLUME ["/weights", "/runtime"]
EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
