# Build the React client once, then serve it from the FastAPI process.
FROM node:20-alpine AS web-build
WORKDIR /build/apps/web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.12.13-slim AS runtime
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    RFMATCH_ARTIFACTS_DIR=/var/lib/rfmatch/artifacts \
    RFMATCH_PROJECTS_DIR=/var/lib/rfmatch/projects \
    RFMATCH_SNP_DIR=/var/lib/rfmatch/snp
WORKDIR /app

COPY packages/rfmatch-core/ /app/packages/rfmatch-core/
COPY apps/api/requirements.txt /app/apps/api/requirements.txt
WORKDIR /app/apps/api
# The API adds the core's source directory to sys.path at startup. Avoid an
# unnecessary editable build of this local pure-Python package in the image.
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" \
    "numpy>=1.24.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.23.0" \
    "pydantic>=2.0.0" \
    "python-multipart>=0.0.6" \
    "reportlab>=3.6.0"

COPY apps/api/ /app/apps/api/
COPY data/ /app/data/
COPY --from=web-build /build/apps/web/dist/ /app/apps/web/dist/

RUN useradd --system --uid 10001 --create-home rfmatch \
    && mkdir -p /var/lib/rfmatch/artifacts /var/lib/rfmatch/projects /var/lib/rfmatch/snp \
    && chown -R rfmatch:rfmatch /var/lib/rfmatch
USER rfmatch
WORKDIR /app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.server:app", "--app-dir", "/app/apps/api", "--host", "0.0.0.0", "--port", "8000"]
