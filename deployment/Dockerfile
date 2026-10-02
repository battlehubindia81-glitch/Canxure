FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CANXURE_PROJECT_DIR=/app \
    CANXURE_BASE_DIR=/app/deployment/artifacts \
    CANXURE_CHECKPOINT=epoch_015.pt

WORKDIR /app

# Production dependencies only.
COPY deployment/requirements.txt /app/deployment/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/deployment/requirements.txt

# Production application source.
COPY api /app/api
COPY src /app/src

# Self-contained real-data inference artifacts.
COPY deployment/artifacts /app/deployment/artifacts

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 canxure \
    && chown -R canxure:canxure /app

USER canxure

EXPOSE 8000

CMD ["python", "api/server.py"]
