FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

# Install deps (simple approach)
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir .

COPY . /app

ENV APP_ENV=container
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
