# Docker image for serving the Iris FastAPI app
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    APP_HOME=/app \
    MLFLOW_TRACKING_URI="file:///app/mlruns" \
    MLFLOW_EXPERIMENT_NAME="iris_classification"

WORKDIR ${APP_HOME}

# No system packages needed because dependencies ship prebuilt wheels

# Copy dependency definitions first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY data ./data
COPY mlruns ./mlruns

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
