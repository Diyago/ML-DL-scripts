FROM python:latest
WORKDIR /app
COPY ann ann
COPY config config
COPY data data
COPY main.py main.py
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN python -m pip install --upgrade pip && pip install poetry && poetry install --no-dev
CMD poetry run uvicorn main:app --host 0.0.0.0 --port 80