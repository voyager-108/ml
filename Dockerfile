FROM python:3.10 AS python_base

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app/

RUN poetry install --no-dev --no-interaction --no-ansi
RUN pip uninstall torch torchvision 
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENV YOLO_CONFIG "/app/config/yolo/default.yaml"

FROM python_base AS python_app

WORKDIR /app/
COPY ./ /app/