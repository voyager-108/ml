FROM python:3.10 AS python_base

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN pip install torch torchvision torchaudio

COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app/

RUN poetry install --no-dev --no-interaction --no-ansi

FROM python_base AS python_app

WORKDIR /app/
COPY ./ /app/
