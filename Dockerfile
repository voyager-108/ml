FROM python:3.10 AS python_base

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app/

RUN poetry install --no-dev --no-interaction --no-ansi
RUN pip uninstall -y torch torchvision 
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# This can only be installed on Linux
RUN pip install lap 
RUN wget -O yolo.pt https://storage.yandexcloud.net/voyager-108/best.pt 
RUN mkdir /app/models
RUN mv yolo.pt /app/models/yolo.pt

ENV YOLO_CONFIG /app/config/yolo/default.yaml
ENV YOLO_PT_PATH /app/models/yolo.pt

FROM python_base AS python_app

WORKDIR /app/
COPY ./ /app/
