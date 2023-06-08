import os
from .yandex import YandexS3VectorStore

vector_store = YandexS3VectorStore(os.environ['VECTOR_STORE_BUCKET'])