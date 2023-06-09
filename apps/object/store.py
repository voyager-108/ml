import os
from .yandex import YandexPickleStorage

storage = YandexPickleStorage(os.environ['OBJECT_STORE_BUCKET'])