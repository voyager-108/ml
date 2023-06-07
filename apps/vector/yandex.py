import numpy as np
import requests
import pickle
from .base import VectorStore

class YandexS3VectorStore(VectorStore):
    def __init__(self, bucket: str):
        self.bucket = bucket

    def _base_url(self):
        return f'https://storage.yandexcloud.net/{self.bucket}'
    
    def _vector_url(self, id):
        return f'{self._base_url()}/vectors/{id}.pkl'
    
    def get(self, id):
        response = requests.get(self._vector_url(id))
        if not response.ok:
            return None

        return pickle.loads(requests.get(self._vector_url(id)).content)
    
    def put(self, id, vector):
        response = requests.put(self._vector_url(id), data=pickle.dumps(vector))
        if not response.ok:
            raise Exception(f'Failed to PUT vector {id} to {self._vector_url(id)}: {response.status_code} {response.text}')
        return vector