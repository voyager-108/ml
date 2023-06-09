import numpy as np
import requests
import pickle
from .base import ObjectStore

class YandexPickleStorage(ObjectStore):
    """Object store implementation for Yandex.Cloud Storage 
    that stores objects as pickled Python objects. Storage is 
    used to preserve the state of the tracker between runs.
    
    Documentation: https://cloud.yandex.ru/docs/storage/
    """


    def __init__(self, bucket: str):
        """Constructor

        Args:
            bucket (str): Yandex.Cloud Storage bucket name
        """
        self.bucket = bucket

    def _base_url(self):
        """Base URL for the bucket"""
        return f'https://storage.yandexcloud.net/{self.bucket}'
    
    def _object_url(self, id):
        """URL for the object with the given id"""
        return f'{self._base_url()}/vectors/{id}.pkl'
    
    def get(self, id):
        """GET the object with the given id and deserialize it from pickle."""
        response = requests.get(self._object_url(id))
        if not response.ok:
            return None

        return pickle.loads(requests.get(self._object_url(id)).content)
    
    def put(self, id, vector):
        """Pickle the object and PUT it to the bucket."""
        response = requests.put(self._object_url(id), data=pickle.dumps(vector))
        if not response.ok:
            raise Exception(f'Failed to PUT vector {id} to {self._object_url(id)}: {response.status_code} {response.text}')
        return vector