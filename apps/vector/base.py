from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def get(self, id):
        pass

    @abstractmethod
    def put(self, id, vector):
        pass
