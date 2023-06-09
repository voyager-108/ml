from abc import ABC, abstractmethod

class ObjectStore(ABC):
    """Abstract class for object store"""

    @abstractmethod
    def get(self, id):
        """Get the object with the given id."""
        pass

    @abstractmethod
    def put(self, id, object):
        """Put the object with the given id."""
        pass
