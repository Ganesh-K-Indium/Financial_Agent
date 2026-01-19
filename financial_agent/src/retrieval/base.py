from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBBase(ABC):
    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass
