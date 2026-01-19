from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MemoryBase(ABC):
    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def get_history(self) -> List[Dict[str, str]]:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
