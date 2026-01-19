import redis
from typing import List, Dict
import json
from src.utils.memory_base import MemoryBase
from src.config import config

class RedisMemory(MemoryBase):
    def __init__(self, session_id: str = "default"):
        if config.REDIS_URL:
            # Use URL if provided (Easiest for Cloud)
            self.client = redis.from_url(config.REDIS_URL, decode_responses=True)
        else:
            # Fallback to individual params
            self.client = redis.Redis(
                host=config.REDIS_HOST, 
                port=config.REDIS_PORT, 
                username=config.REDIS_USERNAME,
                password=config.REDIS_PASSWORD,
                ssl=config.REDIS_SSL,
                decode_responses=True
            )
        self.session_id = session_id
        self.key = f"fin_agent:history:{self.session_id}"

    def add_message(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self.client.rpush(self.key, json.dumps(msg))

    def get_history(self) -> List[Dict[str, str]]:
        msgs = self.client.lrange(self.key, 0, -1)
        return [json.loads(m) for m in msgs]

    def clear(self) -> None:
        self.client.delete(self.key)
