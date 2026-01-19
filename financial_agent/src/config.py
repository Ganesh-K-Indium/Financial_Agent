import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # None by default
    
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_SSL = os.getenv("REDIS_SSL", "False").lower() in ("true", "1", "yes")
    REDIS_URL = os.getenv("REDIS_URL") # Optional: Overrides individual settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # SEC
    SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "FinancialResearchAgent contact@example.com")
    
    # Model details
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o"
    
    # Retrieval
    MIN_RETRIEVAL_SCORE = 0.7
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "../data")

config = Config()
