from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.retrieval.base import VectorDBBase
from src.config import config
from openai import OpenAI
import uuid

class QdrantVectorDB(VectorDBBase):
    def __init__(self, collection_name: str = "financial_docs"):
        if config.QDRANT_API_KEY:
            # Cloud usually implies URL + Key
            # If QDRANT_HOST is a full URL (https://...), QdrantClient handles it as 'url'.
            # If it's just a hostname, we pass host/port.
            # Safe bet: pass url=HOST if it starts with http, else host=HOST.
            if config.QDRANT_HOST.startswith("http"):
                 self.client = QdrantClient(url=config.QDRANT_HOST, api_key=config.QDRANT_API_KEY)
            else:
                 self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, api_key=config.QDRANT_API_KEY)
        else:
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            
        # Check if Qdrant is live, if not fallback to memory?
        # For production readiness we assume it works or we crash.
        self.collection_name = collection_name
        self._ensure_collection()
        self.embedding_client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # text-embedding-3-small
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant or create collection: {e}")

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.embedding_client.embeddings.create(input=[text], model=config.EMBEDDING_MODEL).data[0].embedding

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Embeds and indexes documents.
        documents: list of dicts with 'text' and other metadata.
        """
        if not documents:
            return
            
        points = []
        print(f"Generatings embeddings and upserting {len(documents)} chunks...")
        
        # Batch Size
        batch_size = 100
        
        # tqdm for progress
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Ingesting"):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [d["text"].replace("\n", " ") for d in batch_docs]
            
            try:
                # Batch Embedding Call
                response = self.embedding_client.embeddings.create(input=batch_texts, model=config.EMBEDDING_MODEL)
                embeddings_data = response.data
                
                batch_points = []
                # Zip and Append
                for j, doc in enumerate(batch_docs):
                    # Use provided ID if available, else generate random UUID
                    point_id = doc.get("id", str(uuid.uuid4()))
                    vector = embeddings_data[j].embedding
                    
                    batch_points.append(models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=doc
                    ))
                
                # Upsert Batch Immediately
                if batch_points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    
            except Exception as e:
                print(f"Error acting on batch {i}: {e}")

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            print(f"Search failed: {e}")
            return []
