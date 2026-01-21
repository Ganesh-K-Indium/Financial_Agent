from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from src.retrieval.base import VectorDBBase
from src.config import config
import uuid
import tqdm

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

class QdrantVectorDB(VectorDBBase):
    def __init__(self, collection_name: str = "financial_docs_hybrid"):
        if config.QDRANT_API_KEY:
            if config.QDRANT_HOST.startswith("http"):
                 self.client = QdrantClient(url=config.QDRANT_HOST, api_key=config.QDRANT_API_KEY, timeout=600)
            else:
                 self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, api_key=config.QDRANT_API_KEY, timeout=600)
        else:
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=600)
            
        self.collection_name = collection_name
        
        # Initialize Embedding Models
        # Dense: sentence-transformers/all-MiniLM-L6-v2
        self.dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        # Sparse: Qdrant/bm25
        self.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        # Late Interaction: colbert-ir/colbertv2.0
        self.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                # We need a dummy dense embedding to get the size for config
                # Usually 384 for all-MiniLM-L6-v2
                dense_size = 384 
                
                # We need dummy late interaction embedding to get size
                # Usually 128 for colbertv2.0
                late_size = 128
                
                self.client.create_collection(
                    self.collection_name,
                    vectors_config={
                        "all-MiniLM-L6-v2": models.VectorParams(
                            size=dense_size,
                            distance=models.Distance.COSINE,
                        ),
                        "colbertv2.0": models.VectorParams(
                            size=late_size,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM,
                            )
                        ),
                    },
                    sparse_vectors_config={
                        "bm25": models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    }
                )
                print(f"Created collection {self.collection_name} with Hybrid Config.")
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant or create collection: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Embeds and indexes documents using all 3 models.
        """
        if not documents:
            return
            
        print(f"Generating embeddings and upserting {len(documents)} chunks...")
        
        batch_size = 16 # Reduce batch size for heavy embedding generation
        
        # Convert documents dict list to just a list of texts for processing
        # We process in batches inside the loop
        
        # We iterate manually to match the generator style of fastembed properly
        # Or just do standard batching
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm.tqdm(range(0, len(documents), batch_size), total=total_batches, desc="Ingesting"):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [d["text"].replace("\n", " ") for d in batch_docs]
            
            try:
                # Generate Embeddings
                # Note: fastembed generators yield batches, but here we are passing a small batch explicitly.
                # safely wrap in list() to consume the generator for this small batch.
                
                dense_embeddings = list(self.dense_embedding_model.passage_embed(batch_texts))
                bm25_embeddings = list(self.bm25_embedding_model.passage_embed(batch_texts))
                late_interaction_embeddings = list(self.late_interaction_embedding_model.passage_embed(batch_texts))
                
                points = []
                for j, doc in enumerate(batch_docs):
                    point_id = doc.get("id", str(uuid.uuid4()))
                    
                    # Construct Point
                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            "all-MiniLM-L6-v2": dense_embeddings[j].tolist(),
                            "bm25": bm25_embeddings[j].as_object(),
                            "colbertv2.0": late_interaction_embeddings[j].tolist(),
                        },
                        payload=doc
                    ))
                    
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points
                )
                
            except Exception as e:
                print(f"Error acting on batch {i}: {e}")

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs hybrid search: Dense + Sparse + Late Interaction
        """
        try:
            # 1. Embed Query
            # query_embed returns a generator, consume it
            dense_query_vector = list(self.dense_embedding_model.query_embed(query))[0]
            sparse_query_vector = list(self.bm25_embedding_model.query_embed(query))[0]
            late_query_vector = list(self.late_interaction_embedding_model.query_embed(query))[0]

            # 2. Qdrant Search
            results = self.client.query_points(
                self.collection_name,
                prefetch=[
                    models.Prefetch(
                        prefetch=[
                            models.Prefetch(
                                query=dense_query_vector.tolist(),
                                using="all-MiniLM-L6-v2",
                                limit=100,
                            )
                        ],
                        query=models.SparseVector(**sparse_query_vector.as_object()),
                        using="bm25",
                        limit=50,
                    ),
                ],
                query=late_query_vector.tolist(),
                using="colbertv2.0",
                with_payload=True,
                limit=limit,
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": hit.payload
                }
                for hit in results.points
            ]
            
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
