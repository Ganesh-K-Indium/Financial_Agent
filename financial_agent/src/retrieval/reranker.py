from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest

class Reranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        # FlashRank defaults to a quantized onnx model which is very fast.
        # model_name here can be passed if we want specific flashrank models, 
        # but the library handles it.
        self.ranker = Ranker(model_name=model_name, cache_dir="opt")

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        # FlashRank expects list of dicts with {"text": "...", "id": ...} or similar (pass_through)
        # We need to adapt our docs structure.
        
        passages = []
        for doc in docs:
            # metadata is in payload usually
            passages.append({
                "id": doc.get("id", "0"),
                "text": doc.get("text", "") if "text" in doc else doc.get("payload", {}).get("text", ""),
                "meta": doc # keep original doc
            })
            
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # Map back to our structure
        reranked_docs = []
        for hit in results[:top_k]:
            original_doc = hit["meta"]
            original_doc["rerank_score"] = hit["score"]
            reranked_docs.append(original_doc)
            
        return reranked_docs
