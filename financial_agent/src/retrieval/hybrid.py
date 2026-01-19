from typing import List, Dict, Any
import os
import json
import uuid
from rank_bm25 import BM25Okapi
from src.retrieval.vector_db import QdrantVectorDB
from src.retrieval.reranker import Reranker
from src.config import config
from src.utils.llm import llm_client
from openai import OpenAI

class HybridRetriever:
    def __init__(self, ticker: str = ""):
        self.vector_db = QdrantVectorDB()
        self.reranker = Reranker() # Uses FlashRank
        self.embedding_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.ticker = ticker
        self.bm25 = None
        self.corpus_map = {} # Map Index -> Chunk
        
        if ticker:
            self.load_bm25(ticker)
            
    def load_bm25(self, ticker: str):
        """
        Loads the corpus for the ticker and builds BM25 index.
        """
        corpus_path = os.path.join(config.DATA_DIR, "corpus", f"{ticker}.json")
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r") as f:
                    chunks = json.load(f)
                
                # Tokenize
                tokenized_corpus = [doc["text"].lower().split(" ") for doc in chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.corpus_map = {i: doc for i, doc in enumerate(chunks)}
                print(f"Loaded BM25 index for {ticker} with {len(chunks)} chunks.")
            except Exception as e:
                print(f"Error loading BM25: {e}")
        else:
            print(f"No corpus found for {ticker} at {corpus_path}. BM25 disabled.")

    def _embed_query(self, query: str) -> List[float]:
        response = self.embedding_client.embeddings.create(input=[query], model=config.EMBEDDING_MODEL)
        return response.data[0].embedding

    def generate_queries(self, original_query: str) -> List[str]:
        """
        Uses LLM to generate variations of the query.
        """
        prompt = f"""
        You are a financial research assistant. 
        Generate 3 different search queries based on this user question to strictly improve retrieval recall. 
        Note: We are focusing on data from 10-K filings. Please generate queries that are extremely relevant to 10-K filings.
        User Question: "{original_query}"
        Return ONLY the queries, one per line.
        """
        response = llm_client.analyze_text(prompt)
        queries = [q.strip() for q in response.splitlines() if q.strip()]
        return [original_query] + queries[:3]

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # 1. MultiQuery
        queries = self.generate_queries(query)
        print(f"Generated queries: {queries}")
        
        all_candidates = {} # Map ID (or text hash) -> Doc
        
        for q in queries:
            # Dense
            try:
                q_vec = self._embed_query(q)
                dense_hits = self.vector_db.search(q_vec, limit=limit)
                for hit in dense_hits:
                    # Dedupe by text using hash if ID is random UUID
                    # But Qdrant hits have IDs.
                    # Ideally we use stable IDs. 
                    # If ID is UUID per chunk, duplicate content might have different IDs if re-ingested.
                    # We assume ID is unique per chunk.
                    doc_id = hit["id"]
                    hit["payload"]["retrieval_source"] = "dense"
                    all_candidates[doc_id] = hit["payload"]
            except Exception as e:
                print(f"Dense search error: {e}")
                
            # Sparse (BM25)
            if self.bm25:
                q_tokens = q.lower().split(" ")
                bm25_scores = self.bm25.get_scores(q_tokens)
                top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:limit]
                for idx in top_n:
                    doc = self.corpus_map[idx]
                    # We don't have the UUID from Qdrant here easily unless stored in corpus.
                    # We rely on text dedupe or just adding it.
                    # Let's create a synthesis ID or use hash
                    doc["retrieval_source"] = "bm25"
                    all_candidates[f"bm25_{idx}"] = doc
        
        candidates_list = list(all_candidates.values())
        print(f"Total candidates before reranking: {len(candidates_list)}")
        
        if not candidates_list:
            return []
            
        # 2. Rerank (FlashRank)
        try:
            # FlashRank reranks based on the ORIGINAL query (most relevant intent)
            final_results = self.reranker.rerank(query, candidates_list, top_k=limit)
            return final_results
        except Exception as e:
            print(f"Reranking failed: {e}. Returning raw.")
            return candidates_list[:limit]
