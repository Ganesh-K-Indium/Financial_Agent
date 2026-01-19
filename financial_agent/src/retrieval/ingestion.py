import re
import json
import os
import uuid
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config

class IngestionEngine:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.corpus_dir = os.path.join(config.DATA_DIR, "corpus")
        os.makedirs(self.corpus_dir, exist_ok=True)

    def process_document(self, ticker: str, doc_type: str, content: Any, year: str = "Latest", source_url: str = "") -> List[Dict[str, Any]]:
        """
        Chunks the document and prepares it for ingestion.
        content: Can be a raw string OR a list of dicts (for multimodal: [{'type': 'text', 'content': ...}])
        year: Year of the filing (e.g., "2024"). Injected into text for context.
        """
        chunks = []
        
        # Metadata Header to prepend to each chunk
        meta_header = f"[Ticker: {ticker} | Year: {year} | Type: {doc_type}]"
        
        # Handle Multimodal List
        if isinstance(content, list):
            full_text = ""
            for block in content:
                # Merge into a single stream for context, or treat image descriptions as text?
                # Best strategy: Interleave them.
                full_text += f"\n\n{block['content']}"
            
            # Now chunk the merged text
            chunks = self.text_splitter.split_text(full_text)
            
        elif isinstance(content, str):
            chunks = self.text_splitter.split_text(content)
        
        processed_chunks = []
        # Define a namespace for our app
        APP_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "financial_agent")
        
        for i, chunk in enumerate(chunks):
            # Inject Metadata into the text itself for better LLM context
            enriched_text = f"{meta_header}\n{chunk}"
            
            # Generate Deterministic ID
            # Combined key: ticker + year + doc_type + chunk_index + content_hash
            # We assume chunk_index is stable for the same file content.
            # Even better: mix in a hash of the content itself.
            content_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            unique_str = f"{ticker}_{year}_{doc_type}_{i}_{content_hash}"
            chunk_uuid = str(uuid.uuid5(APP_NAMESPACE, unique_str))
            
            metadata = {
                "id": chunk_uuid,
                "ticker": ticker,
                "doc_type": doc_type,
                "year": year,
                "source": source_url,
                "chunk_id": i,
                "text": enriched_text 
            }
            # We will rely on the vector DB to generate UUIDs or handle ID generation if needed
            processed_chunks.append(metadata)
            
        # Save to local corpus for BM25
        corpus_path = os.path.join(self.corpus_dir, f"{ticker}.json")
        try:
            with open(corpus_path, "w") as f:
                json.dump(processed_chunks, f, indent=2)
            print(f"Saved corpus for {ticker} to {corpus_path}")
        except Exception as e:
            print(f"Error saving corpus: {e}")
            
        return processed_chunks

    def extract_tables(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Placeholder for refined table extraction.
        Ideally uses weak HTML parsing to keep table structure intact.
        """
        # Todo: Implement beautifulsoup table parsing
        pass
