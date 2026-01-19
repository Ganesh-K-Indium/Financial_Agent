from typing import List, Dict
from src.config import config
import requests

class WebSearch:
    def __init__(self):
        self.api_key = config.TAVILY_API_KEY

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Executes a web search using Tavily API.
        """
        if not self.api_key:
            print("Warning: TAVILY_API_KEY not set. Returning mock data.")
            return [
                {"title": "Missing Config", "snippet": "Tavily API key is missing in .env."}
            ]
            
        print(f"Searching web for: {query}")
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")
                }
                for r in results
            ]
        except Exception as e:
            print(f"Tavily Search Error: {e}")
            return []

web_search_tool = WebSearch()
