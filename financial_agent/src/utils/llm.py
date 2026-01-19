from openai import OpenAI
from src.config import config
import json

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL

    def analyze_text(self, prompt: str, system_prompt: str = "You are a financial analyst.") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def specific_extraction(self, text: str, schema: dict) -> dict:
        """
        Uses JSON mode to extract specific fields.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Extract the following fields using this schema: {json.dumps(schema)}"},
                {"role": "user", "content": text}
            ],
            response_format={ "type": "json_object" },
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}

llm_client = LLMClient()
