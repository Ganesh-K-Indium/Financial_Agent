from src.utils.llm import llm_client

class HorizonAnalyzer:
    def analyze(self, business_text: str, competition_text: str) -> dict:
        """
        Analyzes Horizon: Moat durability, R&D, Market Share.
        """
        prompt = f"""
        Evaluate the company's Competitive Moat and Long-term Horizon:
        - Network effects, Switching costs, Intangibles?
        - R&D expenditure vs peers (is it sustainable?)
        - Market share trends.
        
        Text: {business_text[:3000]} {competition_text[:1000]}
        """
        analysis = llm_client.analyze_text(prompt)
        
        score_prompt = f"""
        Based on: "{analysis}", assign a Horizon Score from 0 to 100.
        - Wide Moat (Network effects, high switching costs) -> >85
        - Commodity product, low pricing power -> <50
        Return JSON {{ "score": int, "rationale": str }}
        """
        score_output = llm_client.specific_extraction(score_prompt, {"score": "int", "rationale": "string"})
        
        return {
            "analysis": analysis,
            "score": score_output.get("score", 50),
            "rationale": score_output.get("rationale", "N/A")
        }
