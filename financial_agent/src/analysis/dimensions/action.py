from src.utils.llm import llm_client

class ActionAnalyzer:
    def analyze(self, market_data_text: str, recent_news: str) -> dict:
        """
        Analyzes Action: Valuation (P/E, EV/EBITDA), Technicals, Catalysts.
        """
        prompt = f"""
        Provide a Timing/Action assessment:
        - Valuation context (Expensive/Cheap relative to history/peers).
        - Recent price action (if mentioned).
        - Upcoming catalysts (Earnings, Product launches).
        
        Text: {market_data_text[:3000]} {recent_news[:1000]}
        """
        analysis = llm_client.analyze_text(prompt)
        
        score_prompt = f"""
        Based on: "{analysis}", assign an Action/Timing Score from 0 to 100.
        - Undervalued + Catalysts -> >80
        - Overvalued + No Catalysts -> <40
        Return JSON {{ "score": int, "rationale": str }}
        """
        score_output = llm_client.specific_extraction(score_prompt, {"score": "int", "rationale": "string"})
        
        return {
            "analysis": analysis,
            "score": score_output.get("score", 50),
            "rationale": score_output.get("rationale", "N/A")
        }
