from src.utils.llm import llm_client

class PerformanceAnalyzer:
    def analyze(self, financial_text: str) -> dict:
        """
        Analyzes Performance: Revenue, Margins, Cash Flow.
        """
        prompt = f"""
        Extract the following financial metrics from the text if available:
        - Revenue CAGR (approx 5 year or recent)
        - Operating Margin (latest)
        - Operating Cash Flow vs Net Income (Ratio)
        - Any non-recurring items
        
        Text: {financial_text[:4000]}
        """
        analysis = llm_client.analyze_text(prompt)
        
        score_prompt = f"""
        Based on: "{analysis}", assign a Performance Score from 0 to 100.
        Rules: 
        - High margins and CAGR > 20% -> High Score (>80)
        - OCF < Net Income -> Red Flag (Lower Score)
        Return JSON {{ "score": int, "rationale": str }}
        """
        score_output = llm_client.specific_extraction(score_prompt, {"score": "int", "rationale": "string"})
        
        return {
            "analysis": analysis,
            "score": score_output.get("score", 50),
            "rationale": score_output.get("rationale", "N/A")
        }
