from src.utils.llm import llm_client

class LiquidityAnalyzer:
    def analyze(self, risk_text: str, macro_text: str) -> dict:
        """
        Analyzes Liquidity: Sector headwinds, commodity exposure, interest rates.
        """
        prompt = f"""
        Analyze the following text for Liquidity and Environmental risks:
        - Identify sector-specific headwinds/tailwinds.
        - Commodity or input cost exposure.
        - Interest rate sensitivity.
        - Competitive pressures.
        
        Text: {risk_text[:3000]} {macro_text[:1000]}
        """
        analysis = llm_client.analyze_text(prompt)
        
        score_prompt = f"""
        Based on the risk analysis: "{analysis}", assign a Liquidity/Risk Score from 0 to 100.
        (Higher is better/safer).
        - Major regulatory or macro headwinds -> Lower Score (<60)
        - Strong capital structure and tailwinds -> Higher Score (>80)
        Return JSON {{ "score": int, "rationale": str }}
        """
        score_output = llm_client.specific_extraction(score_prompt, {"score": "int", "rationale": "string"})
        
        return {
            "analysis": analysis,
            "score": score_output.get("score", 50),
            "rationale": score_output.get("rationale", "N/A")
        }
