from src.utils.llm import llm_client

class AlignmentAnalyzer:
    def analyze(self, mda_text: str, insider_text: str) -> dict:
        """
        Analyzes Alignment: Management Sentiment + Insider Trading.
        """
        # 1. Sentiment Analysis
        sentiment_prompt = f"""
        Analyze the tone of the following Management Discussion and Analysis (MD&A) excerpt. 
        Determine if it is Defensive, Neutral, or Confident. 
        Cite 1-2 quotes nicely.
        Text: {mda_text[:4000]}
        """
        sentiment_analysis = llm_client.analyze_text(sentiment_prompt)
        
        # 2. Insider Trading Analysis
        insider_prompt = f"""
        Summarize the insider trading activity described here. 
        Focus on net buying/selling and CEO/CFO moves.
        Text: {insider_text[:2000]}
        """
        insider_analysis = llm_client.analyze_text(insider_prompt)
        
        # 3. Scoring (simplified logic for now)
        # In production, we'd ask the LLM to output a score explicitly.
        score_prompt = f"""
        Based on the sentiment: "{sentiment_analysis}" and insider activity: "{insider_analysis}",
        assign an Alignment Score from 0 to 100. return JSON {{ "score": int, "rationale": str }}
        """
        score_output = llm_client.specific_extraction(score_prompt, {"score": "int", "rationale": "string"})
        
        return {
            "sentiment": sentiment_analysis,
            "insider": insider_analysis,
            "score": score_output.get("score", 50),
            "rationale": score_output.get("rationale", "N/A"),
            "analysis": f"Sentiment: {sentiment_analysis}\n\nInsider Activity: {insider_analysis}"
        }
