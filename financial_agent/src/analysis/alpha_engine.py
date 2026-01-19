from typing import Dict, Any
from src.analysis.dimensions.alignment import AlignmentAnalyzer
from src.analysis.dimensions.liquidity import LiquidityAnalyzer
from src.analysis.dimensions.performance import PerformanceAnalyzer
from src.analysis.dimensions.horizon import HorizonAnalyzer
from src.analysis.dimensions.action import ActionAnalyzer

class AlphaEngine:
    def __init__(self):
        self.alignment = AlignmentAnalyzer()
        self.liquidity = LiquidityAnalyzer()
        self.performance = PerformanceAnalyzer()
        self.horizon = HorizonAnalyzer()
        self.action = ActionAnalyzer()

    def analyze(self, ticker: str, retrieved_context: Dict[str, str]) -> Dict[str, Any]:
        """
        Runs the full ALPHA analysis based on retrieved context.
        """
        
        def safe_analyze(analyzer_func, *args):
            # Check if any text argument is meaningful (len > 50)
            if not any(len(str(arg)) > 50 for arg in args):
                return {
                    "score": 50, 
                    "rationale": "Insufficient data retrieved for this dimension.", 
                    "analysis": "Data missing."
                }
            try:
                return analyzer_func(*args)
            except Exception as e:
                print(f"Analysis Error: {e}")
                return {
                    "score": 50, 
                    "rationale": f"Error during analysis: {e}", 
                    "analysis": "Error."
                }

        # Run dimensions
        a_res = safe_analyze(self.alignment.analyze, retrieved_context.get("mda", ""), retrieved_context.get("insider", ""))
        l_res = safe_analyze(self.liquidity.analyze, retrieved_context.get("risk", ""), retrieved_context.get("mda", ""))
        p_res = safe_analyze(self.performance.analyze, retrieved_context.get("financials", ""))
        h_res = safe_analyze(self.horizon.analyze, retrieved_context.get("business", ""), retrieved_context.get("risk", ""))
        act_res = safe_analyze(self.action.analyze, retrieved_context.get("market", ""), retrieved_context.get("news", ""))
        
        # Scoring
        scores = [a_res["score"], l_res["score"], p_res["score"], h_res["score"], act_res["score"]]
        
        # Avoid division by zero if all failed (though safe_analyze returns 50)
        composite_score = sum(scores) / len(scores) if scores else 50
        
        # Weighted Decision Matrix
        quality_score = (a_res["score"] + h_res["score"]) / 2
        health_score = p_res["score"]
        entry_score = (l_res["score"] + act_res["score"]) / 2
        
        weighted_score = (quality_score * 0.4) + (health_score * 0.3) + (entry_score * 0.3)
        
        verdict = "HOLD"
        if weighted_score > 75:
            verdict = "STRONG BUY"
        elif weighted_score > 60:
            verdict = "ACCUMULATE"
        elif weighted_score < 50:
            verdict = "AVOID"
            
        return {
            "ticker": ticker,
            "composite_score": int(composite_score),
            "weighted_score": int(weighted_score),
            "verdict": verdict,
            "dimensions": {
                "Alignment": a_res,
                "Liquidity": l_res,
                "Performance": p_res,
                "Horizon": h_res,
                "Action": act_res
            }
        }
