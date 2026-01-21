import argparse
import sys
import os
import logging

# Silence httpx logger
logging.getLogger("httpx").setLevel(logging.WARNING)

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.tools.sec_edgar import sec_handler
from src.analysis.alpha_engine import AlphaEngine
from src.utils.memory import RedisMemory
from src.retrieval.ingestion import IngestionEngine
from src.retrieval.vector_db import QdrantVectorDB

def main():
    parser = argparse.ArgumentParser(description="Financial Research Agent - ALPHA Framework (Production)")
    parser.add_argument("--query", type=str, help="Investment query/question")
    parser.add_argument("--ticker", type=str, help="Company Ticker (e.g., NVDA)")
    parser.add_argument("--ingest", action="store_true", help="Ingest data for the ticker")
    parser.add_argument("--file", type=str, help="Local PDF file to ingest")
    parser.add_argument("--year", type=str, default="Latest", help="Year of the filing (e.g. 2024)")
    parser.add_argument("--type", type=str, default="10-K", help="Form type (10-K, 10-Q, 8-K, 4)")
    
    args = parser.parse_args()
    
    memory = RedisMemory()
    alpha_engine = AlphaEngine()
    
    # Init retriever (Now just QdrantVectorDB)
    # We instantiate it once. 
    vector_db = QdrantVectorDB()
    
    if args.ingest and args.ticker:
        print(f"Ingesting data for {args.ticker} (Year: {args.year}, Type: {args.type})...")
        ingestion = IngestionEngine()
        # vector_db already inited above
        
        raw_content = None
        
        if args.file:
            # Local File Ingestion (Multimodal)
            from src.retrieval.file_loader import file_loader
            print(f"Processing local file: {args.file}...")
            # This returns a LIST of dicts
            raw_content = file_loader.process_file(args.file)
        else:
            # Cloud Fetch
            # 1. Fetch Real Data
            print(f"Fetching {args.type} for {args.ticker}...")
            raw_content = sec_handler.fetch_latest_filing(args.ticker, form_type=args.type)
            if "Error" in raw_content:
                print(raw_content)
                return

        # 2. Chunk
        print("Chunking document...")
        # process_document handles list or str
        # We pass args.type as the doc_type logic
        chunks = ingestion.process_document(args.ticker, args.type, raw_content, year=args.year)
        print(f"Created {len(chunks)} chunks.")
        
        # 3. Embed & Index
        vector_db.add_documents(chunks)
        print("Ingestion complete.")
        return

    if args.query:
        print(f"Processing Query: {args.query}")
        memory.add_message("user", args.query)
        
        ticker = args.ticker if args.ticker else "NVDA" 
        
        # 2. Retrieve Context (Real)
        # We search based on the query.
        print("Retrieving context...")
        # New QdrantVectorDB.search takes a string query directly
        general_results = vector_db.search(args.query, limit=10)
        
        print(f"  [DEBUG] General Search Results found: {len(general_results)}")
        for i, r in enumerate(general_results):
            print(f"    - {i+1} [Score: {r['score']:.4f}] {r['text'][:100].replace(chr(10), ' ')}...")

        general_context = "\n".join([r['text'] for r in general_results])
        
        doc_context = {}
        
        dimensions_queries = {
            "mda": f"{ticker} Item 7 Management's Discussion and Analysis of Financial Condition and Results of Operations MD&A",
            "risk": f"{ticker} Item 1A Risk Factors market risks regulatory challenges competition",
            "financials": f"{ticker} Item 8 Financial Statements consolidated balance sheets income statement cash flows notes",
            "business": f"{ticker} Item 1 Business overview strategy products segments competition",
            "insider": f"{ticker} Item 12 Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
            "market": f"{ticker} Item 5 Market for Registrant’s Common Equity Related Stockholder Matters and Issuer Purchases of Equity Securities"
        }
        
        for key, q in dimensions_queries.items():
            # Simply search with the query string
             res = vector_db.search(q, limit=3)
             print(f"  [DEBUG] Dimension '{key}' Search Results found: {len(res)}")
             for i, r in enumerate(res):
                 print(f"    - {i+1} [Score: {r['score']:.4f}] {r['text'][:100].replace(chr(10), ' ')}...")
                 
             text = "\n".join([r['text'] for r in res])
             # Enrich dimension context with relevant general context
             doc_context[key] = text + "\n---\n" + general_context
             
        # Gap Bridging: Perform targeted efficient web searches for missing data
        print("Bridging data gaps with Web Search...")
        from src.tools.web_search import web_search_tool
        
        bridge_queries = {
            "financials": f"{ticker} revenue growth last 5 years operating margin vs peers",
            "risk": f"{ticker} current macroeconomic headwinds interest rate sensitivity",
            "market": f"{ticker} analyst ratings price targets"
        }
        
        for key, q in bridge_queries.items():
            try:
                web_results = web_search_tool.search(q)
                web_text = "\n".join([f"{w['title']}: {w['snippet']}" for w in web_results])
                if key in doc_context:
                    doc_context[key] += f"\n\n[External Web Data]:\n{web_text}"
            except Exception as e:
                print(f"Bridge search failed for {key}: {e}")
             
        # Fallback to Web Search for 'market' and 'news'
        print("Fetching live market data (Web Search)...")
        web_res = web_search_tool.search(f"{ticker} stock price valuation news")
        web_text = "\n".join([f"{w['title']}: {w['snippet']}" for w in web_res])
        doc_context["market"] += ("\n" + web_text)
        doc_context["news"] = web_text 
        
        # 3. Analyze
        print("Analyzing dimensions...")
        result = alpha_engine.analyze(ticker, doc_context)
        
        # 4. Generate Output
        print("\n" + "="*50)
        print(f"**ALPHA Composite Score: {result['composite_score']}/100**")
        print("="*50)
        
        print("\n**Timing Assessment:** " + result['verdict'])
        print(f"Weighted Score: {result['weighted_score']}")
        
        for dim, data in result['dimensions'].items():
            print(f"\n{dim[0]} - {dim} ({data['score']}/100)")
            # Fix: Ensure 'analysis' key exists or use fallback
            analysis_text = data.get('analysis', data.get('details', 'No analysis provided.'))
            print(f"{analysis_text[:200]}...") 
            print(f"Rationale: {data.get('rationale', 'No rationale provided.')}")
            
        memory.add_message("assistant", str(result))
        
        # 5. Save Report to File
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        filename = f"{reports_dir}/{ticker}_{timestamp}.md"
        
        md_content = f"# ALPHA Investment Report: {ticker}\n"
        md_content += f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        md_content += f"## Verdict: {result['verdict']}\n"
        md_content += f"**Composite Score:** {result['composite_score']}/100\n"
        md_content += f"**Weighted Score:** {result['weighted_score']}\n\n"
        md_content += "---\n\n"
        
        for dim, data in result['dimensions'].items():
            analysis_text = data.get('analysis', data.get('details', 'No analysis provided.'))
            md_content += f"### {dim} ({data['score']}/100)\n"
            md_content += f"{analysis_text}\n\n"
            md_content += f"**Rationale:** {data.get('rationale', 'N/A')}\n\n"
            md_content += "---\n"
            
        with open(filename, "w") as f:
            f.write(md_content)
            
        print(f"\n[SUCCESS] Report saved to: {filename}")

if __name__ == "__main__":
    main()
