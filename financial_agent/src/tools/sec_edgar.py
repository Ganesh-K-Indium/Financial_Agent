import requests
from sec_edgar_api import EdgarClient
from src.config import config
import re
from bs4 import BeautifulSoup
import time

class SECHandler:
    def __init__(self):
        self.user_agent = config.SEC_USER_AGENT
        # EdgarClient for metadata queries
        self.client = EdgarClient(user_agent=self.user_agent)
        self.headers = {
            "User-Agent": self.user_agent
        }

    def get_cik(self, ticker: str) -> str:
        """
        Maps ticker to CIK. 
        In production, cache this mapping.
        For now, query SEC's company tickers JSON.
        """
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers)
            data = resp.json()
            
            for entry in data.values():
                if entry["ticker"] == ticker.upper():
                    # CIK must be 10 digits zero-padded
                    return str(entry["cik_str"]).zfill(10)
            raise ValueError(f"Ticker {ticker} not found.")
        except Exception as e:
            print(f"Error fetching CIK: {e}")
            return ""

    def fetch_latest_filing(self, ticker: str, form_type: str = "10-K") -> str:
        """
        Fetches the text content of the latest filing of a specific type.
        """
        param_cik = self.get_cik(ticker)
        if not param_cik:
            return "Error: CIK not found."
            
        # Get submissions
        try:
            # We enforce a small rate limit here
            time.sleep(0.2)
            submissions = self.client.get_submissions(cik=param_cik)
            
            # Filter for Form
            # Recent filings are in 'filings' -> 'recent'
            recent = submissions["filings"]["recent"]
            
            accession_number = None
            primary_document = None
            
            print(f"Searching for latest {form_type}...")
            
            for i, form in enumerate(recent["form"]):
                # Simple exact match for form type (e.g. '10-K', '8-K', '4')
                if form == form_type:
                    accession_number = recent["accessionNumber"][i]
                    primary_document = recent["primaryDocument"][i]
                    break
            
            if not accession_number:
                return f"Error: No {form_type} found in recent submissions for {ticker}."
                
            # Construct URL
            # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}
            accession_nodash = accession_number.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{param_cik}/{accession_nodash}/{primary_document}"
            
            print(f"Fetching {form_type} from: {url}")
            resp = requests.get(url, headers=self.headers)
            
            # Simple text extraction
            # In production, use a robust parser like 'edgar-tools' or custom XBRL/HTML parser
            text = self.clean_html(resp.text)
            return text
            
        except Exception as e:
            return f"Error fetching filing: {e}"

    def clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()    
        
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

sec_handler = SECHandler()
