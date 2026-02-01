import feedparser
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from groq import Groq
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# 1. Load the variables from .env into the environment
load_dotenv()

# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. COMPONENT: FETCHER
# ---------------------------------------------------------
class NewsFetcher:
    def fetch_rss(self, rss_url: str) -> List[Dict]:
        """Fetches raw entries from an RSS feed."""
        print(f"📡 Fetching RSS: {rss_url}")
        feed = feedparser.parse(rss_url)
        
        articles = []
        for entry in feed.entries[:5]: # Limit to 5 for testing
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get('summary', '')
            })
        print("FETCH RSS:- \n ", articles)
        return articles

    def scrape_article_content(self, url: str) -> str:
        """(Optional) Scrapes full text if RSS only gives summaries."""
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            # varied by site, but generally looking for <p> tags
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            return text[:2000] # Truncate to save tokens
        except Exception as e:
            return ""

# 3. COMPONENT: PREPROCESSOR (NLTK)
# ---------------------------------------------------------
class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """
        Tokenizes and removes stopwords to reduce token usage 
        before sending to OpenAI.
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove non-alphanumeric and stopwords
        clean_tokens = [
            word for word in tokens 
            if word.isalnum() and word.lower() not in self.stop_words
        ]
        print("CLEANED TOKENS:- \n","".join(clean_tokens))
        return " ".join(clean_tokens)

# 4. COMPONENT: ANALYZER (OpenAI)
# ---------------------------------------------------------
class AIAnalyzer:
    def analyze_signal(self, text: str) -> Dict[str, Any]:
        """
        Uses OpenAI to extract structured data (Sentiment, Entities, Confidence).
        """
        
        system_prompt = """
        You are a crypto trading analyst. Analyze the provided news text.
        Return a strict JSON object with these fields:
        - "sentiment": "BULLISH", "BEARISH", or "NEUTRAL"
        - "confidence_score": integer between 0-100 indicating strength of signal.
        - "affected_assets": list of strings (e.g., ["BTC", "ETH", "ONE"]).
        - "reasoning": short summary of why.
        """

        try:
            response = client.chat.completions.create(
                model="qwen/qwen3-32b", # Use gpt-4-turbo for better reasoning
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"News Content: {text}"}
                ],
                temperature=0,
                response_format={ "type": "json_object" } # Enforce JSON
            )
            print("MESSAGE FROM GROQ:- \n", json.loads(response.choices[0].message.content))
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ AI Analysis Failed: {e}")
            return None

# 5. ORCHESTRATOR (The Pipeline)
# ---------------------------------------------------------
def run_pipeline():
    # A. Init Modules
    fetcher = NewsFetcher()
    preprocessor = TextPreprocessor()
    analyzer = AIAnalyzer()

    # B. Define Source (CoinDesk RSS for example)
    rss_source = "https://www.coindesk.com/arc/outboundfeeds/rss/"

    # C. Execute Flow
    raw_articles = fetcher.fetch_rss(rss_source)

    print(f"\nProcessing {len(raw_articles)} articles...\n")

    for article in raw_articles:
        print(f"📰 Analyzing: {article['title']}")
        
        # 1. Combine title + summary for context
        full_text = f"{article['title']} {article['summary']}"
        
        # 2. Preprocess (Clean Noise)
        # Note: We keep this light because LLMs actually benefit from some grammatical context,
        # but removing heavy stopwords can save cost on very large inputs.
        clean_text = preprocessor.clean_text(full_text)
        
        # 3. AI Analysis
        result = analyzer.analyze_signal(clean_text)
        
        if result:
            # 4. Output Logic
            print(f"   👉 Sentiment: {result.get('sentiment')} ({result.get('confidence_score')}%)")
            print(f"   👉 Assets: {result.get('affected_assets')}")
            print(f"   👉 Reason: {result.get('reasoning')}\n")
            
            # HERE: You would add logic to call your Move Smart Contract
            # if result['confidence_score'] > 80:
            #     execute_on_chain(...)

if __name__ == "__main__":
    run_pipeline()