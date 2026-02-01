import hashlib
import json
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from groq import Groq
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
# --- 1. SETUP ---
# Download necessary NLTK data (run once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

class HybridAnalyzer:
    def __init__(self):
        # Tools
        self.sia = SentimentIntensityAnalyzer() # The "Dumb" Backup
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Memory (In-Memory Cache for Hackathon, use Redis for Prod)
        self.cache: Dict[str, dict] = {} 
        
        # Config
        self.KEYWORDS = {"bitcoin", "btc", "eth", "ethereum", "one", "onechain", "crypto", "defi"}

    # --- 2. THE GATEKEEPER (Cost Saver) ---
    def _is_relevant(self, text: str) -> bool:
        """
        Uses NLTK tokenization to check if the text is worth paying for.
        """
        tokens = word_tokenize(text.lower())
        # Check if any keyword exists in tokens
        return not self.KEYWORDS.isdisjoint(set(tokens))

    # --- 3. THE BACKUP (Resilience) ---
    def _fallback_analysis(self, text: str) -> dict:
        """
        Uses NLTK VADER for free, offline sentiment analysis.
        """
        print("⚠️ Groq Unavailable. Switching to NLTK Fallback...")
        scores = self.sia.polarity_scores(text)
        
        # Convert VADER (-1 to 1) to our format
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "BULLISH"
            confidence = int(compound * 100)
        elif compound <= -0.05:
            sentiment = "BEARISH"
            confidence = int(abs(compound) * 100)
        else:
            sentiment = "NEUTRAL"
            confidence = 50

        return {
            "source": "NLTK_FALLBACK",
            "sentiment": sentiment,
            "confidence": confidence,
            "impact_score": 5, # Default generic impact
            "reasoning": "VADER keyword analysis (API limit reached)"
        }

    # --- 4. THE BRAIN (OpenAI) ---
    def _groq_analysis(self, text: str) -> dict:
        """
        The expensive, high-quality analysis.
        """
        try:
            # Short prompt to save tokens
            response = self.client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905", # Cheaper than GPT-4
                messages=[
                    {"role": "system", "content": "Return JSON: {sentiment: 'BULLISH'|'BEARISH'|'NEUTRAL', confidence: 0-100, impact: 1-10}"},
                    {"role": "user", "content": text}
                ],
                response_format={ "type": "json_object" },
                temperature=0
            )
            data = json.loads(response.choices[0].message.content)
            data["source"] = "GROQ"
            return data
            
        except Exception as e:
            print(f"❌ Groq Error: {e}")
            return None # Trigger fallback

    # --- 5. THE ORCHESTRATOR (Main Function) ---
    def analyze(self, text: str) -> dict:
        # A. Check Cache (Speed + Cost)
        # Create a hash of the text to use as a key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            print("⚡ Cache Hit!")
            return self.cache[text_hash]

        # B. Check Relevance (Cost)
        if not self._is_relevant(text):
            return {"status": "IGNORED", "reason": "Irrelevant text"}

        # C. Try OpenAI (Quality)
        result = self._groq_analysis(text)

        # D. Trigger Fallback if OpenAI failed (Reliability)
        if result is None:
            result = self._fallback_analysis(text)

        # E. Save to Cache
        self.cache[text_hash] = result
        return result

# --- Usage Example ---
if __name__ == "__main__":
    bot = HybridAnalyzer()

    # 1. Test Relevance Filter (Should be ignored)
    print(bot.analyze("The weather in London is rainy today.")) 
    
    # 2. Test Groq (Should work)
    print(bot.analyze("OneChain partners with major bank, token price expected to surge."))
    
    # 3. Test Cache (Should be instant)
    print(bot.analyze("OneChain partners with major bank, token price expected to surge."))
    
    # 4. Simulate API Failure
    bot.client.api_key = "invalid-key" 
    print(bot.analyze("Bitcoin crashes below 50k as regulations tighten."))