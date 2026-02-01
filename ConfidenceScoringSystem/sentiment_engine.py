import os
import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- 1. Define Data Models (The "Shape" of the Answer) ---
class SentimentType(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class TimeFrame(str, Enum):
    IMMEDIATE = "IMMEDIATE"   # < 24 hours
    SHORT_TERM = "SHORT_TERM" # 1-7 days
    LONG_TERM = "LONG_TERM"   # > 1 week

class MarketAnalysis(BaseModel):
    sentiment: SentimentType = Field(..., description="Direction of the market impact")
    impact_score: int = Field(..., description="Magnitude of impact from 1 (Low) to 10 (High)")
    time_sensitivity: TimeFrame = Field(..., description="How quickly the market will react")
    reasoning: str = Field(..., description="Brief explanation of the rating")

# --- 2. The Scoring Logic (Python Side) ---
def calculate_composite_score(analysis: MarketAnalysis) -> float:
    """
    Calculates a 0-100% confidence score based on Impact and Time.
    """
    # Base score derived from Impact (1-10 mapped to 10-100)
    base_score = analysis.impact_score * 10
    
    # Apply Time Decay (We want IMMEDIATE action for prediction markets)
    time_multipliers = {
        TimeFrame.IMMEDIATE: 1.0,   # 100% of the impact score
        TimeFrame.SHORT_TERM: 0.85, # 85% confidence if it's slower
        TimeFrame.LONG_TERM: 0.50   # 50% penalty for long-term vague news
    }
    
    multiplier = time_multipliers[analysis.time_sensitivity]
    
    final_score = base_score * multiplier
    
    # Cap at 100 just in case
    print("Calculated Score:- \n", min(round(final_score, 1), 100.0))
    return min(round(final_score, 1), 100.0)

# --- 3. The Analyzer Function ---
def analyze_news(news_text: str) -> dict:
    print(f"🧠 Analyzing: {news_text[:50]}...")
    
    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905", # Using a strong Groq model
            messages=[
                {
                    "role": "system", 
                    "content": """You are a senior crypto quant researcher. 
                                 Analyze the news for immediate price impact on OneChain (or general crypto). 
                                 Be conservative.
                                 Output JSON only using the schema provided."""
                },
                {"role": "user", "content": news_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "market_analysis",
                    "schema": MarketAnalysis.model_json_schema()
                }
            }
        )

        # Parse the JSON response
        raw_result = json.loads(response.choices[0].message.content or "{}")
        
        # Validate with Pydantic
        analysis = MarketAnalysis.model_validate(raw_result)
        
        print("ANALYSIS OBTAINED FROM:- \n", analysis)
        
        # Calculate our custom score
        confidence = calculate_composite_score(analysis)
        print("CONFIDENCE:- \n", confidence)
        
        return {
            "status": "success",
            "sentiment": analysis.sentiment,
            "impact": analysis.impact_score,
            "timing": analysis.time_sensitivity,
            "confidence_score": confidence, # <--- The Magic Number (0-100)
            "reasoning": analysis.reasoning
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 4. Test Run ---
if __name__ == "__main__":
    # Example 1: High Impact
    news_1 = "OneChain announces official partnership with Visa for instant payments."
    result_1 = analyze_news(news_1)
    
    # Example 2: Low Impact / Vague
    news_2 = "OneChain CEO says blockchain technology is the future in a podcast interview."
    result_2 = analyze_news(news_2)

    print("\n--- RESULTS ---")
    
    # Check if result_1 is valid before accessing keys
    if result_1.get("status") == "success":
        print(f"NEWS 1 Score: {result_1['confidence_score']}% ({result_1['sentiment']})")
        print(f"Reason: {result_1['reasoning']}")
    else:
        print(f"NEWS 1 Failed: {result_1.get('message')}")
        
    print("-" * 30)
    
    # Check if result_2 is valid before accessing keys
    if result_2.get("status") == "success":
        print(f"NEWS 2 Score: {result_2['confidence_score']}% ({result_2['sentiment']})")
        print(f"Reason: {result_2['reasoning']}")
    else:
        print(f"NEWS 2 Failed: {result_2.get('message')}")