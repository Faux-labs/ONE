import os
import json
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ImpactDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"

class FinancialAssessment(BaseModel):
    # 1. Anti-Hallucination: Force it to quote the text
    key_claims: List[str] = Field(..., description="Extract 1-3 factual claims from the text. Do not infer.")
    
    # 2. Reasoning: Force it to think before scoring
    analysis_logic: str = Field(..., description="Step-by-step reasoning linking claims to market impact.")
    
    # 3. The Data Points
    affected_assets: List[str] = Field(..., description="Ticker symbols (e.g. BTC, ETH, ONE).")
    direction: ImpactDirection = Field(..., description="Market direction.")
    impact_score: int = Field(..., description="Magnitude from 1 (Low) to 10 (High).")
    confidence: int = Field(..., description="0-100% confidence in this assessment.")
    
    # 4. Context Check
    market_fit: str = Field(..., description="How does this news interact with current market trend? (e.g. 'Contrarian', 'Fuel to fire')")

SYSTEM_PROMPT = """
You are a Senior Quantitative Risk Analyst for a crypto hedge fund. 
Your goal is to evaluate news headlines for **IMMEDIATE** price impact on specific assets.

### YOUR PRIME DIRECTIVES:
1. **Be Conservative:** False positives cause money loss. If news is vague, mark it NEUTRAL.
2. **No Hallucinations:** Only use facts present in the text. If a detail is missing, do not invent it.
3. **Context Matters:** A "Bullish" headline in a "Crashing" market often has zero impact. Adjust scores accordingly.
4. **Structured Output:** You must return valid JSON matching the schema provided.

### SCORING GUIDE:
- Impact 1-3: Minor rumors, opinion pieces, or already priced-in info.
- Impact 4-7: Official partnership announcements, regulatory approvals, mainnet launches.
- Impact 8-10: Major hacks, SEC lawsuits, or central bank policy shifts.

### CONFIDENCE CALCULATION:
- Reduce confidence if: Source is unverified, wording is ambiguous ("rumored", "sources say"), or timeline is unclear.
"""

USER_PROMPT_TEMPLATE = """
### CURRENT MARKET CONTEXT:
The market is currently: {market_state} 
(Volatility Index: {volatility_score}/100)

### NEWS TEXT TO ANALYZE:
"{news_text}"

### INSTRUCTION:
Analyze the text above. 
1. First, extract the facts.
2. Second, compare it against the Market Context (e.g., does good news matter if the market is fearful?).
3. Finally, assign sentiment and confidence scores.
"""

def analyze_crypto_news(news_text: str, market_state: str = "Neutral/Chop", volatility: int = 20):
    """
    Analyzes news with context awareness.
    """
    
    # 1. Fill the template
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(
        market_state=market_state,
        volatility_score=volatility,
        news_text=news_text
    )

    try:
        # 2. Call OpenAI with "Structured Outputs"
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905", # Recommended for complex reasoning
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "market_analysis",
                    "schema": FinancialAssessment.model_json_schema()
                }
            }, 
            temperature=0.1, # Low temperature = More deterministic/factual
        )

        # 3. Parse manually since standard Groq client doesn't have .parsed
        raw_result = json.loads(response.choices[0].message.content or "{}")
        result = FinancialAssessment.model_validate(raw_result)
        
        return result

    except Exception as e:
        print(f"❌ Analysis Failed: {e}")
        return None

# --- Usage Example ---
if __name__ == "__main__":
    
    # Scenario A: Good news in a BAD market
    # (A good analyst knows this usually results in a "sell the news" event)
    news = "OneChain releases roadmap for Q3, promising faster transaction speeds."
    context = "Extreme Fear. Bitcoin has dropped 15% this week."
    
    result = analyze_crypto_news(news, market_state=context, volatility=85)
    
    if result:
        print(f"\n📰 Analysis Report for: '{news[:30]}...'")
        print(f"------------------------------------------------")
        print(f"Sentiment:  {result.direction.value}")
        print(f"Impact:     {result.impact_score}/10")
        print(f"Confidence: {result.confidence}%")
        print(f"Logic:      {result.analysis_logic}")
        print(f"Market Fit: {result.market_fit}")
        print(f"Key Claims: {result.key_claims}")