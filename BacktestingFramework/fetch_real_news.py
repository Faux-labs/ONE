import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import csv
import os

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def fetch_and_process_news(ticker="AAPL", filename="real_news.csv"):
    print(f"Fetching news for {ticker} via yfinance...")
    t = yf.Ticker(ticker)
    news_items = t.news
    
    if not news_items:
        print("No news found via yfinance.")
        return

    sia = SentimentIntensityAnalyzer()
    
    records = []
    print(f"Processing {len(news_items)} articles...")
    
    for item in news_items:
        # yfinance news structure varies, handling common fields
        content = item.get('content', {})
        title = content.get('title', item.get('title', ''))
        summary = content.get('summary', item.get('summary', ''))
        pubDate = content.get('pubDate', item.get('pubDate'))
        
        # Combined text for sentiment
        full_text = f"{title}. {summary}"
        
        # Analyze Sentiment
        scores = sia.polarity_scores(full_text)
        compound = scores['compound']
        
        # Map to our engine's format: Sentiment (-1 or 1), Confidence (0.5 to 1.0)
        # VADER compound is -1 to 1. 
        # If compound > 0.05 => Bullish (1), < -0.05 => Bearish (-1), else Neutral (0)
        
        if compound > 0.05:
            sentiment = 1
            confidence = 0.5 + (compound / 2) # Scale to 0.5 - 1.0 roughly
        elif compound < -0.05:
            sentiment = -1
            confidence = 0.5 + (abs(compound) / 2)
        else:
            sentiment = 0 # Neutral, maybe ignore in backtest?
            confidence = 0.5

        if pubDate:
            # Clean timestamp to standard format matchable with engine
            try:
                # Example: 2026-02-01T18:47:00Z -> 2026-02-01
                ts = pd.to_datetime(pubDate).strftime('%Y-%m-%d')
                
                records.append({
                    "timestamp": ts,
                    "headline": title,
                    "sentiment": sentiment,
                    "ai_confidence": round(confidence, 2)
                })
            except Exception as e:
                print(f"Date parsing error: {e}")

    if records:
        df = pd.DataFrame(records)
        # Average multiple news per day if any
        # For simplicity, just saving all. The Backtester might need to handle duplicates or join effectively.
        # But 'backtest_engine.py' does: price_data.join(news_data, how='left')
        # If news_data has duplicates on index, join might explode rows.
        # Let's aggregate by day: take the mean sentiment or max confidence.
        
        # Group by date to match price data daily index
        df_grouped = df.groupby('timestamp').agg({
            'sentiment': lambda x: 1 if x.mean() > 0 else (-1 if x.mean() < 0 else 0),
            'ai_confidence': 'max',
            'headline': 'first' # Just keep one
        }).reset_index()
        
        # Backfill with synthetic news for demonstration if not enough overlap
        # (Since yfinance only gives recent news)
        start_date = pd.to_datetime("today") - pd.Timedelta(days=30)
        dates = pd.date_range(start=start_date, periods=20, freq='B') # Business days
        
        import random
        for d in dates:
            ts_str = d.strftime('%Y-%m-%d')
            if ts_str not in df_grouped['timestamp'].values:
                # Add a random news event
                sentiment = random.choice([1, -1])
                new_row = {
                    'timestamp': ts_str,
                    'headline': f"Historical News for {ts_str} (Backfilled)",
                    'sentiment': sentiment,
                    'ai_confidence': round(random.uniform(0.75, 0.95), 2)
                }
                # Use pd.concat instead of append
                df_grouped = pd.concat([df_grouped, pd.DataFrame([new_row])], ignore_index=True)
        
        # Sort by date
        df_grouped = df_grouped.sort_values('timestamp')

        df_grouped.to_csv(filename, index=False)
        print(f"Saved {len(df_grouped)} news days to {filename} (including backfilled data)")
        print("Sample:")
        print(df_grouped.head())
    else:
        print("No valid news records processed.")

if __name__ == "__main__":
    fetch_and_process_news()
