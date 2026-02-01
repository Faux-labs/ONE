import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.calibration import calibration_curve

# ==========================================
# 1. DATA LAYER (Engine)
# ==========================================
class DataEngine:
    def __init__(self, db_name="backtest_data.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # Price Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                timestamp DATETIME,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        self.conn.commit()

    def fetch_crypto_data(self, symbol="BTC/USDT", timeframe='1d', limit=365):
        """Fetches from CCXT (Binance) and saves to SQLite"""
        print(f"⬇️ Downloading Crypto: {symbol}...")
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        data = []
        for row in ohlcv:
            # Convert ms timestamp to datetime
            dt = datetime.fromtimestamp(row[0] / 1000).isoformat()
            data.append((symbol, dt, row[1], row[2], row[3], row[4], row[5]))

        self._save_to_db(data)
        return self._load_from_db(symbol)

    def fetch_stock_data(self, ticker="AAPL", period="1y"):
        """Fetches from Yahoo Finance and saves to SQLite"""
        print(f"⬇️ Downloading Stock: {ticker}...")
        df = yf.download(ticker, period=period, progress=False)
        df.reset_index(inplace=True)
        
        data = []
        for _, row in df.iterrows():
            dt = row['Date'].isoformat() if isinstance(row['Date'], pd.Timestamp) else row['Date']
            # Handle yfinance multi-index columns if present, usually it's simple
            data.append((ticker, dt, float(row['Open']), float(row['High']), float(row['Low']), float(row['Close']), float(row['Volume'])))

        self._save_to_db(data)
        return self._load_from_db(ticker)

    def _save_to_db(self, data):
        cursor = self.conn.cursor()
        cursor.executemany('INSERT OR IGNORE INTO prices VALUES (?,?,?,?,?,?,?)', data)
        self.conn.commit()

    def _load_from_db(self, symbol):
        df = pd.read_sql(f"SELECT * FROM prices WHERE symbol='{symbol}' ORDER BY timestamp", self.conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    def generate_mock_news(self, price_data):
        """
        Simulates news events for testing.
        In production, replace this with a real News API fetcher.
        """
        news_events = []
        for ts, row in price_data.iterrows():
            # 10% chance of news happening on any given day
            if random.random() < 0.1: 
                # Create a "fake" confidence score (0.5 to 1.0)
                confidence = round(random.uniform(0.55, 0.99), 2)
                
                # Mock Logic: If price went up next day, assume "Good News" for testing
                # (In reality, your AI would predict this based on text)
                news_events.append({
                    "timestamp": ts,
                    "headline": "Sample Bullish/Bearish Event",
                    "sentiment": random.choice([-1, 1]), # -1 Sell, 1 Buy
                    "ai_confidence": confidence 
                })
        
        return pd.DataFrame(news_events).set_index("timestamp")

# ==========================================
# 2. SIMULATION ENGINE (Backtester)
# ==========================================
class Backtester:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.holdings = 0
        self.equity_curve = []
        self.trade_log = []
        self.prediction_log = [] # For calibration (confidence vs outcome)

    def run(self, price_data, news_data):
        """
        The Event Loop: Iterates through time, combining price and news.
        """
        print("🚀 Starting Simulation...")
        
        # Merge Price and News (Left Join)
        full_timeline = price_data.join(news_data, how='left')
        
        for ts, row in full_timeline.iterrows():
            current_price = row['close']
            
            # --- A. AI LOGIC (Simulated) ---
            if pd.notna(row['sentiment']):
                signal = row['sentiment']      # 1 or -1
                confidence = row['ai_confidence'] # 0.0 - 1.0
                
                # Log prediction for calibration later
                self._log_prediction(ts, signal, confidence, price_data)

                # --- B. EXECUTION LOGIC ---
                # Strategy: Only trade if confidence > 70%
                if confidence > 0.70:
                    if signal == 1 and self.capital > 0:
                        # BUY ALL
                        self.holdings = self.capital / current_price
                        self.capital = 0
                        self.trade_log.append({'date': ts, 'action': 'BUY', 'price': current_price, 'conf': confidence})
                    
                    elif signal == -1 and self.holdings > 0:
                        # SELL ALL
                        self.capital = self.holdings * current_price
                        self.holdings = 0
                        self.trade_log.append({'date': ts, 'action': 'SELL', 'price': current_price, 'conf': confidence})

            # --- C. UPDATE EQUITY ---
            current_val = self.capital + (self.holdings * current_price)
            self.equity_curve.append({'date': ts, 'equity': current_val})

        # Final Sell to liquidate
        if self.holdings > 0:
            final_price = price_data.iloc[-1]['close']
            self.capital = self.holdings * final_price
        
        return pd.DataFrame(self.equity_curve).set_index('date')

    def _log_prediction(self, timestamp, signal, confidence, price_data):
        """
        Checks if the prediction came true (Lookahead 5 days)
        """
        try:
            current_idx = price_data.index.get_loc(timestamp)
            future_price = price_data.iloc[current_idx + 5]['close'] # 5 days later
            current_price = price_data.iloc[current_idx]['close']
            
            # Did price move in predicted direction?
            actual_move = 1 if future_price > current_price else -1
            is_correct = (signal == actual_move)
            
            self.prediction_log.append({
                'confidence': confidence,
                'is_correct': int(is_correct)
            })
        except IndexError:
            pass # End of data

# ==========================================
# 3. PERFORMANCE METRICS
# ==========================================
class PerformanceAnalyst:
    @staticmethod
    def calculate_metrics(equity_df, initial_capital=10000):
        equity = equity_df['equity']
        
        # 1. Total Return
        final_value = equity.iloc[-1]
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # 2. Sharpe Ratio
        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            sharpe = 0
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # 3. Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        return {
            "Final Equity": round(final_value, 2),
            "Total Return": f"{total_return:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2f}%"
        }

    @staticmethod
    def plot_calibration_curve(prediction_log):
        """
        Plots: "When AI says 90% confidence, is it actually right 90% of the time?"
        """
        if not prediction_log:
            print("No predictions to calibrate.")
            return

        df = pd.DataFrame(prediction_log)
        
        # Use Scikit-Learn to bin data
        prob_true, prob_pred = calibration_curve(df['is_correct'], df['confidence'], n_bins=5)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='AI Performance')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('AI Predicted Confidence')
        plt.ylabel('Actual Win Rate')
        plt.title('Confidence Calibration Curve')
        plt.legend()
        plt.grid()
        plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup Data
    engine = DataEngine()
    
    # Choose: Crypto or Stock
    price_data = engine.fetch_crypto_data("BTC/USDT", limit=500)
    # price_data = engine.fetch_stock_data("AAPL", period="2y")
    
    # Generate Fake News (replace this with real news DB later)
    news_data = engine.generate_mock_news(price_data)

    # 2. Run Backtest
    bot = Backtester(initial_capital=10000)
    equity_df = bot.run(price_data, news_data)

    # 3. Analyze Results
    metrics = PerformanceAnalyst.calculate_metrics(equity_df)
    print("\n📊 PERFORMANCE REPORT")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 4. Plotting
    # Plot Equity Curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df, label='Portfolio Value')
    plt.title('Backtest Equity Curve')
    plt.legend()
    plt.show()

    # Plot Calibration (The Hackathon Winner Feature)
    PerformanceAnalyst.plot_calibration_curve(bot.prediction_log)