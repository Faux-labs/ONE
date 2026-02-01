import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class RiskConfig:
    # 1. Capital Preservation
    MAX_DAILY_LOSS: float = 50.0   # Stop trading if we lose 50 ONE
    MAX_BETS_PER_DAY: int = 10     # Hard cap on activity
    
    # 2. Position Sizing
    BASE_BET_SIZE: float = 5.0     # Standard bet (in ONE)
    MAX_BET_SIZE: float = 20.0     # Never bet more than this
    
    # 3. Volatility Logic
    BASE_CONFIDENCE: float = 75.0  # Minimum confidence needed in calm markets
    VOLATILITY_PENALTY: float = 0.5 # How much to increase threshold per volatility unit
    
    # 4. Timing
    COOLDOWN_MINUTES: int = 15     # Wait time after any trade

class DecisionEngine:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        
        # State Tracking
        self.daily_pnl = 0.0
        self.daily_bets_count = 0
        self.last_trade_time = datetime.min
        self.last_reset_time = datetime.now()
        
        # Market Data Memory (for volatility calc)
        # Stores last 20 prices to calculate StdDev
        self.price_history = [] 

    # --- 1. Helper: Volatility Calculation ---
    def update_market_data(self, current_price: float):
        """Ingests new price data to track volatility."""
        self.price_history.append(current_price)
        if len(self.price_history) > 20:
            self.price_history.pop(0)

    def get_market_volatility(self) -> float:
        """
        Returns a volatility score (0-100).
        Uses Standard Deviation of % returns over the last 20 ticks.
        """
        if len(self.price_history) < 5:
            return 0.0 # Not enough data, assume calm
            
        series = pd.Series(self.price_history)
        pct_change = series.pct_change().dropna()
        std_dev = pct_change.std()
        
        # Normalize: A std_dev of 0.05 (5% swings) is HIGH volatility (Score 100)
        # A std_dev of 0.001 (0.1% swings) is LOW (Score 2)
        vol_score = min((std_dev * 2000), 100.0)
        print("VOLTALITY SCORE:- \n", vol_score) 
        return vol_score

    # --- 2. Core Logic: Can we trade? ---
    def check_risk_constraints(self) -> dict:
        """
        Checks hard stops (Daily limits, Cooldowns).
        Returns: {'allowed': bool, 'reason': str}
        """
        now = datetime.now()
        
        # A. Reset Daily Counters (if new day)
        if now.day != self.last_reset_time.day:
            self.daily_pnl = 0.0
            self.daily_bets_count = 0
            self.last_reset_time = now

        # B. Check Cooldown
        time_since_last = (now - self.last_trade_time).total_seconds() / 60
        if time_since_last < self.cfg.COOLDOWN_MINUTES:
            return {'allowed': False, 'reason': f"Cooldown active ({int(self.cfg.COOLDOWN_MINUTES - time_since_last)}m left)"}

        # C. Check Daily Limits
        if self.daily_bets_count >= self.cfg.MAX_BETS_PER_DAY:
            return {'allowed': False, 'reason': "Max daily bets reached"}
            
        if self.daily_pnl <= -self.cfg.MAX_DAILY_LOSS:
            return {'allowed': False, 'reason': "Daily Stop-Loss Hit"}

        return {'allowed': True, 'reason': "OK"}

    # --- 3. The Decision: Size & Threshold ---
    def evaluate_trade(self, ai_confidence: float) -> dict:
        """
        Inputs: AI Confidence (0-100)
        Outputs: Decision (Buy/Pass) and Size
        """
        
        # Step A: Check Hard Constraints
        risk_check = self.check_risk_constraints()
        if not risk_check['allowed']:
            return {"action": "SKIP", "reason": risk_check['reason']}

        # Step B: Calculate Dynamic Threshold
        volatility = self.get_market_volatility()
        
        # Formula: Base + (Vol * Penalty)
        # Example: 75 + (20 * 0.5) = 85. 
        # In a volatile market, we need 85% confidence, not 75%.
        required_threshold = self.cfg.BASE_CONFIDENCE + (volatility * self.cfg.VOLATILITY_PENALTY)

        # Cap threshold at 95 (don't make it impossible)
        required_threshold = min(required_threshold, 95.0)
        print("required threshold:- \n", required_threshold)

        if ai_confidence < required_threshold:
            return {
                "action": "SKIP", 
                "reason": f"Confidence {ai_confidence}% < Dynamic Threshold {required_threshold:.1f}% (Vol: {volatility:.1f})"
            }

        # Step C: Calculate Position Size
        # Linear Scaling: 
        # If we barely pass threshold -> Base Size
        # If we are 100% confident -> Max Size
        
        excess_confidence = ai_confidence - required_threshold
        # Every 1% extra confidence adds 10% to the bet size
        size_multiplier = 1 + (excess_confidence * 0.1) 
        
        bet_size = self.cfg.BASE_BET_SIZE * size_multiplier
        bet_size = min(bet_size, self.cfg.MAX_BET_SIZE) # Hard cap
        print("BET SIZE:- \n", bet_size)

        return {
            "action": "TRADE",
            "amount": round(bet_size, 2),
            "reason": f"High Conf ({ai_confidence}%), Volatility OK"
        }

    # --- 4. Post-Trade: Update State ---
    def record_execution(self, amount, realized_pnl=0):
        """Call this after the trade settles (or optimistically)"""
        self.last_trade_time = datetime.now()
        self.daily_bets_count += 1
        self.daily_pnl += realized_pnl

        print("Recoded execution details:- \n", self.last_trade_time, self.daily_bets_count, self.daily_pnl)

# --- Test Runner ---
def run_tests():
    print("\n🚀 Starting Decision Engine Tests...\n")
    
    # Init Engine
    config = RiskConfig()
    engine = DecisionEngine(config)
    
    # 1. Warm up Volatility (simulate calm market)
    prices = [1.00, 1.01, 1.00, 1.01, 1.00, 1.01, 1.00]
    for p in prices:
        engine.update_market_data(p)
        
    print(f"📊 Market Volatility Score: {engine.get_market_volatility():.2f}")
    
    # 2. Test Trade (High Confidence)
    print("\n🧪 Test 1: High Confidence (85%) in Calm Market")
    decision1 = engine.evaluate_trade(ai_confidence=85.0)
    print(f"   Result: {decision1['action']} | Size: {decision1.get('amount')} | Reason: {decision1['reason']}")
    
    # Record it to trigger cooldown
    if decision1['action'] == "TRADE":
        engine.record_execution(amount=decision1.get('amount', 0))

    # 3. Test Cooldown
    print("\n🧪 Test 2: Immediate Follow-up (Should Fail Cooldown)")
    decision2 = engine.evaluate_trade(ai_confidence=90.0)
    print(f"   Result: {decision2['action']} | Reason: {decision2['reason']}")
    
    # 4. Simulate Volatility Spike
    print("\n🧪 Test 3: Volatility Spike")
    # Add wild prices
    engine.update_market_data(1.05)
    engine.update_market_data(1.15)
    engine.update_market_data(1.02)
    
    vol = engine.get_market_volatility()
    print(f"   New Volatility: {vol:.2f}")
    
    # Try trading with moderate confidence (should fail due to dynamic threshold)
    print("   Attempting trade with 80% confidence...")
    # Cheat cooldown for test
    engine.last_trade_time = datetime.min 
    
    decision3 = engine.evaluate_trade(ai_confidence=80.0)
    print(f"   Result: {decision3['action']} | Reason: {decision3['reason']}")

if __name__ == "__main__":
    run_tests()