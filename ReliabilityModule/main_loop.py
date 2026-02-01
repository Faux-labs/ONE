import time
import traceback
from guardian import logger, send_alert, log_trade
from rate_limiter import RateLimiter
from circuit_breaker import CircuitBreaker

# 1. Initialize Safety Modules
groq_limiter = RateLimiter(max_requests=3, period_seconds=60) # 3 req/min
rpc_limiter = RateLimiter(max_requests=10, period_seconds=10)   # 1 req/sec
circuit = CircuitBreaker(failure_threshold=5, reset_timeout=600)

def trading_cycle():
    """
    One single iteration of logic: Fetch News -> Analyze -> Trade
    """
    # 1. Check Circuit Breaker
    if not circuit.can_proceed():
        logger.warning("🛑 Trading Halted due to Circuit Breaker.")
        return

    # 2. Rate Limit Check
    groq_limiter.wait_for_slot()
    
    # --- [INSERT YOUR ANALYSIS CODE HERE] ---
    # news = fetch_news()
    # analysis = analyze(news)
    
    # Mocking Success for demo
    logger.info("✅ Cycle completed successfully.")
    circuit.record_success() 

def run_forever():
    logger.info("🤖 System Started. 24/7 Watchdog active.")
    send_alert("System Up", "OneBot has started monitoring.", "00ff00")

    while True:
        try:
            trading_cycle()
            
            # Standard Heartbeat Sleep
            time.sleep(60) 

        except KeyboardInterrupt:
            logger.info("👋 Manual Stop.")
            break
            
        except Exception as e:
            # --- GLOBAL ERROR HANDLER ---
            # This block catches ANY crash to keep the script running
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            logger.error(f"CRITICAL CRASH: {error_msg}")
            
            # Trip the circuit breaker logic
            just_tripped = circuit.record_failure()
            
            if just_tripped:
                send_alert("🚨 Circuit Breaker Tripped", f"Too many errors. Pausing for 10m.\nError: {error_msg}", "ff0000")
            
            # Exponential Backoff for the loop itself
            # If we are failing, sleep longer to let API recover
            sleep_time = min(60 * (2 ** circuit.failure_count), 3600) 
            logger.info(f"Sleeping {sleep_time}s before restart...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    run_forever()