from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout # Seconds to wait before retrying
        self.last_failure_time = None
        self.is_open = False # Open = Broken Circuit (No Trading)

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            return True # Circuit just tripped
        return False

    def record_success(self):
        # If we successfully trade, reset the counter
        if not self.is_open:
            self.failure_count = 0

    def can_proceed(self):
        """
        Returns True if system is healthy.
        Returns False if Circuit is Open (stop trading).
        """
        if not self.is_open:
            return True

        # Check if enough time has passed to try again (Half-Open)
        if datetime.now() - self.last_failure_time > timedelta(seconds=self.reset_timeout):
            print("⚠️ Circuit Breaker: Probing system (Half-Open state)...")
            self.is_open = False # Tentatively close it
            self.failure_count = 0
            return True
            
        return False