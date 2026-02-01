import time
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, period_seconds: int):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.timestamps = deque()

    def wait_for_slot(self):
        """
        Blocks execution until a slot is available.
        """
        now = time.time()
        
        # Remove old timestamps
        while self.timestamps and now - self.timestamps[0] > self.period_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.max_requests:
            # We are full. Calculate wait time.
            oldest = self.timestamps[0]
            wait_time = self.period_seconds - (now - oldest)
            
            if wait_time > 0:
                print(f"🚦 Rate Limit Hit. Sleeping {wait_time:.2f}s...")
                time.sleep(wait_time)
                # Recursive call to clean up again after sleep
                return self.wait_for_slot()

        self.timestamps.append(time.time())