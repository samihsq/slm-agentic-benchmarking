"""
Adaptive Rate Limiter for API calls.

Dynamically adjusts concurrency based on error rates:
- Starts at max concurrency
- Reduces on errors with exponential backoff
- Gradually recovers when successful
"""

import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


@dataclass
class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts concurrency based on errors.
    
    Usage:
        limiter = AdaptiveRateLimiter(max_concurrency=50)
        
        async with limiter.acquire():
            result = await make_api_call()
            if error:
                limiter.record_error()
            else:
                limiter.record_success()
    """
    
    max_concurrency: int = 50
    min_concurrency: int = 1
    initial_concurrency: Optional[int] = None  # Defaults to max
    
    # Backoff parameters
    backoff_factor: float = 0.5  # Reduce to 50% on error
    recovery_factor: float = 1.1  # Increase by 10% on success streak
    success_streak_threshold: int = 10  # Successes needed before recovery
    
    # Internal state
    _current_concurrency: int = field(init=False)
    _semaphore: asyncio.Semaphore = field(init=False, default=None)
    _lock: Lock = field(init=False, default_factory=Lock)
    _success_streak: int = field(init=False, default=0)
    _total_errors: int = field(init=False, default=0)
    _total_successes: int = field(init=False, default=0)
    _last_error_time: float = field(init=False, default=0)
    _backoff_until: float = field(init=False, default=0)
    
    def __post_init__(self):
        self._current_concurrency = self.initial_concurrency or self.max_concurrency
        self._semaphore = None  # Will be created lazily
        self._lock = Lock()
        self._success_streak = 0
        self._total_errors = 0
        self._total_successes = 0
        self._last_error_time = 0
        self._backoff_until = 0
    
    @property
    def current_concurrency(self) -> int:
        return self._current_concurrency
    
    @property
    def stats(self) -> dict:
        return {
            "current_concurrency": self._current_concurrency,
            "max_concurrency": self.max_concurrency,
            "total_successes": self._total_successes,
            "total_errors": self._total_errors,
            "success_streak": self._success_streak,
            "error_rate": self._total_errors / max(1, self._total_successes + self._total_errors),
        }
    
    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore (must be called from async context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._current_concurrency)
        return self._semaphore
    
    async def acquire(self):
        """Context manager for rate-limited execution."""
        return _AcquireContext(self)
    
    def record_success(self):
        """Record a successful API call."""
        with self._lock:
            self._total_successes += 1
            self._success_streak += 1
            
            # Try to recover concurrency after success streak
            if self._success_streak >= self.success_streak_threshold:
                if self._current_concurrency < self.max_concurrency:
                    new_concurrency = min(
                        self.max_concurrency,
                        int(self._current_concurrency * self.recovery_factor)
                    )
                    if new_concurrency > self._current_concurrency:
                        self._current_concurrency = new_concurrency
                        self._success_streak = 0  # Reset streak
    
    def record_error(self, is_rate_limit: bool = True):
        """Record an error (rate limit or other)."""
        with self._lock:
            self._total_errors += 1
            self._success_streak = 0
            self._last_error_time = time.time()
            
            if is_rate_limit:
                # Reduce concurrency
                new_concurrency = max(
                    self.min_concurrency,
                    int(self._current_concurrency * self.backoff_factor)
                )
                self._current_concurrency = new_concurrency
                
                # Calculate backoff time (exponential based on recent errors)
                recent_errors = sum(1 for _ in range(min(5, self._total_errors)))
                backoff_seconds = min(60, 2 ** recent_errors)
                self._backoff_until = time.time() + backoff_seconds
    
    async def wait_if_backing_off(self):
        """Wait if we're in a backoff period."""
        if time.time() < self._backoff_until:
            wait_time = self._backoff_until - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)


class _AcquireContext:
    """Async context manager for semaphore acquisition."""
    
    def __init__(self, limiter: AdaptiveRateLimiter):
        self.limiter = limiter
    
    async def __aenter__(self):
        # Wait if backing off
        await self.limiter.wait_if_backing_off()
        
        # Acquire semaphore
        sem = self.limiter._get_semaphore()
        await sem.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Release semaphore
        sem = self.limiter._get_semaphore()
        sem.release()
        return False


# Thread-safe version for ThreadPoolExecutor
class ThreadSafeAdaptiveLimiter:
    """
    Thread-safe adaptive rate limiter for use with ThreadPoolExecutor.
    """
    
    def __init__(
        self,
        max_concurrency: int = 50,
        min_concurrency: int = 1,
        backoff_factor: float = 0.5,
        recovery_threshold: int = 10,
    ):
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.backoff_factor = backoff_factor
        self.recovery_threshold = recovery_threshold
        
        self._current_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = Lock()
        self._success_streak = 0
        self._error_count = 0
        self._backoff_until = 0.0
    
    @property
    def current_concurrency(self) -> int:
        return self._current_concurrency
    
    def wait_if_needed(self) -> float:
        """
        Check if we need to wait due to backoff.
        Returns the time waited (0 if no wait needed).
        """
        now = time.time()
        if now < self._backoff_until:
            wait_time = self._backoff_until - now
            time.sleep(wait_time)
            return wait_time
        return 0.0
    
    def record_success(self):
        """Record successful call, potentially increase concurrency."""
        with self._lock:
            self._success_streak += 1
            if self._success_streak >= self.recovery_threshold:
                if self._current_concurrency < self.max_concurrency:
                    self._current_concurrency = min(
                        self.max_concurrency,
                        self._current_concurrency + 5  # Gradual recovery
                    )
                self._success_streak = 0
    
    def record_error(self, backoff_seconds: float = 2.0):
        """Record error, reduce concurrency and set backoff."""
        with self._lock:
            self._error_count += 1
            self._success_streak = 0
            
            # Reduce concurrency
            self._current_concurrency = max(
                self.min_concurrency,
                int(self._current_concurrency * self.backoff_factor)
            )
            
            # Set backoff (exponential based on consecutive errors)
            actual_backoff = min(60, backoff_seconds * (2 ** min(5, self._error_count)))
            self._backoff_until = time.time() + actual_backoff
            
            return actual_backoff
    
    def get_stats(self) -> dict:
        return {
            "current_concurrency": self._current_concurrency,
            "max_concurrency": self.max_concurrency,
            "error_count": self._error_count,
            "success_streak": self._success_streak,
        }
