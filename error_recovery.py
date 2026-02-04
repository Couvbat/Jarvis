"""
Error recovery utilities for Jarvis voice assistant.

Provides retry logic, fallback chains, and graceful degradation
for handling transient and permanent failures.
"""

import time
import logging
from typing import Callable, Any, Optional, List, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass


class RetryPolicy:
    """Retry policy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: Tuple[type, ...] = (Exception,)
    ):
        """
        Initialize retry policy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            backoff_factor: Multiplier for delay between retries
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exceptions: Tuple of exception types to retry on
        """
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Succeeded on attempt {attempt}")
                return result
            except self.exceptions as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {str(e)}"
                )
                
                if attempt < self.max_attempts:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts failed"
                    )
        
        raise RetryExhaustedError(
            f"Failed after {self.max_attempts} attempts: {last_exception}"
        ) from last_exception


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[type, ...] = (Exception,)
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to retry on
        
    Example:
        @retry_with_backoff(max_attempts=3, initial_delay=2.0)
        def fetch_data():
            # Code that may fail transiently
            pass
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exceptions=exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return policy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


class FallbackChain:
    """Execute a chain of fallback functions until one succeeds."""
    
    def __init__(self, primary: Callable, *fallbacks: Callable):
        """
        Initialize fallback chain.
        
        Args:
            primary: Primary function to try first
            *fallbacks: Fallback functions to try in order
        """
        self.functions = [primary] + list(fallbacks)
    
    def execute(self, *args, **kwargs) -> Tuple[Any, int]:
        """
        Execute function chain until one succeeds.
        
        Args:
            *args: Positional arguments for functions
            **kwargs: Keyword arguments for functions
            
        Returns:
            Tuple of (result, index) where index indicates which function succeeded
            
        Raises:
            Exception: If all functions in chain fail
        """
        last_exception = None
        
        for idx, func in enumerate(self.functions):
            try:
                logger.info(
                    f"Trying {'primary' if idx == 0 else f'fallback {idx}'}: "
                    f"{func.__name__}"
                )
                result = func(*args, **kwargs)
                if idx > 0:
                    logger.warning(
                        f"Primary function failed, using fallback {idx}: "
                        f"{func.__name__}"
                    )
                return result, idx
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"{'Primary' if idx == 0 else f'Fallback {idx}'} function "
                    f"{func.__name__} failed: {str(e)}"
                )
                continue
        
        logger.error("All functions in fallback chain failed")
        raise last_exception


def fallback_chain(primary: Callable, *fallbacks: Callable):
    """
    Decorator for fallback chain.
    
    Args:
        primary: Primary function to try first
        *fallbacks: Fallback functions to try in order
        
    Example:
        def primary_stt(audio):
            return whisper_transcribe(audio)
            
        def fallback_stt(audio):
            return google_transcribe(audio)
            
        @fallback_chain(primary_stt, fallback_stt)
        def transcribe(audio):
            pass  # Will try primary_stt, then fallback_stt
    """
    chain = FallbackChain(primary, *fallbacks)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, _ = chain.execute(*args, **kwargs)
            return result
        return wrapper
    return decorator


class ServiceHealthChecker:
    """Health check utilities for external services."""
    
    @staticmethod
    def check_ollama(host: str = "http://localhost:11434") -> bool:
        """
        Check if Ollama service is available.
        
        Args:
            host: Ollama server URL
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    @staticmethod
    def check_audio_device() -> bool:
        """
        Check if audio input device is available.
        
        Returns:
            True if audio device is available, False otherwise
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            # Check for at least one input device
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            return len(input_devices) > 0
        except Exception as e:
            logger.warning(f"Audio device health check failed: {e}")
            return False
    
    @staticmethod
    def check_file_exists(filepath: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        import os
        return os.path.isfile(filepath)


class DegradedModeManager:
    """Manage degraded mode state and capabilities."""
    
    def __init__(self):
        """Initialize degraded mode manager."""
        self.degraded_services = set()
        self.offline_mode = False
    
    def mark_degraded(self, service: str):
        """
        Mark a service as degraded.
        
        Args:
            service: Service name (e.g., 'stt', 'llm', 'tts')
        """
        self.degraded_services.add(service)
        logger.warning(f"Service '{service}' is now degraded")
    
    def mark_recovered(self, service: str):
        """
        Mark a service as recovered.
        
        Args:
            service: Service name
        """
        if service in self.degraded_services:
            self.degraded_services.remove(service)
            logger.info(f"Service '{service}' has recovered")
    
    def is_degraded(self, service: str) -> bool:
        """
        Check if a service is degraded.
        
        Args:
            service: Service name
            
        Returns:
            True if service is degraded, False otherwise
        """
        return service in self.degraded_services
    
    def get_degraded_services(self) -> List[str]:
        """
        Get list of degraded services.
        
        Returns:
            List of degraded service names
        """
        return list(self.degraded_services)
    
    def enable_offline_mode(self):
        """Enable offline mode."""
        self.offline_mode = True
        logger.warning("Offline mode enabled")
    
    def disable_offline_mode(self):
        """Disable offline mode."""
        self.offline_mode = False
        logger.info("Offline mode disabled")
    
    def is_offline(self) -> bool:
        """Check if in offline mode."""
        return self.offline_mode


# Global degraded mode manager instance
degraded_mode = DegradedModeManager()
