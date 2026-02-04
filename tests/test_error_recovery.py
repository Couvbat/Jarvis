"""Unit tests for error_recovery module."""

import pytest
import time
from unittest.mock import MagicMock, patch
from error_recovery import (
    RetryPolicy,
    retry_with_backoff,
    FallbackChain,
    ServiceHealthChecker,
    DegradedModeManager,
    RetryExhaustedError
)


class TestRetryPolicy:
    """Test RetryPolicy class."""
    
    def test_retry_policy_success_first_attempt(self):
        """Test successful execution on first attempt."""
        policy = RetryPolicy(max_attempts=3)
        func = MagicMock(return_value="success")
        
        result = policy.execute(func)
        
        assert result == "success"
        assert func.call_count == 1
    
    def test_retry_policy_success_after_failures(self):
        """Test successful execution after retries."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.1)
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        result = policy.execute(func)
        
        assert result == "success"
        assert func.call_count == 3
    
    def test_retry_policy_exhausted(self):
        """Test retry exhaustion."""
        policy = RetryPolicy(max_attempts=3, initial_delay=0.1)
        func = MagicMock(side_effect=Exception("always fails"))
        
        with pytest.raises(RetryExhaustedError):
            policy.execute(func)
        
        assert func.call_count == 3
    
    def test_retry_policy_backoff(self):
        """Test exponential backoff."""
        policy = RetryPolicy(
            max_attempts=3,
            initial_delay=0.1,
            backoff_factor=2.0
        )
        func = MagicMock(side_effect=[Exception("fail"), Exception("fail"), "success"])
        
        start = time.time()
        policy.execute(func)
        duration = time.time() - start
        
        # Should take at least 0.1 + 0.2 = 0.3 seconds
        assert duration >= 0.3
        assert func.call_count == 3


class TestRetryDecorator:
    """Test retry_with_backoff decorator."""
    
    def test_decorator_success(self):
        """Test decorated function succeeds."""
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
    
    def test_decorator_retry(self):
        """Test decorated function retries."""
        call_count = {"value": 0}
        
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def test_func():
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise Exception("fail")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count["value"] == 3
    
    def test_decorator_exhausted(self):
        """Test decorated function exhausts retries."""
        @retry_with_backoff(max_attempts=2, initial_delay=0.1)
        def test_func():
            raise Exception("always fails")
        
        with pytest.raises(RetryExhaustedError):
            test_func()


class TestFallbackChain:
    """Test FallbackChain class."""
    
    def test_fallback_primary_succeeds(self):
        """Test primary function succeeds."""
        primary = MagicMock(return_value="primary")
        fallback = MagicMock(return_value="fallback")
        
        chain = FallbackChain(primary, fallback)
        result, index = chain.execute()
        
        assert result == "primary"
        assert index == 0
        assert primary.call_count == 1
        assert fallback.call_count == 0
    
    def test_fallback_primary_fails(self):
        """Test fallback when primary fails."""
        primary = MagicMock(side_effect=Exception("fail"))
        fallback = MagicMock(return_value="fallback")
        
        chain = FallbackChain(primary, fallback)
        result, index = chain.execute()
        
        assert result == "fallback"
        assert index == 1
        assert primary.call_count == 1
        assert fallback.call_count == 1
    
    def test_fallback_all_fail(self):
        """Test all functions in chain fail."""
        primary = MagicMock(side_effect=Exception("fail1"))
        fallback1 = MagicMock(side_effect=Exception("fail2"))
        fallback2 = MagicMock(side_effect=Exception("fail3"))
        
        chain = FallbackChain(primary, fallback1, fallback2)
        
        with pytest.raises(Exception):
            chain.execute()


class TestServiceHealthChecker:
    """Test ServiceHealthChecker class."""
    
    @patch('error_recovery.requests.get')
    def test_check_ollama_healthy(self, mock_get):
        """Test Ollama health check when healthy."""
        mock_get.return_value.status_code = 200
        
        result = ServiceHealthChecker.check_ollama()
        
        assert result is True
        mock_get.assert_called_once()
    
    @patch('error_recovery.requests.get')
    def test_check_ollama_unhealthy(self, mock_get):
        """Test Ollama health check when unhealthy."""
        mock_get.side_effect = Exception("Connection refused")
        
        result = ServiceHealthChecker.check_ollama()
        
        assert result is False
    
    @patch('error_recovery.sd.query_devices')
    def test_check_audio_device_available(self, mock_query):
        """Test audio device check when available."""
        mock_query.return_value = [
            {'name': 'test', 'max_input_channels': 2}
        ]
        
        result = ServiceHealthChecker.check_audio_device()
        
        assert result is True
    
    @patch('error_recovery.sd.query_devices')
    def test_check_audio_device_unavailable(self, mock_query):
        """Test audio device check when unavailable."""
        mock_query.side_effect = Exception("No devices")
        
        result = ServiceHealthChecker.check_audio_device()
        
        assert result is False
    
    def test_check_file_exists(self, temp_dir):
        """Test file existence check."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        assert ServiceHealthChecker.check_file_exists(str(test_file)) is True
        assert ServiceHealthChecker.check_file_exists(str(temp_dir / "nonexistent.txt")) is False


class TestDegradedModeManager:
    """Test DegradedModeManager class."""
    
    def test_mark_degraded(self):
        """Test marking service as degraded."""
        manager = DegradedModeManager()
        
        manager.mark_degraded("stt")
        
        assert manager.is_degraded("stt") is True
        assert "stt" in manager.get_degraded_services()
    
    def test_mark_recovered(self):
        """Test marking service as recovered."""
        manager = DegradedModeManager()
        
        manager.mark_degraded("stt")
        manager.mark_recovered("stt")
        
        assert manager.is_degraded("stt") is False
        assert "stt" not in manager.get_degraded_services()
    
    def test_offline_mode(self):
        """Test offline mode toggling."""
        manager = DegradedModeManager()
        
        assert manager.is_offline() is False
        
        manager.enable_offline_mode()
        assert manager.is_offline() is True
        
        manager.disable_offline_mode()
        assert manager.is_offline() is False
    
    def test_multiple_degraded_services(self):
        """Test multiple degraded services."""
        manager = DegradedModeManager()
        
        manager.mark_degraded("stt")
        manager.mark_degraded("llm")
        manager.mark_degraded("tts")
        
        services = manager.get_degraded_services()
        assert len(services) == 3
        assert set(services) == {"stt", "llm", "tts"}
