"""Unit tests for llm_module."""

import pytest
from unittest.mock import MagicMock, patch
from llm_module import ConversationHistory, LLMModule
from persistence_module import SQLiteBackend


class TestConversationHistory:
    """Test ConversationHistory class."""
    
    def test_init(self):
        """Test initialization."""
        history = ConversationHistory(max_history=5)
        
        assert history.max_history == 5
        assert len(history.messages) == 0
        assert history.session_id is None
    
    def test_add_message(self):
        """Test adding messages."""
        history = ConversationHistory()
        
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        
        assert len(history.messages) == 2
        assert history.messages[0]['role'] == 'user'
        assert history.messages[0]['content'] == 'Hello'
        assert 'timestamp' in history.messages[0]
    
    def test_max_history_limit(self):
        """Test message history limit."""
        history = ConversationHistory(max_history=3)
        
        # Add system message
        history.add_message("system", "System prompt")
        
        # Add more messages than the limit
        for i in range(5):
            history.add_message("user", f"Message {i}")
        
        # Should keep system + last 3 messages
        assert len(history.messages) == 4  # 1 system + 3 recent
        assert history.messages[0]['role'] == 'system'
    
    def test_get_messages(self):
        """Test getting messages for LLM."""
        history = ConversationHistory()
        
        history.add_message("user", "Hello", metadata={"test": "data"})
        
        messages = history.get_messages()
        
        # Should return only role and content
        assert len(messages) == 1
        assert 'role' in messages[0]
        assert 'content' in messages[0]
        assert 'timestamp' not in messages[0]
        assert 'metadata' not in messages[0]
    
    def test_get_full_messages(self):
        """Test getting full messages with metadata."""
        history = ConversationHistory()
        
        history.add_message("user", "Hello", metadata={"test": "data"})
        
        messages = history.get_full_messages()
        
        # Should return full message data
        assert len(messages) == 1
        assert 'timestamp' in messages[0]
        assert 'metadata' in messages[0]
    
    def test_clear(self):
        """Test clearing history."""
        history = ConversationHistory()
        
        history.add_message("system", "System prompt")
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        
        history.clear()
        
        # Should keep only system message
        assert len(history.messages) == 1
        assert history.messages[0]['role'] == 'system'
    
    def test_set_session_id(self):
        """Test setting session ID."""
        history = ConversationHistory()
        
        history.set_session_id("test_session")
        
        assert history.session_id == "test_session"
        assert history.started_at is not None
    
    def test_save_to_storage(self, temp_dir):
        """Test saving to storage."""
        storage = SQLiteBackend(str(temp_dir / "test.db"))
        history = ConversationHistory(storage_backend=storage)
        
        history.set_session_id("test_save")
        history.add_message("user", "Test message")
        
        history.save_to_storage()
        
        # Verify it was saved
        loaded = storage.load_session("test_save")
        assert loaded is not None
        assert loaded['session_id'] == "test_save"
    
    def test_load_from_storage(self, temp_dir, sample_conversation_messages):
        """Test loading from storage."""
        storage = SQLiteBackend(str(temp_dir / "test.db"))
        
        # Save a session
        storage.save_session("test_load", sample_conversation_messages, {'language': 'en'})
        
        # Load it
        history = ConversationHistory(storage_backend=storage)
        success = history.load_from_storage("test_load")
        
        assert success is True
        assert history.session_id == "test_load"
        assert len(history.messages) == len(sample_conversation_messages)
    
    def test_load_last_session(self, temp_dir, sample_conversation_messages):
        """Test loading the most recent session."""
        storage = SQLiteBackend(str(temp_dir / "test.db"))
        
        # Save multiple sessions
        storage.save_session("old_session", sample_conversation_messages, 
                           {'started_at': '2026-02-01T10:00:00'})
        storage.save_session("recent_session", sample_conversation_messages, 
                           {'started_at': '2026-02-04T10:00:00'})
        
        # Load last session
        history = ConversationHistory(storage_backend=storage)
        success = history.load_last_session()
        
        assert success is True
        assert history.session_id == "recent_session"


class TestLLMModule:
    """Test LLMModule class."""
    
    @patch('llm_module.ollama.chat')
    def test_chat_success(self, mock_chat, mock_settings, mock_ollama_response):
        """Test successful chat interaction."""
        with patch('llm_module.settings', mock_settings):
            mock_chat.return_value = mock_ollama_response
            
            llm = LLMModule(enable_persistence=False)
            result = llm.chat("Hello")
            
            assert 'response' in result
            assert result['response'] == "This is a test response."
            assert mock_chat.called
    
    @patch('llm_module.ollama.chat')
    def test_chat_with_tools(self, mock_chat, mock_settings):
        """Test chat with tool calls."""
        with patch('llm_module.settings', mock_settings):
            mock_response = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "/tmp/test.txt"}
                            }
                        }
                    ]
                }
            }
            mock_chat.return_value = mock_response
            
            llm = LLMModule(enable_persistence=False)
            result = llm.chat("Read the file")
            
            assert result['tool_calls'] is not None
            assert len(result['tool_calls']) > 0
    
    @patch('llm_module.ollama.chat')
    def test_chat_with_retry(self, mock_chat, mock_settings):
        """Test chat with retry on failure."""
        with patch('llm_module.settings', mock_settings):
            # Fail twice, then succeed
            mock_chat.side_effect = [
                Exception("Connection error"),
                Exception("Connection error"),
                {
                    "message": {
                        "role": "assistant",
                        "content": "Success after retry",
                        "tool_calls": []
                    }
                }
            ]
            
            llm = LLMModule(enable_persistence=False)
            result = llm.chat("Test")
            
            assert result['response'] == "Success after retry"
            assert mock_chat.call_count == 3
    
    def test_offline_cache(self, mock_settings):
        """Test offline response caching."""
        with patch('llm_module.settings', mock_settings):
            llm = LLMModule(enable_persistence=False)
            
            # Manually add to cache
            llm._offline_cache["hello"] = {
                "response": "Cached hello",
                "tool_calls": None
            }
            
            # Get from cache
            result = llm._get_cached_response("Hello")
            
            assert result is not None
            assert result['response'] == "Cached hello"
    
    def test_degraded_response(self, mock_settings):
        """Test degraded mode responses."""
        with patch('llm_module.settings', mock_settings):
            llm = LLMModule(enable_persistence=False)
            
            # Test greeting response
            result = llm._get_degraded_response("Hello")
            assert "Hello" in result['response'] or "connection issues" in result['response']
            
            # Test exit response
            result = llm._get_degraded_response("goodbye")
            assert "Goodbye" in result['response']
    
    @patch('llm_module.ServiceHealthChecker.check_ollama')
    def test_check_health(self, mock_check, mock_settings):
        """Test Ollama health check."""
        with patch('llm_module.settings', mock_settings):
            mock_check.return_value = True
            
            llm = LLMModule(enable_persistence=False)
            result = llm.check_health()
            
            assert result is True
            mock_check.assert_called_once()
    
    def test_add_tool_result(self, mock_settings):
        """Test adding tool results to history."""
        with patch('llm_module.settings', mock_settings):
            llm = LLMModule(enable_persistence=False)
            
            llm.add_tool_result("read_file", "File contents: test")
            
            # Should add a tool message
            messages = llm.history.get_full_messages()
            assert any(msg['role'] == 'tool' for msg in messages)
    
    def test_reset_conversation(self, mock_settings):
        """Test resetting conversation."""
        with patch('llm_module.settings', mock_settings):
            llm = LLMModule(enable_persistence=False)
            
            # Add some messages
            llm.history.add_message("user", "Test")
            llm.history.add_message("assistant", "Response")
            
            # Reset
            llm.reset_conversation()
            
            # Should only have system message
            messages = llm.history.get_full_messages()
            assert len(messages) == 1
            assert messages[0]['role'] == 'system'
