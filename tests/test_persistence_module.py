"""Unit tests for persistence_module."""

import pytest
import json
from datetime import datetime
from persistence_module import SQLiteBackend, JSONBackend


class TestSQLiteBackend:
    """Test SQLiteBackend storage."""
    
    def test_init_creates_database(self, temp_dir):
        """Test database initialization."""
        db_path = str(temp_dir / "test.db")
        backend = SQLiteBackend(db_path)
        
        assert backend.db_path == db_path
        # Database file should be created
        assert (temp_dir / "test.db").exists()
    
    def test_save_and_load_session(self, temp_dir, sample_conversation_messages):
        """Test saving and loading a session."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        session_id = "test_session_001"
        metadata = {
            'started_at': '2026-02-04T10:00:00',
            'ended_at': '2026-02-04T10:05:00',
            'language': 'en'
        }
        
        # Save session
        backend.save_session(session_id, sample_conversation_messages, metadata)
        
        # Load session
        loaded = backend.load_session(session_id)
        
        assert loaded is not None
        assert loaded['session_id'] == session_id
        assert loaded['message_count'] == len(sample_conversation_messages)
        assert loaded['language'] == 'en'
        assert len(loaded['messages']) == len(sample_conversation_messages)
        
        # Verify message content
        assert loaded['messages'][0]['role'] == 'system'
        assert loaded['messages'][1]['role'] == 'user'
        assert loaded['messages'][2]['role'] == 'assistant'
    
    def test_load_nonexistent_session(self, temp_dir):
        """Test loading non-existent session."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        result = backend.load_session("nonexistent_session")
        
        assert result is None
    
    def test_list_sessions(self, temp_dir, sample_conversation_messages):
        """Test listing sessions."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        # Save multiple sessions
        for i in range(3):
            backend.save_session(
                f"session_{i}",
                sample_conversation_messages,
                {'started_at': f'2026-02-04T10:0{i}:00', 'language': 'en'}
            )
        
        # List sessions
        sessions = backend.list_sessions(limit=5)
        
        assert len(sessions) == 3
        # Should be in reverse chronological order
        assert sessions[0]['session_id'] == 'session_2'
    
    def test_delete_session(self, temp_dir, sample_conversation_messages):
        """Test deleting a session."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        session_id = "test_delete"
        backend.save_session(session_id, sample_conversation_messages, {'language': 'en'})
        
        # Verify it exists
        assert backend.load_session(session_id) is not None
        
        # Delete it
        backend.delete_session(session_id)
        
        # Verify it's gone
        assert backend.load_session(session_id) is None
    
    def test_search_sessions(self, temp_dir, sample_conversation_messages):
        """Test searching sessions by content."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        # Modify messages for unique search term
        messages = sample_conversation_messages.copy()
        messages[1]['content'] = "Tell me about quantum computing"
        
        backend.save_session("unique_session", messages, {'language': 'en'})
        backend.save_session("other_session", sample_conversation_messages, {'language': 'en'})
        
        # Search for unique term
        results = backend.search_sessions("quantum computing")
        
        assert len(results) == 1
        assert results[0]['session_id'] == 'unique_session'
    
    def test_get_last_session_id(self, temp_dir, sample_conversation_messages):
        """Test getting the most recent session ID."""
        backend = SQLiteBackend(str(temp_dir / "test.db"))
        
        # Initially no sessions
        assert backend.get_last_session_id() is None
        
        # Save sessions
        backend.save_session("session_1", sample_conversation_messages, 
                           {'started_at': '2026-02-04T10:00:00'})
        backend.save_session("session_2", sample_conversation_messages, 
                           {'started_at': '2026-02-04T11:00:00'})
        
        # Should return most recent
        assert backend.get_last_session_id() == "session_2"


class TestJSONBackend:
    """Test JSONBackend storage."""
    
    def test_init_creates_directory(self, temp_dir):
        """Test directory initialization."""
        storage_dir = temp_dir / "json_storage"
        backend = JSONBackend(str(storage_dir))
        
        assert backend.storage_dir == storage_dir
        assert storage_dir.exists()
    
    def test_save_and_load_session(self, temp_dir, sample_conversation_messages):
        """Test saving and loading a session."""
        backend = JSONBackend(str(temp_dir / "json_storage"))
        
        session_id = "test_json_session"
        metadata = {
            'started_at': '2026-02-04T10:00:00',
            'ended_at': '2026-02-04T10:05:00',
            'language': 'en'
        }
        
        # Save session
        backend.save_session(session_id, sample_conversation_messages, metadata)
        
        # Verify file exists
        session_file = temp_dir / "json_storage" / f"{session_id}.json"
        assert session_file.exists()
        
        # Load session
        loaded = backend.load_session(session_id)
        
        assert loaded is not None
        assert loaded['session_id'] == session_id
        assert loaded['message_count'] == len(sample_conversation_messages)
        assert len(loaded['messages']) == len(sample_conversation_messages)
    
    def test_load_nonexistent_session(self, temp_dir):
        """Test loading non-existent session."""
        backend = JSONBackend(str(temp_dir / "json_storage"))
        
        result = backend.load_session("nonexistent")
        
        assert result is None
    
    def test_list_sessions(self, temp_dir, sample_conversation_messages):
        """Test listing sessions."""
        backend = JSONBackend(str(temp_dir / "json_storage"))
        
        # Save multiple sessions
        for i in range(3):
            backend.save_session(f"session_{i}", sample_conversation_messages, {'language': 'en'})
        
        # List sessions
        sessions = backend.list_sessions(limit=5)
        
        assert len(sessions) <= 3
    
    def test_delete_session(self, temp_dir, sample_conversation_messages):
        """Test deleting a session."""
        backend = JSONBackend(str(temp_dir / "json_storage"))
        
        session_id = "delete_me"
        backend.save_session(session_id, sample_conversation_messages, {'language': 'en'})
        
        # Verify file exists
        session_file = temp_dir / "json_storage" / f"{session_id}.json"
        assert session_file.exists()
        
        # Delete session
        backend.delete_session(session_id)
        
        # Verify file is gone
        assert not session_file.exists()
    
    def test_search_sessions(self, temp_dir, sample_conversation_messages):
        """Test searching sessions."""
        backend = JSONBackend(str(temp_dir / "json_storage"))
        
        # Create session with unique content
        messages = sample_conversation_messages.copy()
        messages[1]['content'] = "artificial intelligence research"
        
        backend.save_session("ai_session", messages, {'language': 'en'})
        backend.save_session("other_session", sample_conversation_messages, {'language': 'en'})
        
        # Search
        results = backend.search_sessions("artificial intelligence")
        
        assert len(results) >= 1
        assert any(s['session_id'] == 'ai_session' for s in results)
