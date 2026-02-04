"""
Conversation persistence module for Jarvis voice assistant.

Provides storage backends for saving and loading conversation history
across sessions using SQLite or JSON.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_session(self, session_id: str, messages: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Save a conversation session."""
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation session."""
        pass
    
    @abstractmethod
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str):
        """Delete a session."""
        pass
    
    @abstractmethod
    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search sessions by content."""
        pass


class SQLiteBackend(StorageBackend):
    """SQLite storage backend for conversation persistence."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            # Default to user data directory
            data_dir = Path.home() / ".local" / "share" / "jarvis"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "conversations.db")
        
        self.db_path = db_path
        self._init_database()
        logger.info(f"SQLite persistence initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                message_count INTEGER,
                language TEXT,
                metadata TEXT
            )
        """)
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_time 
            ON sessions(started_at DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_message_session 
            ON messages(session_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def save_session(self, session_id: str, messages: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Save a conversation session.
        
        Args:
            session_id: Unique session identifier
            messages: List of message dictionaries
            metadata: Session metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract timestamps
            started_at = metadata.get('started_at', datetime.now().isoformat())
            ended_at = metadata.get('ended_at', datetime.now().isoformat())
            language = metadata.get('language', 'auto')
            
            # Insert or update session
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, started_at, ended_at, message_count, language, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                started_at,
                ended_at,
                len(messages),
                language,
                json.dumps(metadata)
            ))
            
            # Delete existing messages for this session (if updating)
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            
            # Insert messages
            for msg in messages:
                timestamp = msg.get('timestamp', datetime.now().isoformat())
                msg_metadata = msg.get('metadata', {})
                
                cursor.execute("""
                    INSERT INTO messages 
                    (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    msg.get('role', ''),
                    msg.get('content', ''),
                    timestamp,
                    json.dumps(msg_metadata)
                ))
            
            conn.commit()
            logger.debug(f"Saved session {session_id} with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data with messages and metadata, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Load session metadata
            cursor.execute("""
                SELECT started_at, ended_at, message_count, language, metadata
                FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            started_at, ended_at, message_count, language, metadata_json = session_row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Load messages
            cursor.execute("""
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            messages = []
            for row in cursor.fetchall():
                role, content, timestamp, msg_metadata_json = row
                msg_metadata = json.loads(msg_metadata_json) if msg_metadata_json else {}
                
                messages.append({
                    'role': role,
                    'content': content,
                    'timestamp': timestamp,
                    'metadata': msg_metadata
                })
            
            logger.debug(f"Loaded session {session_id} with {len(messages)} messages")
            
            return {
                'session_id': session_id,
                'started_at': started_at,
                'ended_at': ended_at,
                'message_count': message_count,
                'language': language,
                'metadata': metadata,
                'messages': messages
            }
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None
        finally:
            conn.close()
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT session_id, started_at, ended_at, message_count, language
                FROM sessions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                session_id, started_at, ended_at, message_count, language = row
                sessions.append({
                    'session_id': session_id,
                    'started_at': started_at,
                    'ended_at': ended_at,
                    'message_count': message_count,
                    'language': language
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
        finally:
            conn.close()
    
    def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.info(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search sessions by content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching sessions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Search in message content
            cursor.execute("""
                SELECT DISTINCT s.session_id, s.started_at, s.ended_at, 
                       s.message_count, s.language
                FROM sessions s
                JOIN messages m ON s.session_id = m.session_id
                WHERE m.content LIKE ?
                ORDER BY s.started_at DESC
                LIMIT ?
            """, (f'%{query}%', limit))
            
            sessions = []
            for row in cursor.fetchall():
                session_id, started_at, ended_at, message_count, language = row
                sessions.append({
                    'session_id': session_id,
                    'started_at': started_at,
                    'ended_at': ended_at,
                    'message_count': message_count,
                    'language': language
                })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return []
        finally:
            conn.close()
    
    def get_last_session_id(self) -> Optional[str]:
        """
        Get the most recent session ID.
        
        Returns:
            Session ID or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT session_id FROM sessions
                ORDER BY started_at DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Error getting last session: {e}")
            return None
        finally:
            conn.close()


class JSONBackend(StorageBackend):
    """JSON file storage backend for conversation persistence."""
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize JSON backend.
        
        Args:
            storage_dir: Directory for JSON files
        """
        if storage_dir is None:
            storage_dir = str(Path.home() / ".local" / "share" / "jarvis" / "sessions")
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON persistence initialized: {self.storage_dir}")
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.storage_dir / f"{session_id}.json"
    
    def save_session(self, session_id: str, messages: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Save a conversation session to JSON file."""
        session_data = {
            'session_id': session_id,
            'started_at': metadata.get('started_at', datetime.now().isoformat()),
            'ended_at': metadata.get('ended_at', datetime.now().isoformat()),
            'message_count': len(messages),
            'language': metadata.get('language', 'auto'),
            'metadata': metadata,
            'messages': messages
        }
        
        session_path = self._get_session_path(session_id)
        
        try:
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved session {session_id} to {session_path}")
        except Exception as e:
            logger.error(f"Error saving session to JSON: {e}")
            raise
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation session from JSON file."""
        session_path = self._get_session_path(session_id)
        
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            logger.debug(f"Loaded session {session_id} from {session_path}")
            return session_data
        except Exception as e:
            logger.error(f"Error loading session from JSON: {e}")
            return None
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions."""
        session_files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        sessions = []
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    'session_id': data['session_id'],
                    'started_at': data['started_at'],
                    'ended_at': data.get('ended_at'),
                    'message_count': data['message_count'],
                    'language': data.get('language', 'auto')
                })
            except Exception as e:
                logger.error(f"Error reading session file {session_file}: {e}")
        
        return sessions
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        session_path = self._get_session_path(session_id)
        
        try:
            if session_path.exists():
                session_path.unlink()
                logger.info(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
    
    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search sessions by content."""
        matching_sessions = []
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Search in message content
                for msg in data.get('messages', []):
                    if query.lower() in msg.get('content', '').lower():
                        matching_sessions.append({
                            'session_id': data['session_id'],
                            'started_at': data['started_at'],
                            'ended_at': data.get('ended_at'),
                            'message_count': data['message_count'],
                            'language': data.get('language', 'auto')
                        })
                        break
                
                if len(matching_sessions) >= limit:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching session file {session_file}: {e}")
        
        return matching_sessions
    
    def get_last_session_id(self) -> Optional[str]:
        """Get the most recent session ID."""
        session_files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not session_files:
            return None
        
        try:
            with open(session_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['session_id']
        except Exception as e:
            logger.error(f"Error getting last session: {e}")
            return None
