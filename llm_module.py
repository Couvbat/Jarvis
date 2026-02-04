"""LLM module with Ollama integration and function calling."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import ollama
from loguru import logger
from config import settings
from error_recovery import retry_with_backoff, RetryExhaustedError, degraded_mode, ServiceHealthChecker
from persistence_module import SQLiteBackend, StorageBackend


class ConversationHistory:
    """Manages conversation history with size limits and persistence."""
    
    def __init__(self, max_history: int = 10, storage_backend: Optional[StorageBackend] = None):
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.session_id: Optional[str] = None
        self.started_at: Optional[str] = None
        self.storage = storage_backend
        
    def set_session_id(self, session_id: str):
        """Set the current session ID."""
        self.session_id = session_id
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        logger.debug(f"Session ID set to: {session_id}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to history with timestamp.
        
        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Optional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Keep only the system message and recent history
        if len(self.messages) > self.max_history + 1:  # +1 for system message
            # Keep system message (first) and recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in format expected by LLM.
        
        Returns:
            List of messages with only role and content
        """
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def get_full_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages with full metadata.
        
        Returns:
            List of messages with timestamps and metadata
        """
        return self.messages
    
    def clear(self):
        """Clear conversation history except system message."""
        if self.messages:
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def save_to_storage(self, session_id: Optional[str] = None):
        """
        Save current session to persistent storage.
        
        Args:
            session_id: Optional session ID (uses current if not provided)
        """
        if self.storage is None:
            logger.warning("No storage backend configured, cannot save session")
            return
        
        sid = session_id or self.session_id
        if not sid:
            logger.warning("No session ID set, cannot save")
            return
        
        metadata = {
            'started_at': self.started_at or datetime.now().isoformat(),
            'ended_at': datetime.now().isoformat(),
            'language': settings.whisper_language
        }
        
        try:
            self.storage.save_session(sid, self.messages, metadata)
            logger.debug(f"Session {sid} saved to storage")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def load_from_storage(self, session_id: str):
        """
        Load a session from persistent storage.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.storage is None:
            logger.warning("No storage backend configured, cannot load session")
            return False
        
        try:
            session_data = self.storage.load_session(session_id)
            if session_data:
                self.messages = session_data['messages']
                self.session_id = session_id
                self.started_at = session_data['started_at']
                logger.info(f"Loaded session {session_id} with {len(self.messages)} messages")
                return True
            else:
                logger.warning(f"Session {session_id} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def load_last_session(self) -> bool:
        """
        Load the most recent session from storage.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.storage is None:
            return False
        
        try:
            # Try to get last session ID
            if hasattr(self.storage, 'get_last_session_id'):
                last_session_id = self.storage.get_last_session_id()
                if last_session_id:
                    return self.load_from_storage(last_session_id)
            
            # Fallback: list sessions and load first
            sessions = self.storage.list_sessions(limit=1)
            if sessions:
                return self.load_from_storage(sessions[0]['session_id'])
            
            return False
        except Exception as e:
            logger.error(f"Failed to load last session: {e}")
            return False
    
    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        if self.storage is None:
            return []
        
        try:
            return self.storage.list_sessions(limit)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def search_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversation history for specific content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching sessions
        """
        if self.storage is None:
            return []
        
        try:
            return self.storage.search_sessions(query, limit)
        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            return []


class LLMModule:
    """LLM service using Ollama with function calling support."""
    
    # System prompt defining assistant behavior and available tools
    SYSTEM_PROMPT = """You are Jarvis, a helpful local voice assistant running on Linux. You can help users with:

1. File operations: create, read, edit, delete files and directories
2. Web information: fetch and summarize web pages
3. Application launching: open applications on the system
4. General questions and conversation

When performing system operations, be careful and confirm destructive actions.
Always provide clear, concise responses suitable for voice output.
Use the available tools when needed to accomplish tasks.

IMPORTANT: Respond in the same language the user is speaking. If they speak French, respond in French. If they speak English, respond in English."""

    # Tool definitions for function calling
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_file_operation",
                "description": "Perform file system operations like creating, reading, editing, or deleting files and directories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["create_file", "read_file", "delete_file", "create_directory", "list_directory"],
                            "description": "The file operation to perform"
                        },
                        "path": {
                            "type": "string",
                            "description": "The file or directory path"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content for create/write operations (optional)"
                        }
                    },
                    "required": ["operation", "path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_web_page",
                "description": "Fetch and extract text content from a web page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "launch_application",
                "description": "Launch an application or execute a system command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "application": {
                            "type": "string",
                            "description": "The application name or command to execute"
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional arguments for the application"
                        }
                    },
                    "required": ["application"]
                }
            }
        }
    ]
    
    def __init__(self, enable_persistence: bool = False):
        """
        Initialize LLM module.
        
        Args:
            enable_persistence: Enable conversation persistence
        """
        self.host = settings.ollama_host
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        # Initialize storage backend if persistence is enabled
        storage = None
        if enable_persistence:
            try:
                storage = SQLiteBackend(settings.conversation_db_path)
                logger.info("Conversation persistence enabled")
            except Exception as e:
                logger.error(f"Failed to initialize persistence: {e}")
                logger.warning("Continuing without persistence")
        
        self.history = ConversationHistory(
            max_history=settings.max_conversation_history,
            storage_backend=storage
        )
        self._offline_cache = {}  # Simple cache for offline mode
        
        # Initialize with system prompt
        self.history.add_message("system", self.SYSTEM_PROMPT)
    
    def check_health(self) -> bool:
        """Check if Ollama service is available."""
        return ServiceHealthChecker.check_ollama(self.host)
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message to the LLM and get response with retry logic.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with 'response' (str) and optional 'tool_calls' (list)
        """
        logger.info(f"User: {user_message}")
        
        # Add user message to history
        self.history.add_message("user", user_message)
        
        # Check if offline mode and try cache first
        if degraded_mode.is_offline() and settings.enable_offline_mode:
            cached_response = self._get_cached_response(user_message)
            if cached_response:
                logger.info("Using cached response (offline mode)")
                return cached_response
        
        # Try with retry logic if enabled
        if settings.enable_retry_logic:
            try:
                return self._chat_with_retry()
            except RetryExhaustedError as e:
                logger.error(f"All LLM retry attempts failed: {e}")
                degraded_mode.mark_degraded('llm')
                # Return degraded response
                return self._get_degraded_response(user_message)
        else:
            return self._chat_once()
    
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        backoff_factor=2.0,
        exceptions=(Exception,)
    )
    def _chat_with_retry(self) -> Dict[str, Any]:
        """Internal method with retry decorator."""
        return self._chat_once()
    
    def _chat_once(self) -> Dict[str, Any]:
        """Single chat attempt without retry."""
        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model,
                messages=self.history.get_messages(),
                tools=self.TOOLS,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            message = response.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Add assistant response to history
            self.history.add_message("assistant", content if content else json.dumps(tool_calls))
            
            result = {
                "response": content,
                "tool_calls": tool_calls if tool_calls else None
            }
            
            logger.info(f"Assistant: {content}")
            if tool_calls:
                logger.info(f"Tool calls: {len(tool_calls)}")
            
            # Mark as recovered if it was degraded
            if degraded_mode.is_degraded('llm'):
                degraded_mode.mark_recovered('llm')
            
            # Cache common responses if offline mode is enabled
            if settings.enable_offline_mode:
                self._cache_response(self.history.messages[-2]["content"], result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise
    
    def _get_cached_response(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Get cached response for common queries."""
        # Normalize message
        normalized = user_message.lower().strip()
        
        # Check cache
        return self._offline_cache.get(normalized)
    
    def _cache_response(self, user_message: str, response: Dict[str, Any]):
        """Cache response for offline use."""
        # Normalize message
        normalized = user_message.lower().strip()
        
        # Only cache non-tool responses
        if not response.get("tool_calls"):
            # Limit cache size
            if len(self._offline_cache) >= settings.offline_response_cache_size:
                # Remove oldest (FIFO)
                self._offline_cache.pop(next(iter(self._offline_cache)))
            
            self._offline_cache[normalized] = response
    
    def _get_degraded_response(self, user_message: str) -> Dict[str, Any]:
        """Get response when LLM is unavailable."""
        # Try cache first
        cached = self._get_cached_response(user_message)
        if cached:
            logger.info("Using cached response (degraded mode)")
            return cached
        
        # Simple rule-based responses for common patterns
        normalized = user_message.lower()
        
        if any(word in normalized for word in ['hello', 'hi', 'hey', 'bonjour', 'salut']):
            response = "Hello! I'm currently experiencing connection issues, but I'm still here to help where I can."
        elif any(word in normalized for word in ['exit', 'quit', 'goodbye', 'bye', 'au revoir']):
            response = "Goodbye! I hope to be fully operational soon."
        elif any(word in normalized for word in ['help', 'aide']):
            response = "I'm currently in degraded mode. My AI capabilities are limited, but I can still try to help with basic tasks."
        else:
            response = "I'm sorry, I'm currently unable to process complex requests. My AI service is unavailable. Please try again later."
        
        error_response = {
            "response": response,
            "tool_calls": None
        }
        self.history.add_message("assistant", response)
        return error_response
    
    def add_tool_result(self, tool_name: str, result: str):
        """Add tool execution result to conversation."""
        tool_message = f"Tool '{tool_name}' result: {result}"
        self.history.add_message("tool", tool_message)
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.history.clear()
        self.history.add_message("system", self.SYSTEM_PROMPT)
        logger.info("Conversation history reset")
