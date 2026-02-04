"""LLM module with Ollama integration and function calling."""

import json
from typing import List, Dict, Any, Optional
import ollama
from loguru import logger
from config import settings


class ConversationHistory:
    """Manages conversation history with size limits."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({"role": role, "content": content})
        
        # Keep only the system message and recent history
        if len(self.messages) > self.max_history + 1:  # +1 for system message
            # Keep system message (first) and recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return self.messages
    
    def clear(self):
        """Clear conversation history except system message."""
        if self.messages:
            self.messages = [self.messages[0]]
        else:
            self.messages = []


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
    
    def __init__(self):
        self.host = settings.ollama_host
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.history = ConversationHistory(settings.max_conversation_history)
        
        # Initialize with system prompt
        self.history.add_message("system", self.SYSTEM_PROMPT)
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message to the LLM and get response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with 'response' (str) and optional 'tool_calls' (list)
        """
        logger.info(f"User: {user_message}")
        
        # Add user message to history
        self.history.add_message("user", user_message)
        
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
            
            return result
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            error_response = "I'm sorry, I encountered an error processing your request."
            self.history.add_message("assistant", error_response)
            return {"response": error_response, "tool_calls": None}
    
    def add_tool_result(self, tool_name: str, result: str):
        """Add tool execution result to conversation."""
        tool_message = f"Tool '{tool_name}' result: {result}"
        self.history.add_message("tool", tool_message)
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.history.clear()
        self.history.add_message("system", self.SYSTEM_PROMPT)
        logger.info("Conversation history reset")
