"""LLM module with Ollama integration and function calling."""

from collections.abc import Mapping
from typing import List, Dict, Any, Optional
import ollama
from loguru import logger
from config import settings


def to_plain(value: Any) -> Any:
    """Convert an Ollama response object into plain, JSON-safe data.

    The client returns pydantic models (mapping-like, but not dicts and not
    JSON-serialisable). Conversation history has to be plain data so it can be
    replayed to the server and, later, persisted.
    """
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if isinstance(value, Mapping):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


class ConversationHistory:
    """Conversation turns in Ollama's message format, with a size limit.

    The system prompt is held apart from the turns so that trimming and
    clearing cannot drop it, duplicate it, or count it against the limit.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max(0, max_history)
        self.system_message: Optional[Dict[str, Any]] = None
        self.turns: List[Dict[str, Any]] = []

    def set_system(self, content: str):
        """Set (or replace) the system prompt."""
        self.system_message = {"role": "system", "content": content}

    def add_message(self, role: str, content: str, **fields: Any):
        """Append a turn. Extra fields (tool_calls, name) are kept as-is."""
        if role == "system":
            self.set_system(content)
            return

        message: Dict[str, Any] = {"role": role, "content": content}
        message.update({key: value for key, value in fields.items() if value})
        self.turns.append(message)
        self._trim()

    def add_user(self, content: str):
        """Append a user turn."""
        self.add_message("user", content)

    def add_assistant(self, content: str, tool_calls: Optional[List[Any]] = None):
        """Append an assistant turn, keeping any tool calls structured."""
        self.add_message("assistant", content or "", tool_calls=to_plain(tool_calls))

    def add_tool_result(self, name: str, content: str):
        """Append a tool result in the role the protocol expects."""
        self.add_message("tool", content, name=name)

    def _trim(self):
        """Drop the oldest turns, without orphaning a tool result."""
        if len(self.turns) <= self.max_history:
            return

        self.turns = self.turns[len(self.turns) - self.max_history:]

        # A tool result whose assistant turn was just dropped has nothing to
        # attach to and confuses the model; drop those too.
        while self.turns and self.turns[0]["role"] == "tool":
            self.turns.pop(0)

    def get_messages(self) -> List[Dict[str, Any]]:
        """The full message list to send to the model."""
        messages = [self.system_message] if self.system_message else []
        return messages + self.turns

    def clear(self):
        """Drop every turn, keeping the system prompt."""
        self.turns = []


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
    
    #: Safety net against a local model that keeps asking for tools forever.
    MAX_TOOL_ITERATIONS = 5

    def __init__(self):
        self.host = settings.ollama_host
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.history = ConversationHistory(settings.max_conversation_history)
        self.history.set_system(self.SYSTEM_PROMPT)

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Send a user message to the LLM and get its reply.

        Args:
            user_message: The user's message

        Returns:
            Dict with 'response' (str) and 'tool_calls' (list or None)
        """
        logger.info(f"User: {user_message}")
        self.history.add_user(user_message)
        return self._generate()

    def continue_after_tools(self) -> Dict[str, Any]:
        """
        Ask the model to carry on from the tool results already in history.

        No message is appended: the tool results are the new information, and
        inventing a user turn to prompt for a summary would pollute the
        conversation and pin the reply to whatever language that turn was
        written in.
        """
        logger.info("Continuing after tool results")
        return self._generate()

    def _generate(self) -> Dict[str, Any]:
        """Run one model turn against the current history."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.history.get_messages(),
                tools=self.TOOLS,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )

            message = response.get("message", {}) or {}
            content = message.get("content", "") or ""
            tool_calls = to_plain(message.get("tool_calls") or [])

            self.history.add_assistant(content, tool_calls)

            logger.info(f"Assistant: {content}")
            if tool_calls:
                logger.info(f"Tool calls: {len(tool_calls)}")

            return {
                "response": content,
                "tool_calls": tool_calls or None
            }

        except Exception as e:
            logger.error(f"LLM error: {e}")
            error_response = "I'm sorry, I encountered an error processing your request."
            self.history.add_assistant(error_response)
            return {"response": error_response, "tool_calls": None}

    def add_tool_result(self, tool_name: str, result: str):
        """Record a tool's output so the model can use it on the next turn."""
        self.history.add_tool_result(tool_name, result)

    def reset_conversation(self):
        """Reset conversation history, keeping the system prompt."""
        self.history.clear()
        logger.info("Conversation history reset")
