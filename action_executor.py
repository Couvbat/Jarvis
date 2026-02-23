"""Action executor for system operations with safety checks."""

import os
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import requests
from bs4 import BeautifulSoup
from loguru import logger
from config import settings
from whitelist_manager import WhitelistManager
from error_recovery import retry_with_backoff, RetryExhaustedError


# Commands that always require confirmation, even if whitelisted
DANGEROUS_COMMANDS = {'rm', 'rmdir', 'shred', 'dd', 'mkfs', 'fdisk', 'kill', 'killall', 'pkill', 'chmod', 'chown'}


class ActionExecutor:
    """Executes system actions with safety validation."""
    
    def __init__(self, confirmation_callback: Optional[Callable[[str, str], tuple[bool, bool]]] = None):
        self.allowed_dirs = settings.allowed_dirs_list
        self.command_whitelist = settings.command_whitelist_list
        self.whitelist_manager = WhitelistManager()
        self.confirmation_callback = confirmation_callback
    
    def _request_confirmation(self, category: str, action_description: str, item: str, force: bool = False) -> tuple[bool, bool]:
        """
        Request user confirmation for an action.
        
        Args:
            category: Whitelist category (file_operations, applications, web_urls)
            action_description: Human-readable description of the action
            item: The item to potentially whitelist
            force: If True, always prompt even if whitelisted (for dangerous operations)
            
        Returns:
            Tuple of (execute_action, add_to_whitelist)
        """
        # Check if already whitelisted (unless forced)
        if not force and self.whitelist_manager.is_whitelisted(category, item):
            logger.info(f"Action auto-approved (whitelisted): {action_description}")
            return (True, False)
        
        if force:
            logger.warning(f"Dangerous operation requires confirmation: {action_description}")
        
        # Request confirmation via callback
        if self.confirmation_callback:
            return self.confirmation_callback(action_description, item)
        
        # Default: don't execute if no callback
        logger.warning(f"No confirmation callback set, denying: {action_description}")
        return (False, False)
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within allowed directories."""
        try:
            resolved_path = path.resolve()
            
            # Check if path is under any allowed directory
            for allowed_dir in self.allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    return True
                except ValueError:
                    continue
            
            logger.warning(f"Path not allowed: {path}")
            return False
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    def execute_file_operation(
        self, 
        operation: str, 
        path: str, 
        content: Optional[str] = None
    ) -> str:
        """
        Execute file system operations safely.
        
        Args:
            operation: Type of operation (create_file, read_file, etc.)
            path: File or directory path
            content: Content for write operations
            
        Returns:
            Result message
        """
        logger.info(f"File operation: {operation} on {path}")
        
        file_path = Path(path).expanduser()
        
        # Validate path
        if not self._is_path_allowed(file_path):
            return f"Error: Path '{path}' is not in allowed directories"
        
        # Request confirmation for destructive operations
        if operation in ["delete_file", "delete_directory", "create_file"]:
            is_destructive = operation.startswith("delete")
            action_desc = f"{operation} on {file_path}"
            whitelist_item = f"{operation}:{file_path.parent}"
            
            execute, add_to_whitelist = self._request_confirmation(
                "file_operations",
                action_desc,
                whitelist_item,
                force=is_destructive
            )
            
            if not execute:
                return f"Action cancelled by user: {action_desc}"
            
            if add_to_whitelist:
                self.whitelist_manager.add_to_whitelist("file_operations", whitelist_item)
        
        try:
            if operation == "create_file":
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content or "")
                return f"File created: {file_path}"
            
            elif operation == "read_file":
                if not file_path.exists():
                    return f"Error: File not found: {file_path}"
                if not file_path.is_file():
                    return f"Error: Not a file: {file_path}"
                content = file_path.read_text()
                # Limit output size
                if len(content) > 1000:
                    content = content[:1000] + "\n... (truncated)"
                return f"File content:\n{content}"
            
            elif operation == "delete_file":
                if not file_path.exists():
                    return f"Error: Path not found: {file_path}"
                if file_path.is_file():
                    file_path.unlink()
                    return f"File deleted: {file_path}"
                else:
                    return f"Error: Not a file: {file_path} (use delete_directory for directories)"
            
            elif operation == "create_directory":
                file_path.mkdir(parents=True, exist_ok=True)
                return f"Directory created: {file_path}"
            
            elif operation == "list_directory":
                if not file_path.exists():
                    return f"Error: Directory not found: {file_path}"
                if not file_path.is_dir():
                    return f"Error: Not a directory: {file_path}"
                
                items = list(file_path.iterdir())
                items.sort()
                
                # Format output
                files = [f"📄 {item.name}" for item in items if item.is_file()]
                dirs = [f"📁 {item.name}" for item in items if item.is_dir()]
                all_items = dirs + files
                
                if len(all_items) > 50:
                    all_items = all_items[:50] + ["... (truncated)"]
                
                return f"Contents of {file_path}:\n" + "\n".join(all_items)
            
            else:
                return f"Error: Unknown operation: {operation}"
                
        except Exception as e:
            logger.error(f"File operation error: {e}")
            return f"Error: {str(e)}"
    
    def fetch_web_page(self, url: str) -> str:
        """
        Fetch and extract text from a web page.
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content
        """
        logger.info(f"Fetching web page: {url}")
        
        # Request confirmation for web access
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        action_desc = f"Fetch web page: {url}"
        whitelist_item = domain
        
        execute, add_to_whitelist = self._request_confirmation(
            "web_urls",
            action_desc,
            whitelist_item
        )
        
        if not execute:
            return f"Action cancelled by user: {action_desc}"
        
        if add_to_whitelist:
            self.whitelist_manager.add_to_whitelist("web_urls", whitelist_item)
        
        # Try with retry logic if enabled
        if settings.enable_retry_logic:
            try:
                return self._fetch_with_retry(url)
            except RetryExhaustedError as e:
                logger.error(f"All web fetch retry attempts failed: {e}")
                return f"Error: Unable to fetch {url} after multiple attempts"
        else:
            return self._fetch_once(url)
    
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        backoff_factor=2.0,
        exceptions=(requests.RequestException, ConnectionError, TimeoutError)
    )
    def _fetch_with_retry(self, url: str) -> str:
        """Internal method with retry decorator for transient network errors."""
        return self._fetch_once(url)
    
    def _fetch_once(self, url: str) -> str:
        """Single fetch attempt without retry."""
        try:
            # Add timeout and user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit size
            if len(text) > 2000:
                text = text[:2000] + "\n... (truncated)"
            
            logger.info(f"Fetched {len(text)} characters from {url}")
            return f"Content from {url}:\n{text}"
            
        except Exception as e:
            logger.error(f"Web fetch error: {e}")
            return f"Error fetching web page: {str(e)}"
    
    def launch_application(self, application: str, args: Optional[List[str]] = None) -> str:
        """
        Launch an application or execute a whitelisted command.
        
        Args:
            application: Application name or command
            args: Optional command arguments
            
        Returns:
            Result message
        """
        logger.info(f"Launching application: {application}")
        
        # Check if command is whitelisted
        base_command = application.split()[0] if ' ' in application else application
        
        if base_command not in self.command_whitelist:
            return f"Error: Command '{base_command}' is not in whitelist"
        
        # Request confirmation (always for dangerous commands)
        cmd_with_args = f"{application} {' '.join(args)}" if args else application
        action_desc = f"Launch: {cmd_with_args}"
        whitelist_item = cmd_with_args
        is_dangerous = base_command in DANGEROUS_COMMANDS
        
        if is_dangerous:
            action_desc = f"⚠ DANGEROUS: {cmd_with_args}"
        
        execute, add_to_whitelist = self._request_confirmation(
            "applications",
            action_desc,
            whitelist_item,
            force=is_dangerous
        )
        
        if not execute:
            return f"Action cancelled by user: {action_desc}"
        
        if add_to_whitelist:
            self.whitelist_manager.add_to_whitelist("applications", whitelist_item)
        
        try:
            # Build command safely
            cmd = [application]
            if args:
                cmd.extend(args)
            
            # Execute in background for GUI applications
            if base_command in ['code', 'firefox', 'nautilus', 'gedit']:
                # Launch and detach
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                return f"Launched {application}"
            else:
                # Execute and capture output for CLI commands
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                output = result.stdout.strip()
                if result.stderr:
                    output += f"\nErrors: {result.stderr.strip()}"
                
                # Limit output
                if len(output) > 1000:
                    output = output[:1000] + "\n... (truncated)"
                
                return f"Command output:\n{output}" if output else f"Command executed: {application}"
                
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out: {application}"
        except Exception as e:
            logger.error(f"Application launch error: {e}")
            return f"Error launching application: {str(e)}"
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """
        Execute a tool call from the LLM.
        
        Args:
            tool_call: Tool call dict with function name and arguments
            
        Returns:
            Execution result
        """
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        arguments = function.get("arguments", {})
        
        logger.info(f"Executing tool: {function_name}")
        
        try:
            if function_name == "execute_file_operation":
                return self.execute_file_operation(
                    operation=arguments.get("operation"),
                    path=arguments.get("path"),
                    content=arguments.get("content")
                )
            
            elif function_name == "fetch_web_page":
                return self.fetch_web_page(arguments.get("url"))
            
            elif function_name == "launch_application":
                return self.launch_application(
                    application=arguments.get("application"),
                    args=arguments.get("args")
                )
            
            else:
                return f"Error: Unknown tool: {function_name}"
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing tool: {str(e)}"
