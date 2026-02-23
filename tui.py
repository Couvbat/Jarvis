"""Terminal User Interface for Jarvis using Rich."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich import box
from loguru import logger


class JarvisTUI:
    """Terminal UI for displaying chat history and actions."""
    
    def __init__(self):
        self.console = Console()
        self.chat_history: List[Dict[str, Any]] = []
        self.actions_log: List[Dict[str, Any]] = []
        self.current_status = "Initializing..."
        self.current_language = "en"
        self.live = None
        
    def _make_header(self) -> Panel:
        """Create the header panel."""
        header_text = Text()
        header_text.append("🤖 ", style="bold cyan")
        header_text.append("JARVIS", style="bold white")
        header_text.append(" - Local Voice Assistant", style="dim white")
        
        info = Text()
        info.append(f" Language: ", style="dim")
        info.append(f"{self.current_language.upper()}", style="bold yellow")
        info.append(f" | Status: ", style="dim")
        info.append(self.current_status, style="bold green")
        
        content = Group(
            Align.center(header_text),
            Align.center(info)
        )
        
        return Panel(
            content,
            box=box.DOUBLE,
            style="cyan",
            padding=(0, 1)
        )
    
    def _make_chat_panel(self) -> Panel:
        """Create the chat history panel."""
        if not self.chat_history:
            content = Text("No conversation yet...", style="dim italic")
        else:
            content = Group()
            # Show last 10 messages
            messages = self.chat_history[-10:]
            
            for msg in messages:
                role = msg.get("role", "unknown")
                text = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                
                if role == "user":
                    line = Text()
                    line.append(f"[{timestamp}] ", style="dim")
                    line.append("👤 You: ", style="bold blue")
                    line.append(text, style="white")
                    content.renderables.append(line)
                    
                elif role == "assistant":
                    line = Text()
                    line.append(f"[{timestamp}] ", style="dim")
                    line.append("🤖 Jarvis: ", style="bold green")
                    line.append(text, style="white")
                    content.renderables.append(line)
                    
                elif role == "system":
                    line = Text()
                    line.append(f"[{timestamp}] ", style="dim")
                    line.append("⚙️  ", style="yellow")
                    line.append(text, style="yellow italic")
                    content.renderables.append(line)
                
                # Add spacing between messages
                content.renderables.append(Text(""))
        
        return Panel(
            content,
            title="💬 Conversation",
            border_style="blue",
            padding=(1, 2),
            height=20
        )
    
    def _make_actions_panel(self) -> Panel:
        """Create the actions log panel."""
        if not self.actions_log:
            content = Text("No actions performed yet...", style="dim italic")
        else:
            table = Table(
                show_header=True,
                header_style="bold magenta",
                box=box.SIMPLE,
                padding=(0, 1)
            )
            
            table.add_column("Time", style="dim", width=8)
            table.add_column("Action", style="cyan", width=20)
            table.add_column("Details", style="white")
            
            # Show last 8 actions
            for action in self.actions_log[-8:]:
                timestamp = action.get("timestamp", "")
                action_type = action.get("action", "")
                details = action.get("details", "")
                status = action.get("status", "")
                
                # Truncate long details
                if len(details) > 50:
                    details = details[:47] + "..."
                
                # Color code by status
                if status == "success":
                    action_style = "green"
                elif status == "error":
                    action_style = "red"
                else:
                    action_style = "yellow"
                
                table.add_row(
                    timestamp,
                    Text(action_type, style=action_style),
                    details
                )
            
            content = table
        
        return Panel(
            content,
            title="⚡ Actions & Tools",
            border_style="magenta",
            padding=(1, 1),
            height=12
        )
    
    def _make_help_panel(self) -> Panel:
        """Create the help/commands panel."""
        help_text = Text()
        help_text.append("Commands: ", style="bold")
        help_text.append("/options", style="cyan")
        help_text.append(" | ", style="dim")
        help_text.append("exit/quit/goodbye", style="cyan")
        help_text.append(" | ", style="dim")
        help_text.append("switch language", style="cyan")
        help_text.append(" | ", style="dim")
        help_text.append("Ctrl+C to stop", style="red")
        
        return Panel(
            Align.center(help_text),
            border_style="dim",
            padding=(0, 1)
        )
    
    def _make_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="help", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="chat", ratio=2),
            Layout(name="actions", ratio=1)
        )
        
        return layout
    
    def _update_layout(self, layout: Layout):
        """Update the layout with current data."""
        layout["header"].update(self._make_header())
        layout["chat"].update(self._make_chat_panel())
        layout["actions"].update(self._make_actions_panel())
        layout["help"].update(self._make_help_panel())
    
    def start(self):
        """Start the live TUI."""
        layout = self._make_layout()
        self._update_layout(layout)
        
        self.live = Live(
            layout,
            console=self.console,
            refresh_per_second=4,
            screen=True
        )
        self.live.start()
    
    def stop(self):
        """Stop the live TUI."""
        if self.live:
            self.live.stop()
    
    def refresh(self):
        """Manually refresh the display."""
        if self.live:
            layout = self._make_layout()
            self._update_layout(layout)
            self.live.update(layout)
    
    def update_status(self, status: str):
        """Update the current status."""
        self.current_status = status
        self.refresh()
    
    def update_language(self, language: str):
        """Update the current language."""
        self.current_language = language
        self.refresh()
    
    def add_user_message(self, message: str):
        """Add a user message to chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append({
            "role": "user",
            "content": message,
            "timestamp": timestamp
        })
        self.refresh()
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append({
            "role": "assistant",
            "content": message,
            "timestamp": timestamp
        })
        self.refresh()
    
    def add_system_message(self, message: str):
        """Add a system message to chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_history.append({
            "role": "system",
            "content": message,
            "timestamp": timestamp
        })
        self.refresh()
    
    def add_action(self, action_type: str, details: str, status: str = "info"):
        """
        Add an action to the actions log.
        
        Args:
            action_type: Type of action (e.g., "File Operation", "Web Fetch")
            details: Action details
            status: Status - "success", "error", or "info"
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.actions_log.append({
            "action": action_type,
            "details": details,
            "status": status,
            "timestamp": timestamp
        })
        self.refresh()
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        self.refresh()
    
    def clear_actions(self):
        """Clear actions log."""
        self.actions_log = []
        self.refresh()
    
    def prompt_confirmation(self, action_description: str, item: str) -> tuple[bool, bool]:
        """
        Prompt user for action confirmation.
        
        Args:
            action_description: Description of the action
            item: Item to potentially whitelist
            
        Returns:
            Tuple of (execute_action, add_to_whitelist)
        """
        if self.live:
            self.live.stop()
        
        self.console.print()
        self.console.print(Panel(
            f"[bold yellow]⚠️  Confirmation Required[/bold yellow]\n\n"
            f"Action: [cyan]{action_description}[/cyan]\n\n"
            f"Do you want to proceed?",
            border_style="yellow",
            padding=(1, 2)
        ))
        
        self.console.print("[cyan]Options:[/cyan]")
        self.console.print("  [green]y[/green] - Execute this action")
        self.console.print("  [green]a[/green] - Execute and [bold]add to whitelist[/bold]")
        self.console.print("  [red]n[/red] - Cancel this action")
        self.console.print()
        
        while True:
            choice = self.console.input("[bold]Your choice[/bold] [dim](y/a/n)[/dim]: ").lower().strip()
            
            if choice == 'y':
                result = (True, False)
                break
            elif choice == 'a':
                result = (True, True)
                self.console.print(f"[green]✓[/green] Added to whitelist: {item}")
                break
            elif choice == 'n':
                result = (False, False)
                self.console.print("[red]✗[/red] Action cancelled")
                break
            else:
                self.console.print("[red]Invalid choice. Please enter y, a, or n.[/red]")
        
        self.console.print()
        
        if self.live:
            self.live.start()
        
        return result
    
    def show_welcome(self):
        """Show welcome screen."""
        self.console.clear()
        
        welcome = Text()
        welcome.append("\n\n")
        welcome.append("  🤖 ", style="bold cyan")
        welcome.append("JARVIS", style="bold white")
        welcome.append(" - Local Voice Assistant\n\n", style="dim white")
        welcome.append("  Starting up...\n", style="yellow")
        
        self.console.print(Panel(
            Align.center(welcome),
            box=box.DOUBLE,
            border_style="cyan",
            padding=(2, 4)
        ))
    
    def show_options_menu(self, current_settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Show options menu and return updated settings.
        
        Args:
            current_settings: Dictionary of current settings
            
        Returns:
            Dictionary of updated settings or None if cancelled
        """
        if self.live:
            self.live.stop()
        
        self.console.clear()
        self.console.print()
        
        # Display header
        header = Text()
        header.append("⚙️  ", style="bold yellow")
        header.append("JARVIS OPTIONS", style="bold white")
        
        self.console.print(Panel(
            Align.center(header),
            border_style="yellow",
            box=box.DOUBLE
        ))
        
        while True:
            self.console.print()
            
            # Current settings display
            settings_table = Table(
                show_header=False,
                box=box.SIMPLE,
                padding=(0, 2)
            )
            settings_table.add_column("Setting", style="cyan", width=30)
            settings_table.add_column("Value", style="white")
            
            settings_table.add_row(
                "Language",
                current_settings.get('language', 'en').upper()
            )
            settings_table.add_row(
                "Wake Word Detection",
                "Enabled" if current_settings.get('wake_word_enabled', False) else "Disabled"
            )
            settings_table.add_row(
                "Conversation Persistence",
                "Enabled" if current_settings.get('persistence_enabled', False) else "Disabled"
            )
            settings_table.add_row(
                "Retry/Fallback Logic",
                "Enabled" if current_settings.get('retry_enabled', True) else "Disabled"
            )
            
            self.console.print(Panel(
                settings_table,
                title="Current Settings",
                border_style="blue"
            ))
            
            # Menu options
            self.console.print("\n[bold]Available Options:[/bold]\n")
            self.console.print("  [cyan]1[/cyan] - Switch Language (EN ↔ FR)")
            self.console.print("  [cyan]2[/cyan] - Clear Conversation History")
            self.console.print("  [cyan]3[/cyan] - Clear Actions Log")
            self.console.print("  [cyan]4[/cyan] - Save Current Session")
            self.console.print("  [cyan]5[/cyan] - View System Info")
            self.console.print("  [cyan]0[/cyan] - Back to Chat")
            self.console.print()
            
            choice = self.console.input("[bold]Select option[/bold] [dim](0-5)[/dim]: ").strip()
            
            if choice == '0':
                # Return to chat
                break
            elif choice == '1':
                # Toggle language
                current_lang = current_settings.get('language', 'en')
                new_lang = 'fr' if current_lang == 'en' else 'en'
                current_settings['language'] = new_lang
                self.current_language = new_lang
                self.console.print(f"\n[green]✓[/green] Language changed to: {new_lang.upper()}")
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
            elif choice == '2':
                # Clear conversation
                confirm = self.console.input("\n[yellow]Clear conversation history? (y/n):[/yellow] ").lower()
                if confirm == 'y':
                    self.clear_history()
                    current_settings['clear_conversation'] = True
                    self.console.print("[green]✓[/green] Conversation history cleared")
                else:
                    self.console.print("[dim]Cancelled[/dim]")
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
            elif choice == '3':
                # Clear actions
                self.clear_actions()
                self.console.print("\n[green]✓[/green] Actions log cleared")
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
            elif choice == '4':
                # Save session
                current_settings['save_session'] = True
                self.console.print("\n[green]✓[/green] Session will be saved")
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
            elif choice == '5':
                # Show system info
                self._show_system_info(current_settings)
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
            else:
                self.console.print("[red]Invalid option. Please choose 0-5.[/red]")
                self.console.input("\n[dim]Press Enter to continue...[/dim]")
                self.console.clear()
                self.console.print()
        
        if self.live:
            self.live.start()
        
        return current_settings
    
    def _show_system_info(self, settings: Dict[str, Any]):
        """Display system information."""
        self.console.print()
        
        info_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Status", style="white")
        
        # Add component statuses
        info_table.add_row("Ollama LLM", "[green]✓[/green] Connected" if settings.get('ollama_healthy', True) else "[red]✗[/red] Disconnected")
        info_table.add_row("Whisper STT", "[green]✓[/green] Loaded" if settings.get('stt_loaded', True) else "[yellow]⚠[/yellow] Not loaded")
        info_table.add_row("Piper TTS", "[green]✓[/green] Loaded" if settings.get('tts_loaded', True) else "[yellow]⚠[/yellow] Not loaded")
        info_table.add_row("Audio Device", "[green]✓[/green] Available" if settings.get('audio_available', True) else "[red]✗[/red] Not available")
        
        if settings.get('wake_word_enabled', False):
            info_table.add_row("Wake Word", "[green]✓[/green] Active")
        
        self.console.print(Panel(
            info_table,
            title="System Status",
            border_style="magenta"
        ))
