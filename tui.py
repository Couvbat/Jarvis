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
        help_text.append("exit/quit/goodbye", style="cyan")
        help_text.append(" | ", style="dim")
        help_text.append("switch to french/english", style="cyan")
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
    
    def prompt_confirmation(self, decision) -> tuple[bool, bool]:
        """
        Ask the user about one tool call.

        Args:
            decision: The policy engine's Decision, carrying the real
                arguments, the risk and why the question is being asked

        Returns:
            Tuple of (go ahead, remember this for next time)
        """
        may_remember = decision.surface.value != "terminal"

        if self.live:
            self.live.stop()

        self.console.print()
        heading = (
            "[bold red]⚠️  Confirm at the keyboard[/bold red]"
            if not may_remember
            else "[bold yellow]⚠️  Confirmation required[/bold yellow]"
        )
        self.console.print(Panel(
            f"{heading}\n\n"
            f"[cyan]{decision.summary}[/cyan]\n\n"
            f"Risk: [bold]{decision.risk.name}[/bold]\n"
            f"Why ask: {decision.reason}",
            border_style="red" if not may_remember else "yellow",
            padding=(1, 2)
        ))

        self.console.print("[cyan]Options:[/cyan]")
        self.console.print("  [green]y[/green] - do it once")
        if may_remember:
            self.console.print(
                f"  [green]a[/green] - do it and [bold]stop asking[/bold] "
                f"for {decision.scope}"
            )
        self.console.print("  [red]n[/red] - don't")
        self.console.print()

        choices = "y/a/n" if may_remember else "y/n"
        while True:
            choice = self.console.input(
                f"[bold]Your choice[/bold] [dim]({choices})[/dim]: "
            ).lower().strip()

            if choice == 'y':
                result = (True, False)
                break
            if choice == 'a' and may_remember:
                result = (True, True)
                self.console.print(f"[green]✓[/green] Remembered for {decision.scope}")
                break
            if choice == 'n':
                result = (False, False)
                self.console.print("[red]✗[/red] Cancelled")
                break
            self.console.print(f"[red]Please answer {choices}.[/red]")

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
