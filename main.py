"""Main orchestration loop for Jarvis voice assistant."""

import sys
import unicodedata
from loguru import logger
from config import settings
from audio_handler import AudioHandler
from stt_module import STTModule
from llm_module import LLMModule
from policy.engine import PolicyEngine, Surface
from policy.store import ApprovalStore
from policy.taint import TaintState
from tools.builtin import build_default_registry
from tools.schema import ToolResult
from tts_module import TTSModule
from tui import JarvisTUI


#: Utterances that end the session. Matched against the whole normalised
#: utterance, never as a substring: "stop" appears in plenty of requests that
#: are not a request to quit.
EXIT_COMMANDS = frozenset({
    "exit", "quit", "stop", "goodbye", "good bye", "bye", "bye bye", "see you",
    "au revoir", "arrete", "arrete toi", "a plus",
})

#: Trailing politeness that does not change the meaning of a command.
_TRAILING_FILLER = ("s il te plait", "s il vous plait", "please", "now", "maintenant")

#: Ways of addressing the assistant, at either end of a command.
_ADDRESS = ("jarvis", "ok", "okay", "hey")


def normalise_utterance(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse whitespace.

    Whisper's output varies in accents and punctuation between runs, so
    commands are compared on this normalised form.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    unaccented = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in unaccented)
    return " ".join(cleaned.split())


def is_exit_command(text: str) -> bool:
    """True when the whole utterance is a request to stop."""
    phrase = normalise_utterance(text)

    for filler in _TRAILING_FILLER:
        if phrase.endswith(f" {filler}"):
            phrase = phrase[: -len(filler) - 1].strip()

    for address in _ADDRESS:
        if phrase.startswith(f"{address} "):
            phrase = phrase[len(address) + 1:].strip()
        if phrase.endswith(f" {address}"):
            phrase = phrase[: -len(address) - 1].strip()

    return phrase in EXIT_COMMANDS


class Jarvis:
    """Main voice assistant orchestrator."""
    
    def __init__(self, use_tui: bool = False):
        self.use_tui = use_tui
        self.tui = None
        
        if self.use_tui:
            self.tui = JarvisTUI()
            self.tui.show_welcome()
        
        logger.info("Initializing Jarvis...")
        
        # Tools and the policy that gates them
        self.registry = build_default_registry()
        self.policy = PolicyEngine(ApprovalStore(settings.approvals_path))
        self.taint = TaintState()

        # Initialize components
        self.audio = AudioHandler()
        self.stt = STTModule()
        self.llm = LLMModule(self.registry)
        self.tts = TTSModule()
        
        # Load models
        logger.info("Loading models (this may take a moment)...")
        self.stt.initialize()
        self.tts.initialize()
        
        logger.info("Jarvis initialized and ready!")
        
        if self.use_tui:
            self.tui.update_status("Ready")
            self.tui.update_language(settings.whisper_language)
    
    def _confirm(self, decision) -> tuple[bool, bool]:
        """Ask the user about one call.

        Returns (go ahead, remember this for next time).

        Spoken confirmation is not implemented yet, so a VOICE decision is
        asked at the keyboard too. The distinction is still enforced where it
        matters: a TERMINAL decision never offers "always", because those are
        the destructive and tainted calls that should be seen every time.
        """
        if self.use_tui and self.tui:
            return self.tui.prompt_confirmation(decision)

        may_remember = decision.surface is not Surface.TERMINAL

        print("\n[CONFIRMATION REQUIRED]")
        if decision.surface is Surface.TERMINAL:
            print("This one needs your keyboard, not your voice.")
        print(f"Action: {decision.summary}")
        print(f"Why ask: {decision.reason}")
        print(f"Risk:   {decision.risk.name}")
        print("\nOptions:")
        print("  y - do it once")
        if may_remember:
            print(f"  a - do it and stop asking for {decision.scope}")
        print("  n - don't")

        choices = "y/a/n" if may_remember else "y/n"
        while True:
            choice = input(f"\nYour choice ({choices}): ").lower().strip()
            if choice == "y":
                return (True, False)
            if choice == "a" and may_remember:
                return (True, True)
            if choice == "n":
                print("Cancelled")
                return (False, False)
            print(f"Please answer {choices}.")

    def _speak(self, text: str):
        """Say something out loud, falling back to the terminal if TTS fails."""
        try:
            audio, sample_rate = self.tts.synthesize(text)
            if len(audio) > 0:
                # The rate comes from the synthesiser: Piper's *-medium voices
                # are 22050 Hz but *-low voices are 16000 Hz, and assuming one
                # of them plays the other at the wrong speed.
                self.audio.play_audio(audio, sample_rate)
                return
        except Exception as e:
            logger.error(f"TTS error: {e}")

        print(f"Jarvis: {text}")

    def process_user_input(self, user_text: str) -> str:
        """
        Process user input through the LLM, executing tools until it stops
        asking for them.

        Args:
            user_text: User's transcribed text

        Returns:
            Final response text
        """
        if self.use_tui:
            self.tui.update_status("Thinking...")

        # Taint is per turn: what the assistant read a minute ago should not
        # keep prompting for the rest of the session.
        self.taint.reset()

        result = self.llm.chat(user_text)

        for _ in range(self.llm.MAX_TOOL_ITERATIONS):
            tool_calls = result["tool_calls"]
            if not tool_calls:
                break

            logger.info(f"Executing {len(tool_calls)} tool call(s)")
            if self.use_tui:
                self.tui.update_status(f"Executing {len(tool_calls)} action(s)...")

            for tool_call in tool_calls:
                self._execute_tool_call(tool_call)

            if self.use_tui:
                self.tui.update_status("Generating response...")

            # No synthetic user turn here: the tool results are the new
            # information, and the follow-up may legitimately ask for more
            # tools, which is why this is a loop rather than one extra call.
            result = self.llm.continue_after_tools()
        else:
            if result["tool_calls"]:
                logger.warning(
                    f"Stopped after {self.llm.MAX_TOOL_ITERATIONS} tool rounds"
                )

        if self.use_tui:
            self.tui.update_status("Speaking...")

        return result["response"]

    def _execute_tool_call(self, tool_call: dict) -> ToolResult:
        """Run one tool call, subject to policy, and record it everywhere."""
        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        arguments = function.get("arguments") or {}

        if self.use_tui:
            details = str(arguments)
            self.tui.add_action(name, details[:100], "info")

        spec = self.registry.get(name)
        if spec is None:
            result = ToolResult.error(f"unknown tool '{name}'")
        else:
            decision = self.policy.evaluate(spec, arguments, self.taint)

            if not decision.allowed:
                # Refused outright: the user is never asked about a call that
                # could not have run anyway.
                logger.info(f"Refused {name}: {decision.reason}")
                result = ToolResult.error(decision.reason)
            elif decision.needs_confirmation:
                proceed, remember = self._confirm(decision)
                if not proceed:
                    result = ToolResult.error("the user declined this action")
                else:
                    if remember:
                        self.policy.remember(decision)
                    result = self.registry.call(name, arguments)
            else:
                logger.info(f"Auto-approved {name}: {decision.reason}")
                result = self.registry.call(name, arguments)

        # Record before the model sees it: a result from outside the machine
        # raises the bar for whatever it asks for next.
        self.taint.observe(name, result)

        if self.use_tui:
            self.tui.add_action(
                name, result.content[:80], "success" if result.ok else "error"
            )

        self.llm.add_tool_result(name, result.content, untrusted=result.untrusted)
        return result

    def run_interactive(self):
        """Run in interactive voice mode."""
        if not self.use_tui:
            logger.info("\n" + "="*50)
            logger.info("Jarvis Voice Assistant - Interactive Mode")
            logger.info("Press Ctrl+C to exit")
            logger.info("="*50 + "\n")
        else:
            self.tui.start()
            self.tui.add_system_message("Voice assistant started. Speak your commands!")
        
        try:
            while True:
                if not self.use_tui:
                    logger.info("\n--- Ready for your command ---")
                
                if self.use_tui:
                    self.tui.update_status("Listening...")
                
                # Record audio
                try:
                    audio_data = self.audio.record_until_silence(
                        silence_threshold=1.5,
                        max_duration=30.0
                    )
                    
                    if len(audio_data) < 1000:  # Too short
                        logger.warning("Recording too short, skipping...")
                        if self.use_tui:
                            self.tui.update_status("Ready")
                        continue
                    
                except KeyboardInterrupt:
                    logger.info("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    if self.use_tui:
                        self.tui.add_system_message(f"Recording error: {e}")
                        self.tui.update_status("Ready")
                    continue
                
                # Transcribe
                if self.use_tui:
                    self.tui.update_status("Transcribing...")
                
                try:
                    user_text = self.stt.transcribe(audio_data, self.audio.sample_rate)
                    
                    if not user_text or len(user_text.strip()) < 2:
                        logger.info("No speech detected, try again...")
                        if self.use_tui:
                            self.tui.update_status("Ready")
                        continue
                    
                    logger.info(f"You said: {user_text}")
                    
                    if self.use_tui:
                        self.tui.add_user_message(user_text)
                    
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    if self.use_tui:
                        self.tui.add_system_message(f"Transcription error: {e}")
                        self.tui.update_status("Ready")
                    continue
                
                # Check for language switching commands
                lower_text = user_text.lower()
                if "switch to french" in lower_text or "parle français" in lower_text or "en français" in lower_text:
                    self.stt.set_language("fr")
                    response_text = "D'accord, je passe au français."
                    logger.info("Language switched to French")
                    
                    if self.use_tui:
                        self.tui.update_language("fr")
                        self.tui.add_assistant_message(response_text)
                        self.tui.add_system_message("Language changed to French")
                        self.tui.update_status("Speaking...")
                    
                    self._speak(response_text)
                    
                    if self.use_tui:
                        self.tui.update_status("Ready")
                    continue
                
                elif "switch to english" in lower_text or "parle anglais" in lower_text or "in english" in lower_text:
                    self.stt.set_language("en")
                    response_text = "Okay, switching to English."
                    logger.info("Language switched to English")
                    
                    if self.use_tui:
                        self.tui.update_language("en")
                        self.tui.add_assistant_message(response_text)
                        self.tui.add_system_message("Language changed to English")
                        self.tui.update_status("Speaking...")
                    
                    self._speak(response_text)
                    
                    if self.use_tui:
                        self.tui.update_status("Ready")
                    continue
                
                # Check for exit commands
                if is_exit_command(user_text):
                    logger.info("Exit command detected")
                    response_text = "Goodbye!" if self.stt.language == "en" else "Au revoir!"
                    
                    if self.use_tui:
                        self.tui.add_assistant_message(response_text)
                        self.tui.update_status("Shutting down...")
                    
                    # Speak goodbye
                    self._speak(response_text)
                    
                    break
                
                # Process with LLM and tools
                try:
                    response_text = self.process_user_input(user_text)
                    
                    if not response_text:
                        response_text = "I'm not sure how to respond to that."
                    
                    logger.info(f"Jarvis: {response_text}")
                    
                    if self.use_tui:
                        self.tui.add_assistant_message(response_text)
                    
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    response_text = "I encountered an error processing your request."
                
                # Synthesize and speak response
                self._speak(response_text)
                
                if self.use_tui:
                    self.tui.update_status("Ready")
        
        except KeyboardInterrupt:
            logger.info("\n\nShutting down Jarvis...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            if self.use_tui:
                self.tui.stop()
    
    def run_text_mode(self):
        """Run in text-only mode (no voice I/O)."""
        logger.info("\n" + "="*50)
        logger.info("Jarvis Voice Assistant - Text Mode")
        logger.info("Type 'exit' to quit")
        logger.info("="*50 + "\n")
        
        try:
            while True:
                # Get text input
                try:
                    user_text = input("\nYou: ").strip()
                    
                    if not user_text:
                        continue
                    
                    if is_exit_command(user_text):
                        print("Jarvis: Goodbye!")
                        break
                    
                except KeyboardInterrupt:
                    print("\n\nJarvis: Goodbye!")
                    break
                
                # Process
                try:
                    response_text = self.process_user_input(user_text)
                    print(f"\nJarvis: {response_text}")
                    
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    print(f"\nJarvis: I encountered an error: {str(e)}")
        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise


def main():
    """Main entry point."""
    # Check command line arguments
    mode = "voice"
    use_tui = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--text":
            mode = "text"
        elif sys.argv[1] == "--tui":
            mode = "voice"
            use_tui = True
        elif sys.argv[1] == "--help":
            print("Jarvis Voice Assistant")
            print("\nUsage:")
            print("  python main.py          # Voice mode (default)")
            print("  python main.py --tui    # Voice mode with Terminal UI")
            print("  python main.py --text   # Text-only mode")
            print("  python main.py --help   # Show this help")
            return
    
    # Configure logging
    logger.remove()  # Remove default handler
    
    # If using TUI, only log to file to avoid interfering with the UI
    if use_tui:
        logger.add(
            "jarvis.log",
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            level=settings.log_level
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level=settings.log_level
        )
    
    # Create and run assistant
    try:
        jarvis = Jarvis(use_tui=use_tui)
        
        if mode == "voice":
            jarvis.run_interactive()
        else:
            jarvis.run_text_mode()
            
    except Exception as e:
        logger.error(f"Failed to start Jarvis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
