"""Main orchestration loop for Jarvis voice assistant."""

import sys
from loguru import logger
from config import settings
from audio_handler import AudioHandler
from stt_module import STTModule
from llm_module import LLMModule
from action_executor import ActionExecutor
from tts_module import TTSModule
from tui import JarvisTUI


class Jarvis:
    """Main voice assistant orchestrator."""
    
    def __init__(self, use_tui: bool = False):
        self.use_tui = use_tui
        self.tui = None
        
        if self.use_tui:
            self.tui = JarvisTUI()
            self.tui.show_welcome()
        
        logger.info("Initializing Jarvis...")
        
        # Initialize components
        self.audio = AudioHandler()
        self.stt = STTModule()
        self.llm = LLMModule()
        self.executor = ActionExecutor(confirmation_callback=self._confirmation_callback)
        self.tts = TTSModule()
        
        # Load models
        logger.info("Loading models (this may take a moment)...")
        self.stt.initialize()
        self.tts.initialize()
        
        logger.info("Jarvis initialized and ready!")
        
        if self.use_tui:
            self.tui.update_status("Ready")
            self.tui.update_language(settings.whisper_language)
    
    def _confirmation_callback(self, action_description: str, item: str) -> tuple[bool, bool]:
        """
        Handle confirmation requests for actions.
        
        Args:
            action_description: Description of the action
            item: Item to potentially whitelist
            
        Returns:
            Tuple of (execute_action, add_to_whitelist)
        """
        if self.use_tui and self.tui:
            return self.tui.prompt_confirmation(action_description, item)
        else:
            # Simple text-based confirmation
            print(f"\n[CONFIRMATION REQUIRED]")
            print(f"Action: {action_description}")
            print(f"Item: {item}")
            print("\nOptions:")
            print("  y - Execute this action")
            print("  a - Execute and add to whitelist")
            print("  n - Cancel this action")
            
            while True:
                choice = input("\nYour choice (y/a/n): ").lower().strip()
                
                if choice == 'y':
                    return (True, False)
                elif choice == 'a':
                    print(f"✓ Added to whitelist: {item}")
                    return (True, True)
                elif choice == 'n':
                    print("✗ Action cancelled")
                    return (False, False)
                else:
                    print("Invalid choice. Please enter y, a, or n.")
    
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

    def _execute_tool_call(self, tool_call: dict) -> str:
        """Execute one tool call and record it in the conversation and the UI."""
        function = tool_call.get("function", {})
        function_name = function.get("name", "unknown")
        arguments = function.get("arguments", {})

        if self.use_tui:
            action_details = str(arguments)
            if len(action_details) > 100:
                action_details = action_details[:100] + "..."
            self.tui.add_action(function_name, action_details, "info")

        tool_result = self.executor.execute_tool_call(tool_call)

        if self.use_tui:
            failed = "Error" in tool_result or "cancelled" in tool_result
            self.tui.add_action(
                function_name, tool_result[:80], "error" if failed else "success"
            )

        self.llm.add_tool_result(function_name, tool_result)
        return tool_result

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
                if any(cmd in lower_text for cmd in ["exit", "quit", "goodbye", "stop", "au revoir", "arrête"]):
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
                    
                    if user_text.lower() in ["exit", "quit", "goodbye"]:
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
