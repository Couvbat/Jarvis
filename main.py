"""Main orchestration loop for Jarvis voice assistant."""

import sys
from datetime import datetime
from loguru import logger
from config import settings
from audio_handler import AudioHandler
from stt_module import STTModule
from llm_module import LLMModule
from action_executor import ActionExecutor
from tts_module import TTSModule
from tui import JarvisTUI
from error_recovery import ServiceHealthChecker, degraded_mode


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
        
        # Initialize LLM with persistence if enabled
        enable_persistence = getattr(settings, 'enable_conversation_persistence', False)
        self.llm = LLMModule(enable_persistence=enable_persistence)
        
        self.executor = ActionExecutor(confirmation_callback=self._confirmation_callback)
        self.tts = TTSModule()
        
        # Generate session ID
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.llm.history.set_session_id(self.session_id)
        
        # Load previous context if enabled
        if enable_persistence and getattr(settings, 'load_previous_context', False):
            logger.info("Loading previous conversation context...")
            if self.llm.history.load_last_session():
                if self.use_tui:
                    self.tui.add_system_message("✓ Loaded previous conversation")
            else:
                logger.info("No previous session found or failed to load")
        
        # Load models
        logger.info("Loading models (this may take a moment)...")
        self.stt.initialize()
        self.tts.initialize()
        
        # Initialize wake word if enabled
        if getattr(settings, 'enable_wake_word', False):
            logger.info("Initializing wake word detection...")
            access_key = getattr(settings, 'porcupine_access_key', None)
            keyword = getattr(settings, 'wake_word_keyword', 'jarvis')
            sensitivity = getattr(settings, 'wake_word_sensitivity', 0.5)
            
            try:
                self.audio.initialize_wake_word(
                    access_key=access_key,
                    keyword=keyword,
                    sensitivity=sensitivity
                )
                if self.use_tui:
                    self.tui.add_system_message(f"✓ Wake word '{keyword}' enabled")
            except Exception as e:
                logger.error(f"Failed to initialize wake word: {e}")
                if self.use_tui:
                    self.tui.add_system_message("⚠ Wake word detection unavailable")
        
        # Perform health checks
        self._perform_health_checks()
        
        logger.info("Jarvis initialized and ready!")
        
        if self.use_tui:
            self.tui.update_status("Ready")
            self.tui.update_language(settings.whisper_language)
    
    def _perform_health_checks(self):
        """Perform health checks on all services."""
        logger.info("Performing service health checks...")
        
        health_status = []
        
        # Check Ollama service
        if ServiceHealthChecker.check_ollama(settings.ollama_host):
            logger.info("✓ Ollama service is healthy")
            health_status.append(("Ollama", True))
        else:
            logger.warning("✗ Ollama service is not reachable")
            health_status.append(("Ollama", False))
            degraded_mode.mark_degraded('llm')
            if self.use_tui:
                self.tui.add_system_message("⚠ Warning: Ollama service unavailable - AI features degraded")
        
        # Check audio device
        if ServiceHealthChecker.check_audio_device():
            logger.info("✓ Audio devices are available")
            health_status.append(("Audio", True))
        else:
            logger.warning("✗ No audio input device found")
            health_status.append(("Audio", False))
            if self.use_tui:
                self.tui.add_system_message("⚠ Warning: No audio input device detected")
        
        # Check Piper TTS binary
        if ServiceHealthChecker.check_file_exists(str(self.tts.piper_binary)):
            logger.info("✓ Piper TTS binary found")
            health_status.append(("Piper TTS", True))
        else:
            logger.warning("✗ Piper TTS binary not found")
            health_status.append(("Piper TTS", False))
            if self.use_tui:
                self.tui.add_system_message("⚠ Warning: Piper TTS not found - will use fallback")
        
        # Log summary
        healthy_count = sum(1 for _, healthy in health_status if healthy)
        total_count = len(health_status)
        logger.info(f"Health check complete: {healthy_count}/{total_count} services healthy")
        
        if healthy_count < total_count:
            logger.warning("Some services are unavailable. Jarvis will operate in degraded mode for affected features.")

    
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
    
    def process_user_input(self, user_text: str) -> str:
        """
        Process user input through LLM and execute any tools.
        
        Args:
            user_text: User's transcribed text
            
        Returns:
            Final response text
        """
        # Get LLM response
        if self.use_tui:
            self.tui.update_status("Thinking...")
        
        result = self.llm.chat(user_text)
        response_text = result["response"]
        tool_calls = result["tool_calls"]
        
        # Execute any tool calls
        if tool_calls:
            logger.info(f"Executing {len(tool_calls)} tool call(s)")
            
            if self.use_tui:
                self.tui.update_status(f"Executing {len(tool_calls)} action(s)...")
            
            tool_results = []
            for tool_call in tool_calls:
                # Log action to TUI
                function_name = tool_call.get("function", {}).get("name", "unknown")
                arguments = tool_call.get("function", {}).get("arguments", {})
                
                # Format action details for TUI
                action_details = str(arguments)
                if len(action_details) > 100:
                    action_details = action_details[:100] + "..."
                
                if self.use_tui:
                    self.tui.add_action(function_name, action_details, "info")
                
                # Execute tool
                tool_result = self.executor.execute_tool_call(tool_call)
                tool_results.append(tool_result)
                
                # Update action status
                if self.use_tui:
                    status = "success" if "Error" not in tool_result and "cancelled" not in tool_result else "error"
                    self.tui.add_action(function_name, tool_result[:80], status)
                
                # Add result to conversation
                self.llm.add_tool_result(function_name, tool_result)
            
            # Get final response from LLM after tool execution
            if self.use_tui:
                self.tui.update_status("Generating response...")
            
            follow_up = self.llm.chat("Please provide a natural response based on the tool results.")
            response_text = follow_up["response"]
        
        if self.use_tui:
            self.tui.update_status("Speaking...")
        
        return response_text
    
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
                # Wait for wake word if enabled
                if getattr(settings, 'enable_wake_word', False) and self.audio.wake_word_detector:
                    if not self.use_tui:
                        logger.info("\n--- Waiting for wake word ---")
                    
                    if self.use_tui:
                        self.tui.update_status("Waiting for wake word...")
                    
                    wake_word_detected = self.audio.listen_for_wake_word(timeout=None)
                    
                    if not wake_word_detected:
                        continue  # Timeout or error, try again
                    
                    logger.info("Wake word detected!")
                    if self.use_tui:
                        self.tui.add_system_message("🎤 Wake word detected")
                
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
                    
                    try:
                        audio_response = self.tts.synthesize(response_text)
                        self.audio.play_audio(audio_response, 22050)
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        print(f"Jarvis: {response_text}")
                    
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
                    
                    try:
                        audio_response = self.tts.synthesize(response_text)
                        self.audio.play_audio(audio_response, 22050)
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        print(f"Jarvis: {response_text}")
                    
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
                    try:
                        audio_response = self.tts.synthesize(response_text)
                        self.audio.play_audio(audio_response, 22050)
                    except Exception as e:
                        logger.error(f"TTS error: {e}")
                        print(f"Jarvis: {response_text}")
                    
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
                try:
                    audio_response = self.tts.synthesize(response_text)
                    
                    if len(audio_response) > 0:
                        self.audio.play_audio(audio_response, 22050)
                    else:
                        # Fallback to text if TTS fails
                        print(f"Jarvis: {response_text}")
                        
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    # Fallback to text output
                    print(f"Jarvis: {response_text}")
                
                if self.use_tui:
                    self.tui.update_status("Ready")
        
        except KeyboardInterrupt:
            logger.info("\n\nShutting down Jarvis...")
            
            # Cleanup wake word detector
            if self.audio.wake_word_detector:
                self.audio.cleanup_wake_word()
            
            # Save conversation before exit
            if getattr(settings, 'enable_conversation_persistence', False):
                logger.info("Saving conversation...")
                try:
                    self.llm.history.save_to_storage(self.session_id)
                    logger.info("Conversation saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save conversation: {e}")
                    
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
        
            # Save conversation before exit
            if getattr(settings, 'enable_conversation_persistence', False):
                logger.info("Saving conversation...")
                try:
                    self.llm.history.save_to_storage(self.session_id)
                    logger.info("Conversation saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save conversation: {e}")
                    
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
