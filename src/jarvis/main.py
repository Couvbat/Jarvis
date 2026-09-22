"""Main orchestration loop for Jarvis voice assistant."""

import argparse
import asyncio
import sys
import threading
import time

from loguru import logger

from jarvis.audio_handler import AudioHandler
from jarvis.config import settings
from jarvis.conversation_store import ConversationStore
from jarvis.llm_module import LLMModule
from jarvis.policy.engine import PolicyEngine, Surface
from jarvis.policy.store import ApprovalStore
from jarvis.policy.taint import TaintState
from jarvis.speech.barge_in import VAD_FRAME_MS, BargeInListener
from jarvis.speech.chunker import SentenceChunker
from jarvis.speech.pipeline import SpeechPipeline
from jarvis.speech.wake import WakeWord, WakeWordUnavailable
from jarvis.stt_module import STTModule
from jarvis.text_utils import normalise
from jarvis.tools.builtin import attach_mcp_tools, build_default_registry, build_mcp_manager
from jarvis.tools.schema import ToolResult
from jarvis.tts_module import TTSModule
from jarvis.tui import JarvisTUI

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


#: Kept as a module-level name: the exit-command tests read it directly.
normalise_utterance = normalise


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

    def __init__(
        self,
        use_tui: bool = False,
        speak_aloud: bool = True,
        resume: bool = False,
    ):
        self.use_tui = use_tui
        self.speak_aloud = speak_aloud
        self.resume = resume
        self.tui = None

        if self.use_tui:
            self.tui = JarvisTUI()
            self.tui.show_welcome()

        logger.info("Initializing Jarvis...")

        # Tools and the policy that gates them
        self.registry = build_default_registry()
        self.mcp = build_mcp_manager()
        self.policy = PolicyEngine(ApprovalStore(settings.approvals_path))
        self.taint = TaintState()

        # Initialize components
        self.audio = AudioHandler()
        self.stt = STTModule()
        self.conversations = (
            ConversationStore(settings.conversations_path)
            if settings.conversation_history else None
        )
        self.llm = LLMModule(self.registry, store=self.conversations)
        self.tts = TTSModule()
        self.speech = SpeechPipeline(self.tts, self.audio, on_fallback=self._show)
        #: Audio captured by an interruption, to start the next turn with.
        self._carried_audio = None
        self.wake = self._build_wake_word()
        #: Until this moment, a follow-up does not need the wake word again.
        self._awake_until = 0.0
        #: Waiting for the wake word can last all afternoon, in a thread that
        #: Ctrl-C cannot reach. The interpreter joins that thread on the way
        #: out, so without a way to tell it to stop, a woken-up session simply
        #: never exits. aclose() sets this; the listener checks it.
        self._shutdown = threading.Event()

        logger.info("Jarvis initialized and ready!")

        if self.use_tui:
            self.tui.update_status("Ready")
            self.tui.update_language(settings.whisper_language)

    def _build_wake_word(self) -> WakeWord | None:
        """The wake word detector, or None when Jarvis just listens.

        A missing dependency or model is reported once and then ignored: an
        assistant that refuses to start because it cannot do hands-free is
        worse than one that listens the way it always has.
        """
        if not settings.wake_word:
            return None

        try:
            detector = WakeWord(
                model=settings.wake_word_model,
                threshold=settings.wake_word_threshold,
                sample_rate=settings.sample_rate,
            )
        except WakeWordUnavailable as e:
            message = f"Wake word disabled: {e}"
            logger.warning(message)
            if self.use_tui:
                self.tui.add_system_message(message)
            return None

        logger.info(f"Listening for the {detector.describe()}")
        return detector

    def _needs_wake_word(self) -> bool:
        """Whether this turn has to be woken, or follows one that was."""
        if self.wake is None:
            return False
        if self._carried_audio is not None:
            # An interruption is already the user speaking; making them say
            # the phrase to finish a sentence they started would be absurd.
            return False
        return time.monotonic() >= self._awake_until

    def _stay_awake(self) -> None:
        """Open the window in which a follow-up needs no wake word."""
        if self.wake is not None:
            self._awake_until = time.monotonic() + settings.wake_word_follow_up

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

    async def start(self):
        """Bring up anything that needs the event loop.

        MCP servers are subprocesses or network sessions, so they cannot be
        started from __init__; and a server that will not come up must not
        stop the assistant from running without it.
        """
        if self.speak_aloud:
            # Whisper and Piper are only needed by a session that listens and
            # speaks. Text mode used to download and load both before its
            # first prompt.
            logger.info("Loading models (this may take a moment)...")
            await asyncio.to_thread(self.stt.initialize)
            await asyncio.to_thread(self.tts.initialize)
            await self.speech.start()

        await self._check_llm()

        if self.resume:
            restored = self.llm.resume()
            message = (
                f"Resumed {restored} turn(s) from the last conversation"
                if restored else "No earlier conversation to resume"
            )
            logger.info(message)
            if self.use_tui:
                self.tui.add_system_message(message)

        try:
            await attach_mcp_tools(self.registry, self.mcp)
        except Exception as e:
            logger.error(f"MCP startup failed: {e}")

        if self.use_tui and self.mcp.servers:
            for name, status in self.mcp.statuses().items():
                self.tui.add_system_message(f"MCP {name}: {status}")

    async def aclose(self):
        """Shut down the speech pipeline, the MCP servers and the log."""
        # First, so a listener blocked on the microphone stops before the
        # interpreter tries to join its thread.
        self._shutdown.set()
        try:
            self.llm.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Could not close the conversation log: {e}")
        try:
            await self.speech.stop()
        except Exception as e:  # pragma: no cover - reported by the pipeline
            logger.error(f"Speech shutdown failed: {e}")
        try:
            await self.mcp.stop()
        except Exception as e:  # pragma: no cover - reported by the manager
            logger.error(f"MCP shutdown failed: {e}")

    async def _check_llm(self) -> None:
        """Say plainly which provider is answering, and which are not.

        With more than one configured, "Ollama is not running" is no longer
        the whole story: the useful thing to know at startup is which model
        will actually reply, and why the preferred one did not.
        """
        provider, report = await self.llm.status()

        if provider is not None:
            message = f"LLM: {provider.describe()}"
            others = [
                f"{name} ({detail})"
                for name, detail in report.items()
                if name != provider.name
            ]
            if others:
                message += f" - also configured: {', '.join(others)}"
            logger.info(message)
            if self.use_tui:
                self.tui.add_system_message(message)
            return

        detail = ", ".join(f"{name}: {state}" for name, state in report.items())
        message = (
            f"No LLM provider is usable ({detail}). "
            f"Start Ollama with: ollama serve, and pull the model with: "
            f"ollama pull {settings.ollama_model}"
        )
        logger.error(message)
        if self.use_tui:
            self.tui.add_system_message(message)
        else:
            print(message)

    def _show(self, text: str) -> None:
        """Put text in front of the user when it cannot be spoken."""
        print(f"Jarvis: {text}")

    def _emit(self, text: str) -> None:
        """Send a finished sentence wherever this session puts them."""
        if not text.strip():
            return
        if self.speak_aloud:
            self.speech.say(text)
        else:
            print(text, end=" ", flush=True)

    async def _finish_speaking(self) -> bool:
        """Wait for the answer to finish, unless the user talks over it.

        Returns whether they did. The words that interrupted are kept for the
        turn they begin: losing the start of a sentence would make
        interrupting worse than waiting.
        """
        if not self.speak_aloud:
            await self.speech.drain()
            return False

        if not settings.barge_in:
            await self.speech.drain()
            return False

        listener = BargeInListener(
            self.audio,
            min_speech_frames=max(
                1, settings.barge_in_min_speech_ms // VAD_FRAME_MS
            ),
        )
        await listener.start()

        drained = asyncio.create_task(self.speech.drain())
        interrupted = asyncio.create_task(listener.detected.wait())
        done, pending = await asyncio.wait(
            {drained, interrupted}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        captured = await listener.stop()

        if interrupted in done:
            logger.info("Interrupted by the user")
            await self.speech.interrupt()
            self._carried_audio = captured
            if self.use_tui:
                self.tui.add_system_message("Interrupted")
            return True

        return False

    async def _speak(self, text: str) -> None:
        """Say one fixed phrase and wait for it, e.g. a goodbye."""
        self.speech.resume()
        self._emit(text)
        if self.speak_aloud:
            await self.speech.drain()
        else:
            print()

    async def handle_language_switch(self, user_text: str) -> bool:
        """Switch language if that is what was asked. Returns whether it was.

        Shared by both run modes: the voice loop had this inline, so a text
        session could not change language at all.
        """
        phrase = normalise(user_text)

        if any(cue in phrase for cue in ("switch to french", "parle francais",
                                         "en francais", "in french")):
            language, reply = "fr", "D'accord, je passe au français."
        elif any(cue in phrase for cue in ("switch to english", "parle anglais",
                                           "in english", "en anglais")):
            language, reply = "en", "Okay, switching to English."
        else:
            return False

        self.stt.set_language(language)
        logger.info(f"Language switched to {language}")

        if self.use_tui:
            self.tui.update_language(language)
            self.tui.add_assistant_message(reply)
            self.tui.add_system_message(f"Language changed to {language}")

        await self._speak(reply)
        return True

    async def process_user_input(self, user_text: str) -> str:
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
        self.speech.resume()

        chunker = SentenceChunker()

        def on_text(fragment: str) -> None:
            for sentence in chunker.feed(fragment):
                self._emit(sentence)

        def flush() -> None:
            tail = chunker.flush()
            if tail:
                self._emit(tail)

        result = await self.llm.chat(user_text, on_text=on_text)
        flush()

        for _ in range(self.llm.MAX_TOOL_ITERATIONS):
            tool_calls = result["tool_calls"]
            if not tool_calls:
                break

            # Whatever the model said before asking for tools has been queued;
            # let it finish so the user is not talked over by the next round.
            if self.speak_aloud:
                await self.speech.drain()

            logger.info(f"Executing {len(tool_calls)} tool call(s)")
            if self.use_tui:
                self.tui.update_status(f"Executing {len(tool_calls)} action(s)...")

            for tool_call in tool_calls:
                await self._execute_tool_call(tool_call)

            if self.use_tui:
                self.tui.update_status("Generating response...")

            # No synthetic user turn here: the tool results are the new
            # information, and the follow-up may legitimately ask for more
            # tools, which is why this is a loop rather than one extra call.
            result = await self.llm.continue_after_tools(on_text=on_text)
            flush()
        else:
            if result["tool_calls"]:
                logger.warning(
                    f"Stopped after {self.llm.MAX_TOOL_ITERATIONS} tool rounds"
                )

        return result["response"]

    async def _execute_tool_call(self, tool_call: dict) -> ToolResult:
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
                # The prompt reads from stdin; keep it off the event loop so a
                # pending confirmation cannot block everything else.
                proceed, remember = await asyncio.to_thread(self._confirm, decision)
                if not proceed:
                    result = ToolResult.error("the user declined this action")
                else:
                    if remember:
                        self.policy.remember(decision)
                    result = await self.registry.call(name, arguments)
            else:
                logger.info(f"Auto-approved {name}: {decision.reason}")
                result = await self.registry.call(name, arguments)

        # Record before the model sees it: a result from outside the machine
        # raises the bar for whatever it asks for next.
        self.taint.observe(name, result)

        if self.use_tui:
            self.tui.add_action(
                name, result.content[:80], "success" if result.ok else "error"
            )

        self.llm.add_tool_result(name, result.content, untrusted=result.untrusted)
        return result

    async def run_interactive(self):
        """Run in interactive voice mode."""
        # The run method decides the mode, so an instance cannot be left
        # configured to speak in a text session or stay mute in a voice one.
        self.speak_aloud = True
        await self.start()
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

                # Record audio
                try:
                    carried, self._carried_audio = self._carried_audio, None

                    if self._needs_wake_word():
                        if self.use_tui:
                            self.tui.update_status(
                                f"Waiting for '{settings.wake_word_model}'..."
                            )
                        # What follows the phrase in the same breath becomes
                        # the start of the recording, so a command does not
                        # have to be said twice.
                        carried = await asyncio.to_thread(
                            self.audio.wait_for_wake,
                            self.wake,
                            tail=settings.wake_word_tail,
                            should_stop=self._shutdown.is_set,
                        )
                        if self._shutdown.is_set():
                            break

                    if self.use_tui:
                        self.tui.update_status("Listening...")

                    audio_data = await asyncio.to_thread(
                        self.audio.record_until_silence,
                        silence_threshold=1.5,
                        max_duration=30.0,
                        prefix=carried,
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
                    user_text = await asyncio.to_thread(
                        self.stt.transcribe, audio_data, self.audio.sample_rate
                    )

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

                if await self.handle_language_switch(user_text):
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
                    await self._speak(response_text)

                    break

                # Process with LLM and tools
                try:
                    response_text = await self.process_user_input(user_text)

                    if not response_text:
                        response_text = "I'm not sure how to respond to that."

                    logger.info(f"Jarvis: {response_text}")

                    if self.use_tui:
                        self.tui.add_assistant_message(response_text)

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    await self._speak("I encountered an error processing your request.")

                # The answer was spoken sentence by sentence as it arrived;
                # wait for the tail before listening again - unless the user
                # talks over it, which ends this turn and starts the next.
                if self.use_tui:
                    self.tui.update_status("Speaking...")
                await self._finish_speaking()

                # The window starts once Jarvis has stopped talking, not when
                # the answer was generated: it is there so a follow-up can be
                # spoken, and it cannot be spoken over the answer.
                self._stay_awake()

                if self.use_tui:
                    self.tui.update_status("Ready")

        except KeyboardInterrupt:
            logger.info("\n\nShutting down Jarvis...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            await self.aclose()
            if self.use_tui:
                self.tui.stop()

    async def run_text_mode(self):
        """Run in text-only mode (no voice I/O)."""
        self.speak_aloud = False
        await self.start()
        logger.info("\n" + "="*50)
        logger.info("Jarvis Voice Assistant - Text Mode")
        logger.info("Type 'exit' to quit")
        logger.info("="*50 + "\n")

        try:
            while True:
                # Get text input
                try:
                    user_text = (await asyncio.to_thread(input, "\nYou: ")).strip()

                    if not user_text:
                        continue

                    if is_exit_command(user_text):
                        print("Jarvis: Goodbye!")
                        break

                    if await self.handle_language_switch(user_text):
                        continue

                except KeyboardInterrupt:
                    print("\n\nJarvis: Goodbye!")
                    break

                # Process
                try:
                    print("\nJarvis: ", end="", flush=True)
                    await self.process_user_input(user_text)
                    print()

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    print(f"\nJarvis: I encountered an error: {str(e)}")

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            await self.aclose()


def parse_arguments(argv=None):
    """Read the command line.

    argparse rather than a hand-rolled chain, so an unknown flag is an error
    instead of being ignored: "--txt" used to start a voice session with the
    microphone open, which is not what the user asked for.
    """
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="A local voice assistant.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--text", action="store_true",
        help="text-only mode, with no audio input or output",
    )
    mode.add_argument(
        "--tui", action="store_true",
        help="voice mode with the terminal interface",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="carry on from where the last conversation left off",
    )
    return parser.parse_args(argv)


def configure_logging(use_tui: bool) -> None:
    """Send logs somewhere that will not fight with the interface."""
    logger.remove()
    if use_tui:
        # The TUI owns the terminal; logs go to a file or they corrupt it.
        logger.add(
            "jarvis.log",
            format="{time:HH:mm:ss} | {level: <8} | {message}",
            level=settings.log_level,
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> "
                   "| <level>{message}</level>",
            level=settings.log_level,
        )


def main():
    """Main entry point."""
    arguments = parse_arguments()
    configure_logging(arguments.tui)

    try:
        jarvis = Jarvis(use_tui=arguments.tui, resume=arguments.resume)

        if arguments.text:
            asyncio.run(jarvis.run_text_mode())
        else:
            asyncio.run(jarvis.run_interactive())

    except Exception as e:
        logger.error(f"Failed to start Jarvis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
