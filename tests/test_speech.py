"""Tests for sentence chunking and the speech pipeline."""

import asyncio

import numpy as np
import pytest

from speech.chunker import SentenceChunker
from speech.pipeline import SpeechPipeline


def stream(text, size=7):
    """Deliver text the way a model does: in arbitrary slices."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def chunk_all(text, size=7, **kwargs):
    chunker = SentenceChunker(**kwargs)
    pieces = []
    for part in stream(text, size):
        pieces.extend(chunker.feed(part))
    tail = chunker.flush()
    if tail:
        pieces.append(tail)
    return pieces


class TestChunking:
    def test_sentences_come_out_separately(self):
        assert chunk_all("Bonjour. Comment vas-tu aujourd'hui ? Très bien.") == [
            "Bonjour.", "Comment vas-tu aujourd'hui ?", "Très bien.",
        ]

    def test_the_first_sentence_arrives_before_the_rest(self):
        """This is the whole point: first sound, not total time."""
        chunker = SentenceChunker()
        ready = chunker.feed("Bonjour. Je vais mainten")
        assert ready == ["Bonjour."]

    def test_nothing_is_emitted_before_a_boundary(self):
        assert SentenceChunker().feed("Je suis en train de") == []

    def test_text_is_never_lost(self):
        text = "Un. Deux ! Trois ? Et enfin la fin sans ponctuation"
        assert "".join(chunk_all(text)).replace(" ", "") == text.replace(" ", "")

    @pytest.mark.parametrize("size", [1, 3, 7, 50, 500])
    def test_the_result_does_not_depend_on_chunk_size(self, size):
        text = "Bonjour. J'ai créé le fichier demandé. Veux-tu autre chose ?"
        assert chunk_all(text, size) == chunk_all(text, 1)

    def test_flush_returns_the_tail(self):
        chunker = SentenceChunker()
        chunker.feed("Une phrase inachevée")
        assert chunker.flush() == "Une phrase inachevée"

    def test_flush_on_an_empty_buffer_returns_nothing(self):
        assert SentenceChunker().flush() is None

    def test_reset_clears_the_buffer(self):
        chunker = SentenceChunker()
        chunker.feed("Quelque chose")
        chunker.reset()
        assert chunker.flush() is None


class TestNotASentenceEnd:
    def test_decimal_numbers_stay_whole(self):
        assert chunk_all("Le fichier fait 1.5 kilo-octets au total ici.") == [
            "Le fichier fait 1.5 kilo-octets au total ici.",
        ]

    def test_domain_names_stay_whole(self):
        assert chunk_all("Va voir example.com pour en savoir plus.") == [
            "Va voir example.com pour en savoir plus.",
        ]

    @pytest.mark.parametrize("abbreviation", ["M.", "Dr.", "etc.", "Mme."])
    def test_abbreviations_do_not_split(self, abbreviation):
        """A split here puts a pause mid-name and hands the synthesiser a
        fragment that starts in the middle of one."""
        text = f"J'ai vu {abbreviation} Dupont hier soir en ville."
        assert chunk_all(text) == [text]

    def test_a_numbered_list_item_does_not_split_at_the_number(self):
        pieces = chunk_all("Trois choses : 1. du pain 2. du fromage")
        assert not any(piece.strip() in ("1.", "2.") for piece in pieces)


class TestShortPieces:
    def test_a_short_sentence_is_joined_to_the_next(self):
        """A stray "Oui." mid-answer should not become its own audio clip."""
        pieces = chunk_all("Bonjour à toi, mon ami fidèle. Oui. Voilà le résultat final.")
        assert "Oui." not in pieces

    def test_the_first_piece_is_exempt(self):
        chunker = SentenceChunker()
        assert chunker.feed("Oui. Et voici la suite de la réponse.")[0] == "Oui."

    def test_a_short_tail_still_comes_out_on_flush(self):
        assert "Oui." in " ".join(
            chunk_all("Voici une première phrase assez longue. Oui.")
        )


class TestHardBreaks:
    def test_newlines_split_whatever_the_length(self):
        assert chunk_all("Voici la liste :\n- pain\n- fromage\nVoilà.") == [
            "Voici la liste :", "- pain", "- fromage", "Voilà.",
        ]

    def test_blank_lines_are_dropped(self):
        assert chunk_all("Un titre\n\nUn paragraphe.") == ["Un titre", "Un paragraphe."]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class FakeTts:
    sample_rate = 22050

    def __init__(self, delay=0.0, fails_on=()):
        self.delay = delay
        self.fails_on = set(fails_on)
        self.synthesised = []

    def synthesize(self, text):
        import time

        if text in self.fails_on:
            raise RuntimeError("piper missing")
        if self.delay:
            time.sleep(self.delay)
        self.synthesised.append(text)
        return np.zeros(100, dtype=np.int16), self.sample_rate


class FakeAudio:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.played = []
        self.stopped = 0

    def play_audio(self, audio, sample_rate=None):
        import time

        if self.delay:
            time.sleep(self.delay)
        self.played.append(sample_rate)

    def stop_playback(self):
        self.stopped += 1


class TestPipeline:
    async def test_text_is_spoken(self):
        tts, audio = FakeTts(), FakeAudio()
        async with SpeechPipeline(tts, audio) as pipeline:
            pipeline.say("Bonjour")
            await pipeline.drain()
        assert tts.synthesised == ["Bonjour"]
        assert pipeline.spoken == ["Bonjour"]

    async def test_order_is_preserved(self):
        tts, audio = FakeTts(), FakeAudio()
        async with SpeechPipeline(tts, audio) as pipeline:
            for word in ("un", "deux", "trois"):
                pipeline.say(word)
            await pipeline.drain()
        assert pipeline.spoken == ["un", "deux", "trois"]

    async def test_saying_returns_immediately(self):
        """Queueing must not block the turn that is still generating text."""
        tts, audio = FakeTts(delay=0.05), FakeAudio(delay=0.05)
        async with SpeechPipeline(tts, audio) as pipeline:
            loop = asyncio.get_running_loop()
            started = loop.time()
            for word in ("un", "deux", "trois"):
                pipeline.say(word)
            assert loop.time() - started < 0.02
            await pipeline.drain()

    async def test_synthesis_overlaps_playback(self):
        """One loop would leave an audible gap at every sentence boundary."""
        tts, audio = FakeTts(delay=0.04), FakeAudio(delay=0.04)
        async with SpeechPipeline(tts, audio) as pipeline:
            loop = asyncio.get_running_loop()
            started = loop.time()
            for word in ("un", "deux", "trois"):
                pipeline.say(word)
            await pipeline.drain()
            elapsed = loop.time() - started

        # Strictly sequential would be 3 x (0.04 + 0.04) = 0.24s.
        assert elapsed < 0.22

    async def test_the_sample_rate_comes_from_the_synthesiser(self):
        tts, audio = FakeTts(), FakeAudio()
        tts.sample_rate = 16000
        async with SpeechPipeline(tts, audio) as pipeline:
            pipeline.say("Bonjour")
            await pipeline.drain()
        assert audio.played == [16000]

    async def test_empty_text_is_ignored(self):
        tts, audio = FakeTts(), FakeAudio()
        async with SpeechPipeline(tts, audio) as pipeline:
            pipeline.say("")
            pipeline.say("   ")
            await pipeline.drain()
        assert tts.synthesised == []

    async def test_drain_returns_at_once_when_nothing_is_queued(self):
        async with SpeechPipeline(FakeTts(), FakeAudio()) as pipeline:
            await asyncio.wait_for(pipeline.drain(), timeout=1)


class TestPipelineFailures:
    async def test_a_synthesis_failure_falls_back_to_text(self):
        fallbacks = []
        tts = FakeTts(fails_on=["deux"])
        async with SpeechPipeline(tts, FakeAudio(), fallbacks.append) as pipeline:
            for word in ("un", "deux", "trois"):
                pipeline.say(word)
            await pipeline.drain()
        assert fallbacks == ["deux"]
        assert pipeline.spoken == ["un", "trois"]

    async def test_a_failure_does_not_stall_the_queue(self):
        tts = FakeTts(fails_on=["un"])
        async with SpeechPipeline(tts, FakeAudio()) as pipeline:
            pipeline.say("un")
            pipeline.say("deux")
            await asyncio.wait_for(pipeline.drain(), timeout=2)
        assert pipeline.spoken == ["deux"]

    async def test_empty_audio_falls_back_to_text(self):
        fallbacks = []
        tts = FakeTts()
        tts.synthesize = lambda text: (np.array([], dtype=np.int16), 22050)
        async with SpeechPipeline(tts, FakeAudio(), fallbacks.append) as pipeline:
            pipeline.say("rien")
            await pipeline.drain()
        assert fallbacks == ["rien"]

    async def test_a_playback_failure_is_reported_not_fatal(self):
        fallbacks = []
        audio = FakeAudio()

        def broken(audio_data, sample_rate=None):
            raise OSError("device busy")

        audio.play_audio = broken
        async with SpeechPipeline(FakeTts(), audio, fallbacks.append) as pipeline:
            pipeline.say("un")
            await asyncio.wait_for(pipeline.drain(), timeout=2)
        assert fallbacks == ["un"]


class TestInterruption:
    async def test_queued_speech_is_dropped(self):
        tts, audio = FakeTts(delay=0.05), FakeAudio(delay=0.05)
        async with SpeechPipeline(tts, audio) as pipeline:
            for index in range(8):
                pipeline.say(f"phrase {index}")
            await asyncio.sleep(0.02)
            await pipeline.interrupt()
            await asyncio.wait_for(pipeline.drain(), timeout=2)
        assert len(pipeline.spoken) < 8

    async def test_playback_is_cut(self):
        audio = FakeAudio(delay=0.05)
        async with SpeechPipeline(FakeTts(), audio) as pipeline:
            pipeline.say("une phrase")
            await asyncio.sleep(0.01)
            await pipeline.interrupt()
            await pipeline.drain()
        assert audio.stopped == 1

    async def test_drain_still_returns_after_an_interruption(self):
        """Otherwise the turn would hang waiting for speech that was dropped."""
        tts, audio = FakeTts(delay=0.05), FakeAudio(delay=0.05)
        async with SpeechPipeline(tts, audio) as pipeline:
            for index in range(10):
                pipeline.say(f"phrase {index}")
            await pipeline.interrupt()
            await asyncio.wait_for(pipeline.drain(), timeout=2)

    async def test_nothing_new_is_spoken_while_interrupted(self):
        tts, audio = FakeTts(), FakeAudio()
        async with SpeechPipeline(tts, audio) as pipeline:
            await pipeline.interrupt()
            pipeline.say("ignore-moi")
            await asyncio.wait_for(pipeline.drain(), timeout=2)
        assert pipeline.spoken == []

    async def test_resume_accepts_speech_again(self):
        tts, audio = FakeTts(), FakeAudio()
        async with SpeechPipeline(tts, audio) as pipeline:
            await pipeline.interrupt()
            pipeline.resume()
            pipeline.say("me voilà")
            await asyncio.wait_for(pipeline.drain(), timeout=2)
        assert pipeline.spoken == ["me voilà"]

    async def test_interrupting_when_idle_is_safe(self):
        audio = FakeAudio()
        async with SpeechPipeline(FakeTts(), audio) as pipeline:
            await pipeline.interrupt()
        assert audio.stopped == 1


class TestLifecycle:
    async def test_starting_twice_is_safe(self):
        pipeline = SpeechPipeline(FakeTts(), FakeAudio())
        await pipeline.start()
        await pipeline.start()
        await pipeline.stop()

    async def test_stopping_without_starting_is_safe(self):
        await SpeechPipeline(FakeTts(), FakeAudio()).stop()

    async def test_workers_do_not_outlive_the_pipeline(self):
        pipeline = SpeechPipeline(FakeTts(), FakeAudio())
        async with pipeline:
            pipeline.say("bonjour")
            await pipeline.drain()
        assert pipeline._workers == []
