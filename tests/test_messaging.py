"""Tests for utils/messaging.py (Twilio-sized message chunking)."""

from __future__ import annotations

import pytest

from utils.messaging import CHUNK_TARGET, TWILIO_MAX_BODY, number_chunks, split_message


def _reassembled(chunks: list[str]) -> str:
    """Chunk text with all whitespace collapsed, for content-preservation checks."""
    return " ".join(" ".join(chunks).split())


class TestSplitMessage:
    def test_short_text_is_single_chunk(self):
        assert split_message("Hello there.") == ["Hello there."]

    def test_empty_text_gives_no_chunks(self):
        assert split_message("   ") == []

    def test_exact_limit_is_not_split(self):
        text = "a" * CHUNK_TARGET
        assert split_message(text) == [text]

    def test_headroom_for_prefix_stays_under_twilio_limit(self):
        assert CHUNK_TARGET + len("(10/10) ") < TWILIO_MAX_BODY

    def test_splits_on_paragraphs_first(self):
        p1, p2, p3 = "A" * 40, "B" * 40, "C" * 40
        chunks = split_message(f"{p1}\n\n{p2}\n\n{p3}", limit=90)
        assert chunks == [f"{p1}\n\n{p2}", p3]

    def test_keeps_paragraph_breaks_inside_chunks(self):
        chunks = split_message("First para.\n\nSecond para.\n\n" + "X" * 50, limit=60)
        assert chunks[0] == "First para.\n\nSecond para."

    def test_oversized_paragraph_splits_on_sentences(self):
        sentences = [f"Sentence number {i} is here." for i in range(8)]
        paragraph = " ".join(sentences)
        chunks = split_message(paragraph, limit=70)
        assert all(len(c) <= 70 for c in chunks)
        # no sentence is cut in the middle
        for chunk in chunks:
            assert chunk.endswith(".")
        assert _reassembled(chunks) == paragraph

    def test_oversized_sentence_splits_on_whitespace(self):
        sentence = " ".join(["word"] * 60)  # 299 chars, no sentence end
        chunks = split_message(sentence, limit=50)
        assert all(len(c) <= 50 for c in chunks)
        assert all(c == c.strip() and "  " not in c for c in chunks)
        assert _reassembled(chunks) == sentence

    def test_unbreakable_token_is_hard_cut(self):
        token = "x" * 120
        chunks = split_message(token, limit=50)
        assert chunks == ["x" * 50, "x" * 50, "x" * 20]

    def test_realistic_long_answer_respects_limit_and_preserves_content(self):
        paragraph = (
            "You can register with a general practitioner near your home. "
            "Bring your ID and proof of address! Is it urgent? Call 112."
        )
        answer = "\n\n".join([paragraph] * 40)  # ~5k chars
        chunks = split_message(answer)
        assert len(chunks) >= 4
        assert all(len(c) <= CHUNK_TARGET for c in chunks)
        assert _reassembled(chunks) == _reassembled([answer])

    def test_non_latin_text_is_measured_in_characters(self):
        text = "\n\n".join(["Це довге речення українською мовою для перевірки." for _ in range(60)])
        chunks = split_message(text, limit=200)
        assert all(len(c) <= 200 for c in chunks)
        assert _reassembled(chunks) == _reassembled([text])


class TestNumberChunks:
    def test_single_chunk_has_no_prefix(self):
        assert number_chunks(["only one"]) == ["only one"]

    def test_empty_list(self):
        assert number_chunks([]) == []

    def test_multiple_chunks_get_ordinal_prefixes(self):
        assert number_chunks(["a", "b", "c"]) == ["(1/3) a", "(2/3) b", "(3/3) c"]

    def test_prefixed_chunks_fit_twilio_limit(self):
        chunks = number_chunks(split_message("word " * 4000))
        assert len(chunks) > 10  # two-digit prefixes exercised
        assert all(len(c) < TWILIO_MAX_BODY for c in chunks)
