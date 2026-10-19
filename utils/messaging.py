"""Splitting long answers into WhatsApp/SMS-sized messages."""

from __future__ import annotations

import re

# Twilio rejects a <Message> body of 1600+ characters (error 21617) and delivers
# nothing. Leave headroom for the "(i/n) " prefix and multi-unit characters.
TWILIO_MAX_BODY = 1600
CHUNK_TARGET = 1500

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _pack(pieces: list[str], limit: int, joiner: str) -> list[str]:
    """Greedily join pieces into chunks no longer than `limit`."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else f"{current}{joiner}{piece}"
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, limit: int) -> list[str]:
    """Split at the last whitespace before `limit`; cut mid-word only if there is none."""
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def split_message(text: str, limit: int = CHUNK_TARGET) -> list[str]:
    """Split `text` into chunks of at most `limit` characters.

    Prefers paragraph boundaries, then sentence boundaries, then whitespace, so
    each chunk reads on its own if messages arrive out of order.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    # paragraphs that fit as-is; oversized ones are broken down further first
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            pieces.append(paragraph)
            continue
        sentences: list[str] = []
        for sentence in _SENTENCE_END.split(paragraph):
            sentences.extend(_hard_split(sentence, limit) if len(sentence) > limit else [sentence])
        # re-pack the sentences of this paragraph so it is not rendered one line per sentence
        pieces.extend(_pack(sentences, limit, joiner=" "))

    return _pack(pieces, limit, joiner="\n\n")


def number_chunks(chunks: list[str]) -> list[str]:
    """Prefix chunks with "(i/n) " when there is more than one."""
    if len(chunks) <= 1:
        return chunks
    total = len(chunks)
    return [f"({index}/{total}) {chunk}" for index, chunk in enumerate(chunks, start=1)]
