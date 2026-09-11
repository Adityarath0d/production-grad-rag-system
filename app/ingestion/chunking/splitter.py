"""Text chunking utilities for ingestion.

The splitter is deliberately dependency-free: ingestion remains usable for every
loader and the chunk boundaries stay predictable. It favours paragraphs and
sentences, but still guarantees that an unusually long paragraph cannot produce
an oversized embedding request.
"""

import re
from typing import List

import logfire


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_BOUNDARY = re.compile(r"\s+")


def _normalise_paragraphs(text: str) -> List[str]:
    """Return non-empty paragraphs with stable, single-space inline whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        re.sub(r"[\t \f\v]+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]


def _split_oversized_text(text: str, limit: int) -> List[str]:
    """Split a long paragraph by sentences, then words, then characters."""
    if len(text) <= limit:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    # A paragraph with no useful sentence boundary falls through to words.
    if len(sentences) == 1:
        sentences = _WORD_BOUNDARY.split(text)

    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            # Necessary for URLs, identifiers, or OCR output without boundaries.
            pieces.extend(sentence[index : index + limit] for index in range(0, len(sentence), limit))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def _overlap_suffix(text: str, overlap: int) -> str:
    """Take up to ``overlap`` trailing characters without beginning mid-word."""
    if overlap == 0 or not text:
        return ""

    suffix = text[-overlap:]
    if len(text) > overlap:
        first_space = suffix.find(" ")
        if first_space != -1:
            suffix = suffix[first_space + 1 :]
    return suffix.strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into bounded, retrieval-friendly chunks.

    Chunks retain whole paragraphs where possible, then prefer sentence and word
    boundaries. ``chunk_overlap`` preserves trailing context in the following
    chunk, improving retrieval when a relevant passage crosses a boundary.
    Both values are character counts, which makes the limit independent of the
    embedding model's tokenizer.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least zero and smaller than chunk_size")

    with logfire.span("Text Chunking", text_length=len(text), chunk_size=chunk_size):
        paragraphs = _normalise_paragraphs(text)
        if not paragraphs:
            return []

        units = [
            piece
            for paragraph in paragraphs
            for piece in _split_oversized_text(paragraph, chunk_size)
        ]

        chunks: List[str] = []
        current = ""
        for unit in units:
            separator = "\n\n" if current else ""
            if len(current) + len(separator) + len(unit) <= chunk_size:
                current += separator + unit
                continue

            if current:
                chunks.append(current)

            overlap = _overlap_suffix(current, chunk_overlap)
            # An overlap must never make the next chunk exceed its limit.
            available_overlap = max(0, chunk_size - len(unit) - 2)
            overlap = _overlap_suffix(overlap, available_overlap)
            current = f"{overlap}\n\n{unit}" if overlap else unit

        if current:
            chunks.append(current)

        logfire.info("Generated chunks", chunk_count=len(chunks))
        return chunks
