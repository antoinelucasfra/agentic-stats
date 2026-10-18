"""Markdown-aware chunking: headings first, then paragraphs.

One document becomes a handful of chunks of roughly `TARGET` characters, so a
retrieval hit can be quoted as a coherent passage rather than a random 200-char
window. Every chunk keeps its place: the section heading stays in the text, and
a chunk with no heading at all (file preamble, heading-less notes) gets the
document title prepended as a level-1 heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Chunks aim for this size; OVERLAP is the fraction of it carried between
# consecutive chunks inside one section, so a hit straddling a boundary still
# reads whole on one side.
TARGET = 500
OVERLAP = 0.15
# A paragraph longer than this is hard-split before packing: a multi-megabyte
# single line (a one-row CSV upload) must never become one giant chunk, or the
# embedder request fails for the whole upload.
MAX_PARAGRAPH = 4000

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage: its text, where it came from, and its place."""

    text: str
    source: str
    heading: str
    id: str = ""


def _paragraphs(lines: list[str]) -> list[str]:
    """Split blank-line-separated paragraphs, keeping single newlines inside."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", "\n".join(lines))
        if paragraph.strip()
    ]


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split one markdown document into titled, headed, sized chunks."""
    title = ""
    sections: list[tuple[str, str, list[str]]] = []  # (heading line, heading text, lines)
    preamble: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            if sections:
                sections[-1][2].append(line)
            else:
                preamble.append(line)
            continue
        heading_text = match.group(2).strip()
        if not title and len(match.group(1)) == 1:
            title = heading_text
        sections.append((line.strip(), heading_text, []))

    chunks: list[Chunk] = []
    if preamble and preamble != [""]:
        sections.insert(0, ("", "", preamble))

    chunk_id = 0
    for heading_line, heading_text, lines in sections:
        body = _pack(_paragraphs(lines))
        for passage in body:
            if heading_line:
                chunk_text = f"{heading_line}\n\n{passage}"
            else:
                chunk_text = f"# {title or Path(source).stem}\n\n{passage}"
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    heading=heading_text or title or Path(source).stem,
                    id=f"{Path(source).name}#{chunk_id}",
                )
            )
            chunk_id += 1
    return chunks


def _pack(paragraphs: list[str]) -> list[str]:
    """Greedily pack paragraphs up to TARGET, overlapping by the tail of the last.

    A single paragraph longer than TARGET stays whole up to MAX_PARAGRAPH:
    splitting mid-paragraph loses the sentence, which matters more than the
    size budget. Past MAX_PARAGRAPH the size wins and the paragraph is
    hard-split into cap-sized pieces first.

    A buffer holding only a carried overlap never flushes on its own:
    it would become a chunk with no new content in it.
    """
    paragraphs = [
        piece
        for paragraph in paragraphs
        for piece in ([paragraph] if len(paragraph) <= MAX_PARAGRAPH else _hard_split(paragraph))
    ]
    packed: list[str] = []
    buffer: list[str] = []
    carried = False  # buffer[0] is a carried overlap, not fresh content

    def flush() -> None:
        if buffer and not carried:
            packed.append("\n\n".join(buffer))

    for paragraph in paragraphs:
        current = len("\n\n".join(buffer))
        # An overlap never justifies breaking the hard cap: drop it instead.
        if carried and current + len(paragraph) > MAX_PARAGRAPH:
            buffer, carried, current = [], False, 0
        if buffer and not carried and current + len(paragraph) > TARGET:
            flush()
            tail = _tail(buffer[-1])
            carried = len(tail) + 2 + len(paragraph) <= MAX_PARAGRAPH
            buffer = [tail] if carried else []
            current = len("\n\n".join(buffer))
        buffer.append(paragraph)
        carried = False
    flush()
    return packed


def _hard_split(text: str) -> list[str]:
    """Cut a monster paragraph into MAX_PARAGRAPH-sized pieces, on a word boundary."""
    pieces: list[str] = []
    while len(text) > MAX_PARAGRAPH:
        cut = text.rfind(" ", 0, MAX_PARAGRAPH + 1)
        if cut == -1:
            cut = MAX_PARAGRAPH  # no space in the window: cut mid-word
        pieces.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def _tail(text: str, size: int = int(TARGET * OVERLAP)) -> str:
    """The last `size` characters, cut on a word boundary so it reads as text."""
    tail = text[-size:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail
