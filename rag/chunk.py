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

    A single paragraph longer than TARGET stays whole: splitting mid-paragraph
    loses the sentence, which matters more than the size budget.
    """
    packed: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append("\n\n".join(buffer))

    for paragraph in paragraphs:
        current = len("\n\n".join(buffer))
        # A buffer holding only a carried overlap never flushes on its own:
        # it would become a chunk with no new content in it.
        if buffer and current + len(paragraph) > TARGET and current > len(buffer[0]):
            flush()
            buffer = [_tail(buffer[-1])]
        buffer.append(paragraph)
    flush()
    return packed


def _tail(text: str, size: int = int(TARGET * OVERLAP)) -> str:
    """The last `size` characters, cut on a word boundary so it reads as text."""
    tail = text[-size:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail
