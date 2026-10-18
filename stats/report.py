"""Export a session as a report a reader can hand to someone else.

The report is the analysis, not the chat: the tables and their sources, the
ledger, and every stored result with its figures, caveats and limits. Plots are
embedded as data URIs, so one HTML file carries everything.

Everything user-controlled is escaped. A column named `<script>` must read as
text, never run.
"""

from __future__ import annotations

import html as html_module
from typing import Any

from stats.session import SessionStore

__all__ = ["html_report", "markdown_report"]


def _esc(value: Any) -> str:
    return html_module.escape(str(value), quote=True)


def _text(value: Any) -> str:
    """User content for the Markdown export.

    Markdown renderers pass inline HTML through, so angle brackets have to go
    even though quotes can stay readable.
    """
    return html_module.escape(str(value), quote=False)


def _items(payload: Any, name: str) -> list[Any]:
    """A payload list that may or may not exist, whatever the tool stored."""
    items = getattr(payload, name, None) or []
    return list(items)


def _statement(item: Any) -> str:
    """One effect, verdict or reading as a line of prose."""
    for field in ("statement", "summary", "text"):
        text = getattr(item, field, None)
        if text:
            return str(text)
    check, message = getattr(item, "check", ""), getattr(item, "statement", "")
    if check or message:
        return f"{check}: {message}".strip(": ")
    return str(item)


def markdown_report(store: SessionStore) -> str:
    """The session as Markdown: tables, ledger, findings."""
    lines = ["# Analysis report", ""]
    lines.append("## Tables")
    if not store.tables:
        lines.append("No tables are loaded.")
    for ref in store.tables.values():
        marker = " (primary)" if ref.name == store.primary else ""
        lines.append(
            f"- **{_text(ref.name)}**{_text(marker)}: {ref.rows:,} rows, {len(ref.columns)} columns"
        )
        lines.append(f"  - columns: {_text(', '.join(ref.columns))}")
        lines.append(f"  - source: {_text(ref.source or 'unknown')}")
        if ref.sha256:
            lines.append(f"  - sha256: {_text(ref.sha256)}")
        if ref.roles:
            confirmed = ", ".join(f"{role}={column}" for role, column in ref.roles.items())
            lines.append(f"  - confirmed roles: {_text(confirmed)}")
    lines.append("")
    lines.extend(_ledger_lines(store))
    lines.append("")
    lines.append("## Findings")
    if not store.results:
        lines.append("No results are stored.")
    for stored in store.results.values():
        lines.append(f"### {stored.result_id} ({stored.kind}, {stored.evidence_class})")
        if stored.summary:
            lines.append(_text(stored.summary))
        payload = stored.payload
        for item in _items(payload, "effects"):
            lines.append(f"- {_text(_statement(item))}")
        for item in _items(payload, "reading"):
            lines.append(f"- {_text(item)}")
        for item in _items(payload, "verdicts"):
            lines.append(f"- {_text(_statement(item))}")
        for caveat in _items(payload, "caveats"):
            lines.append(f"- Caveat: {_text(caveat)}")
        for limit in _items(payload, "cannot_say"):
            lines.append(f"- Cannot say: {_text(limit)}")
        quotable = list(getattr(stored, "quotable", []) or [])[:24]
        if quotable:
            lines.append(f"- Quotable: {', '.join(quotable)}")
        if getattr(payload, "image", ""):
            lines.append("- Figure: see the HTML export, which embeds it.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _ledger_lines(store: SessionStore) -> list[str]:
    status = store.state.status()
    lines = ["## Ledger"]
    for record in status.stages:
        line = f"- {record.stage}: {record.status}"
        if record.result_id:
            line += f" ({record.result_id})"
        if record.note:
            line += f": {record.note}"
        lines.append(line)
    for gap in status.missing:
        lines.append(f"- open: {gap}")
    lines.append(f"Recommended next: {status.recommended_next}")
    return lines


def html_report(store: SessionStore) -> str:
    """The session as one self-contained HTML file."""
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Analysis report</title>",
        "</head><body>",
        "<h1>Analysis report</h1>",
        "<h2>Tables</h2>",
        "<ul>",
    ]
    for ref in store.tables.values():
        marker = " (primary)" if ref.name == store.primary else ""
        parts.append(
            f"<li><strong>{_esc(ref.name)}</strong>{_esc(marker)}: "
            f"{ref.rows:,} rows, {len(ref.columns)} columns<br>"
            f"columns: {_esc(', '.join(ref.columns))}<br>"
            f"source: {_esc(ref.source or 'unknown')}"
            + (f"<br>sha256: {_esc(ref.sha256)}" if ref.sha256 else "")
            + "</li>"
        )
    parts.append("</ul><h2>Ledger</h2><ul>")
    for line in _ledger_lines(store)[1:]:
        parts.append(f"<li>{_esc(line)}</li>")
    parts.append("</ul><h2>Findings</h2>")
    for stored in store.results.values():
        parts.append(f"<h3>{_esc(stored.result_id)} ({_esc(stored.kind)})</h3>")
        if stored.summary:
            parts.append(f"<p>{_esc(stored.summary)}</p>")
        payload = stored.payload
        statements = [_statement(item) for item in _items(payload, "effects")]
        statements.extend(str(item) for item in _items(payload, "reading"))
        if statements:
            parts.append(
                "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in statements) + "</ul>"
            )
        for caveat in _items(payload, "caveats"):
            parts.append(f"<p><strong>Caveat:</strong> {_esc(caveat)}</p>")
        for limit in _items(payload, "cannot_say"):
            parts.append(f"<p><strong>Cannot say:</strong> {_esc(limit)}</p>")
        image = getattr(payload, "image", "")
        if image and str(image).startswith("data:image/"):
            alt = _esc(getattr(payload, "alt_text", ""))
            parts.append(f'<img src="{_esc(image)}" alt="{alt}" width="100%">')
    parts.append("</body></html>")
    return "\n".join(parts)
