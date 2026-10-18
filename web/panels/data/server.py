"""Data panel server: import a public table, and confirm what it can answer.

The store is shared with the other panels: it arrives as a reactive value from
the root server, and every import replaces it with a refreshed copy so the
Playground form and the ledger see the new columns. An import that came through
`load_table` in the Agent tab does not refresh this panel until its next click,
which is benign: the chat already shows what was loaded.

Roles are reader-confirmed, not reader-invented. The panel shows what the
profile guessed and why, and Apply writes the confirmed map onto the table. The
next `profile_table` reads it, so a reader who knows the data overrules a guess
once instead of correcting every tool call.
"""

from __future__ import annotations

from typing import Any

from shiny import Inputs, Outputs, Session, module, reactive, render, ui

from stats import roles, views, workflow
from stats.session import use_session
from stats.stats_tools import ToolError
from utils import catalog

ROLE_ORDER = ("response", "factor", "group", "dose", "time")

# The catalog index is identical for every session, so one process-wide copy is
# enough. A failed fetch leaves it empty and the panel falls back to curated.
_INDEX: list[catalog.CatalogEntry] | None = None


def _index() -> list[catalog.CatalogEntry]:
    global _INDEX
    if _INDEX is None:
        _INDEX = catalog.fetch_index()
    return _INDEX


@module.server
def data_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    store: reactive.Value,
) -> None:
    found = reactive.value([])

    def _load(url: str, name: str | None = None) -> None:
        try:
            with use_session(store()):
                summary = workflow.load_table(url, name=name)
            store.set(store().refreshed())
        except ToolError as exc:
            ui.notification_show(str(exc), type="error")
            return
        ui.notification_show(
            f"Loaded {summary.name}: {summary.n_rows:,} rows "
            f"across {len(summary.columns)} columns.",
            type="message",
        )

    @reactive.effect
    @reactive.event(input.fetch)
    def _fetch_url() -> None:
        url = input.url().strip()
        if not url:
            ui.notification_show("Paste a URL first.", type="error")
            return
        _load(url)

    @reactive.effect
    @reactive.event(input.catalog_search)
    def _search() -> None:
        query = input.catalog_q().strip()
        try:
            entries = _index()
        except Exception as exc:
            entries = []
            ui.notification_show(
                f"The catalog index is unreachable ({exc}). Searching the curated list.",
                type="warning",
            )
        results = catalog.search([*catalog.curated(), *entries], query)
        found.set(results)
        if not results:
            ui.update_select("catalog_pick", choices={"": "No matches"})
            return
        ui.update_select(
            "catalog_pick",
            choices={
                str(index): f"{entry.name} ({entry.package}) — {entry.title[:60]}"
                for index, entry in enumerate(results)
            },
            selected="0",
        )

    @reactive.effect
    @reactive.event(input.catalog_load)
    def _load_catalog() -> None:
        try:
            entry = found()[int(input.catalog_pick())]
        except (ValueError, IndexError, TypeError):
            ui.notification_show("Search the catalog and pick a table first.", type="error")
            return
        _load(entry.csv_url, name=entry.name)

    @reactive.effect
    def _sync_tables() -> None:
        current = store()
        ui.update_select(
            "table_pick",
            choices={name: name for name in current.tables},
            selected=current.primary,
        )

    @reactive.effect
    @reactive.event(input.make_primary)
    def _make_primary() -> None:
        current = store()
        try:
            current.set_primary(input.table_pick())
        except ToolError as exc:
            ui.notification_show(str(exc), type="error")
            return
        store.set(current.refreshed())
        ui.notification_show(f"{input.table_pick()} is now the primary table.", type="message")

    @render.ui
    def tables_view() -> Any:
        current = store()
        rows = []
        for ref in current.tables.values():
            badge = "PRIMARY" if ref.name == current.primary else ""
            rows.append(
                [
                    ref.name,
                    f"{ref.rows:,}",
                    str(len(ref.columns)),
                    (ref.source or "")[:60],
                    badge,
                ]
            )
        return [
            ui.tags.h5("Loaded tables"),
            views.table(["Table", "Rows", "Columns", "Source", ""], rows).tag,
        ]

    @render.ui
    def roles_view() -> Any:
        current = store()
        try:
            ref = current.table(input.table_pick() or None)
        except ToolError:
            ref = current.table()
        profile = roles.infer_profile(ref.frame, ref.roles or None)
        profile = roles.infer_profile(ref.frame, ref.roles or None)
        by_column: dict[str, list[str]] = {}
        for guess in sorted(profile.guesses, key=lambda item: -item.confidence):
            by_column.setdefault(guess.column, []).append(
                f"{guess.role} ({guess.confidence:.0%}): {guess.reason}"
            )
        guess_lines = [
            ui.tags.li(ui.tags.strong(column), ": ", "; ".join(lines[:2]))
            for column, lines in by_column.items()
        ]
        selects = [
            ui.input_select(
                f"role_{role}",
                f"{role} (suggested: {profile.best(role) or 'none'})",
                choices={"": "auto", **{name: name for name in ref.columns}},
                selected=ref.roles.get(role, ""),
            )
            for role in ROLE_ORDER
        ]
        return [
            ui.tags.h5(f"Roles for {ref.name}"),
            ui.tags.ul(*guess_lines),
            *[block.tag for block in views.notes(profile.notes)],
            *selects,
            ui.input_action_button("apply_roles", "Apply roles", class_="btn-primary"),
        ]

    @reactive.effect
    @reactive.event(input.apply_roles)
    def _apply_roles() -> None:
        current = store()
        try:
            ref = current.table(input.table_pick() or None)
        except ToolError as exc:
            ui.notification_show(str(exc), type="error")
            return
        overrides = {role: (input[f"role_{role}"]() or "") for role in ROLE_ORDER}
        overrides = {role: column for role, column in overrides.items() if column}
        current.tables[ref.name] = ref.with_roles(overrides)
        store.set(current.refreshed())
        ui.notification_show(
            f"Roles confirmed for {ref.name}. Profile the table to see them applied.",
            type="message",
        )
