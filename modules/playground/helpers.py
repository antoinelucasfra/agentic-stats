"""Playground-only helpers: form field widgets and upload validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from shiny import ui

import forms
from stats_tools import DatasetDescription
from utils import data
from utils.config import MAX_UPLOAD_BYTES, UPLOAD_SUFFIXES
from utils.formatting import number


def field_input(field: forms.Field, args: dict[str, Any], dataset: DatasetDescription) -> Any:
    """One form control, prefilled from a card's arguments or the dataset."""
    label = f"{field.name} *" if field.required else field.name
    value = args.get(field.name)
    if value is None:
        value = (
            field.default if field.default is not None else forms.suggestion(field.name, dataset)
        )

    if field.kind == "enum":
        control = ui.input_select(
            field.name, label, choices=field.choices, selected=str(value) if value else None
        )
    elif field.kind == "boolean":
        control = ui.input_checkbox(field.name, label, value=bool(value))
    elif field.kind in ("number", "integer"):
        control = ui.input_numeric(field.name, label, value=number(value))
    elif field.kind == "array":
        joined = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        control = ui.input_text(field.name, label, value=joined)
    else:
        control = ui.input_text(field.name, label, value=str(value or ""))

    children: list[Any] = [control]
    if field.description:
        children.append(ui.tags.small(field.description, class_="text-muted d-block"))
    return ui.tags.div(*children, class_="mb-3")


def upload_problem(info: dict[str, Any]) -> str | None:
    """Reject an upload before it is read. The browser name is never a path."""
    if not str(info["name"]).lower().endswith(UPLOAD_SUFFIXES):
        suffixes = " and ".join(UPLOAD_SUFFIXES)
        return f"Only {suffixes} files are accepted."
    if info["size"] > MAX_UPLOAD_BYTES:
        return f"Keep the file under {MAX_UPLOAD_BYTES // 1024 // 1024} MB."
    return None


def read_uploaded(path: str) -> pd.DataFrame:
    """Read an uploaded CSV or parquet. Parquet goes through duckdb, as on disk."""
    if Path(path).suffix.lower() == ".parquet":
        return data.read_parquet(path)
    return pd.read_csv(path)
