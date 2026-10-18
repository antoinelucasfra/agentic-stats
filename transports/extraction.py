"""Notes to table: one structured-extraction call on top of the LLM client.

The Playground's "Notes to table" block sends messy trial notes plus a list of
column names, and gets rows back as a DataFrame. The pydantic model is built at
call time from the requested columns, so the schema the provider sees always
matches the table the reader gets.
"""

from __future__ import annotations

import re
from typing import Any

import chatlas
import pandas as pd
from pydantic import BaseModel, ValidationError, create_model

from transports import llm

EXTRACTION_PROMPT = (
    "You read messy field-trial notes and return them as structured rows. "
    "Fill exactly the requested columns and no others, one record per trial or "
    "plot the notes describe. Copy values from the notes verbatim; leave a cell "
    "empty (null) when the notes do not state it, rather than guessing."
)

WRAPPER_FIELD = "records"


def _field_name(name: str, used: set[str]) -> str:
    """A column name as a valid Python identifier, unique within one request."""
    field = re.sub(r"\W+", "_", name.strip()).strip("_") or "column"
    if field[0].isdigit():
        field = f"col_{field}"
    while field in used:
        field += "_"
    used.add(field)
    return field


def build_model(columns: list[str]) -> type[BaseModel]:
    """The extraction schema: one record per requested column, plus its wrapper.

    The provider returns ``{records: [...]}``. Field names are sanitised
    identifiers; the original spelling is restored before the DataFrame is
    built, so "grain yield" arrives as a column literally named "grain yield".
    """
    used: set[str] = set()
    fields = {_field_name(name, used): (str | None, None) for name in columns}
    record = create_model("ExtractionRecord", **fields)
    return create_model("ExtractionRecords", **{WRAPPER_FIELD: (list[record], ...)})


def _as_text(value: Any) -> Any:
    """Numbers arrive as JSON numbers; let them validate as text, not fail."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _fail(errors: list[dict[str, Any]]) -> ValueError:
    """One actionable message for a row that will not validate."""
    loc = errors[0].get("loc", ())
    column = next((part for part in reversed(loc) if isinstance(part, str)), "unknown")
    return ValueError(
        f"The model returned a value in the '{column}' column that is not readable "
        "text. Try running again, or drop that column and re-extract."
    )


def _rows(
    payload: dict[str, Any], model: type[BaseModel], columns: list[str]
) -> list[dict[str, Any]]:
    """Validate the response and map sanitised field names back to columns."""
    record = model.model_fields[WRAPPER_FIELD].annotation.__args__[0]
    names = list(record.model_fields)
    try:
        validated = model.model_validate(
            {
                WRAPPER_FIELD: [
                    # A provider may return a bare value instead of a record;
                    # pass it through unchanged so its failure lands in the
                    # same actionable error as any unreadable cell.
                    {name: _as_text(row.get(name)) for name in names}
                    if isinstance(row, dict)
                    else row
                    for row in payload[WRAPPER_FIELD]
                ]
            }
        )
    except ValidationError as exc:
        raise _fail(exc.errors()) from exc
    mapping = dict(zip(names, columns, strict=True))
    return [{mapping[field]: getattr(row, field) for field in names} for row in validated.records]


def extract_columns(
    text: str, columns: list[str], client: chatlas.Chat | None = None
) -> pd.DataFrame:
    """The notes as one DataFrame with exactly the requested columns.

    Values that parse as numbers become floats; ``df.attrs["coerced"]`` names
    the columns that were converted, so the caller can tell the reader. Raises
    ValueError when nothing usable comes back, naming the offending column.
    """
    columns = [name.strip() for name in columns if name.strip()]
    if not columns:
        raise ValueError("Name at least one column to extract.")
    if not text.strip():
        raise ValueError("Paste the trial notes to extract from.")
    if client is None:
        client = _default_client()

    model = build_model(columns)
    try:
        payload = client.chat_structured(text, data_model=model)
    except ValueError as exc:
        raise ValueError(f"The extraction call failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The model did not return rows in the expected shape. Try running again.")
    payload.setdefault(WRAPPER_FIELD, [])
    rows = _rows(payload, model, columns)

    coerced: list[str] = []
    for column in columns:
        hit = False
        for row in rows:
            value = row[column]
            if value and _is_number(value):
                row[column] = float(value)
                hit = True
        if hit:
            coerced.append(column)
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["coerced"] = coerced
    return frame


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _default_client() -> chatlas.ChatOpenAICompletions:
    """The env-configured endpoint, no tools, the extraction system prompt.

    Like the demo preset, a remote endpoint needs AGENTIC_STATS_LLM_API_KEY on
    the server. The local Ollama default stays usable without one, matching
    llm.preset_default().
    """
    model, base_url, key = llm.settings()
    if not key and base_url.rstrip("/") != llm.DEFAULT_BASE_URL.rstrip("/"):
        raise ValueError(
            "Notes to table needs AGENTIC_STATS_LLM_API_KEY set where the app runs. "
            "Locally, set AGENTIC_STATS_LLM_BASE_URL to an Ollama endpoint instead."
        )
    return chatlas.ChatOpenAICompletions(
        model=model,
        base_url=base_url,
        api_key=key or "not-needed",
        system_prompt=EXTRACTION_PROMPT,
    )
