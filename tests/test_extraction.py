"""Notes to table: the dynamic schema, the round trip, and its failure modes.

No network and no browser: the client is a fake with chatlas's chat_structured
shape, the way tests/test_llm.py pokes the client directly.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from shiny import App, reactive, ui
from shiny.testserver import test_server

from transports import extraction


class FakeClient:
    """The one method extract_columns uses, answering with a canned payload."""

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple] = []

    def chat_structured(self, *args, data_model=None, **kwargs):
        self.calls.append((args, data_model))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def records(*rows):
    return {"records": list(rows)}


def test_build_model_sanitises_column_names():
    model = extraction.build_model(["grain yield", "2nd col", "grain yield"])
    record = model.model_fields[extraction.WRAPPER_FIELD].annotation.__args__[0]

    assert list(record.model_fields) == ["grain_yield", "col_2nd_col", "grain_yield_"]
    assert all(field.default is None for field in record.model_fields.values())


def test_extract_columns_round_trip_restores_column_names():
    client = FakeClient(records({"grain_yield": "A", "col_2nd_col": "x"}, {"grain_yield": "B"}))

    frame = extraction.extract_columns("notes", ["grain yield", "2nd col"], client=client)

    assert list(frame.columns) == ["grain yield", "2nd col"]
    assert frame["grain yield"].tolist() == ["A", "B"]
    assert frame["2nd col"].dropna().tolist() == ["x"]
    assert frame["2nd col"].isna().tolist() == [False, True]
    text, data_model = client.calls[0]
    assert "notes" in text
    assert data_model is not None


def test_numeric_text_becomes_floats_with_a_note():
    client = FakeClient(
        records(
            {"block": "1", "nitrogen": "3.5", "variety": "A"},
            {"block": None, "nitrogen": None, "variety": "B"},
        )
    )

    frame = extraction.extract_columns("notes", ["block", "nitrogen", "variety"], client=client)

    assert frame["block"].dropna().tolist() == [1.0]
    assert frame["nitrogen"].dropna().tolist() == [3.5]
    assert frame["block"].isna().tolist() == [False, True]
    assert frame["variety"].tolist() == ["A", "B"]
    assert frame.attrs["coerced"] == ["block", "nitrogen"]


def test_unreadable_value_raises_naming_the_column():
    client = FakeClient(records({"grain": {"unit": "t"}, "variety": "A"}))

    with pytest.raises(ValueError, match="'grain'"):
        extraction.extract_columns("notes", ["grain", "variety"], client=client)


def test_json_numbers_are_read_not_rejected():
    client = FakeClient(records({"grain": 4.2, "variety": "A"}))

    frame = extraction.extract_columns("notes", ["grain", "variety"], client=client)

    assert frame["grain"].tolist() == [4.2]


def test_bad_payload_shapes_raise_actionable_errors():
    with pytest.raises(ValueError, match="at least one column"):
        extraction.extract_columns("notes", ["  "], client=FakeClient(records()))
    with pytest.raises(ValueError, match="Paste the trial notes"):
        extraction.extract_columns("  ", ["grain"], client=FakeClient(records()))
    failing = FakeClient(ValueError("provider down"))
    with pytest.raises(ValueError, match="extraction call failed"):
        extraction.extract_columns("notes", ["grain"], client=failing)
    with pytest.raises(ValueError, match="expected shape"):
        extraction.extract_columns("notes", ["grain"], client=FakeClient([1, 2]))


def test_a_non_dict_record_raises_naming_the_column():
    client = FakeClient(records("a", "b"))

    with pytest.raises(ValueError, match="'records'"):
        extraction.extract_columns("notes", ["grain"], client=client)


def test_default_client_uses_env_settings_without_tools(monkeypatch):
    monkeypatch.setenv("AGENTIC_STATS_LLM_MODEL", "smol")
    monkeypatch.setenv("AGENTIC_STATS_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("AGENTIC_STATS_LLM_API_KEY", raising=False)

    client = extraction._default_client()

    assert client.system_prompt == extraction.EXTRACTION_PROMPT
    assert client._tools == {}
    assert client.model == "smol"


def test_default_client_demands_a_key_for_remote_endpoints(monkeypatch):
    monkeypatch.delenv("AGENTIC_STATS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("AGENTIC_STATS_LLM_BASE_URL", "https://api.example.test/v1")

    with pytest.raises(ValueError, match="AGENTIC_STATS_LLM_API_KEY"):
        extraction._default_client()


def test_csv_payload_shape():
    client = FakeClient(
        records(
            {"block": "A", "variety": "axona", "nitrogen": "0", "grain": "6.1"},
            {"block": "A", "variety": "axona", "nitrogen": "50", "grain": "7.4"},
        )
    )

    frame = extraction.extract_columns(
        "notes", ["block", "variety", "nitrogen", "grain"], client=client
    )
    csv = frame.to_csv(index=False)

    assert csv.splitlines()[0] == "block,variety,nitrogen,grain"
    assert csv.splitlines()[1] == "A,axona,0.0,6.1"
    assert isinstance(pd.read_csv(io.StringIO(csv)), pd.DataFrame)


# --- the Playground section, served headlessly --------------------------------

MODULE_ID = "playground"


def _notes_app() -> App:
    from stats import stats_tools
    from web.panels.playground.server import playground_server
    from web.panels.playground.ui import playground_ui

    def server_fn(input, output, session):
        playground_server(
            MODULE_ID, pending=reactive.Value(None), store=reactive.Value(stats_tools.env_store())
        )

    return App(ui.page_navbar(playground_ui(MODULE_ID)), server_fn)


def _notes_html(server) -> str:
    output = server.get_output(f"{MODULE_ID}-notes_result")
    assert output.status == "ok", f"notes_result rendered {output.status}: {output.error}"
    value = output.value
    return value["html"] if isinstance(value, dict) else str(value)


def test_playground_section_renders_the_extracted_table(monkeypatch):
    captured: dict = {}

    def fake_extract(text, columns, client=None):
        captured.update(text=text, columns=columns)
        return pd.DataFrame({"block": ["A"], "grain yield": [6.0]})

    monkeypatch.setattr(extraction, "extract_columns", fake_extract)

    with test_server(_notes_app(), timeout_secs=20) as server:
        server.set_inputs(
            **{
                f"{MODULE_ID}-notes": "messy notes",
                f"{MODULE_ID}-notes_columns": "block, grain yield",
                f"{MODULE_ID}-extract": 1,
            }
        )
        html = _notes_html(server)

    assert captured == {"text": "messy notes", "columns": ["block", "grain yield"]}
    assert "grain yield" in html and "6.0" in html


def test_playground_section_surfaces_the_key_error(monkeypatch):
    monkeypatch.delenv("AGENTIC_STATS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("AGENTIC_STATS_LLM_BASE_URL", "https://api.example.test/v1")

    with test_server(_notes_app(), timeout_secs=20) as server:
        server.set_inputs(
            **{
                f"{MODULE_ID}-notes": "messy notes",
                f"{MODULE_ID}-notes_columns": "block, grain yield",
                f"{MODULE_ID}-extract": 1,
            }
        )
        html = _notes_html(server)

    assert "AGENTIC_STATS_LLM_API_KEY" in html
