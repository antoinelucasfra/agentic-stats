"""The chatlas client: every registered tool, and JSON results."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from stats import registry
from transports import llm

DUMMY = {"model": "dummy", "base_url": "http://127.0.0.1:1/v1"}


def test_build_client_registers_every_tool():
    client = llm.build_client(**DUMMY)
    registered = client._tools  # chatlas keeps the registry here

    assert set(registered) == {spec.name for spec in registry.TOOLS}
    for spec in registry.TOOLS:
        function = registered[spec.name].schema["function"]
        assert function["name"] == spec.name
        assert set(function["parameters"]["properties"]) == set(spec.input_model.model_fields)
        # chatlas reads the raw docstring; the registry dedents it.
        assert inspect.cleandoc(function["description"]) == spec.description


def test_tool_results_reach_the_model_as_json():
    client = llm.build_client(**DUMMY)

    result = client._tools["anova_effect"].func(response="grain", factor="nitrogen")
    payload = json.loads(result.get_model_value())

    assert payload["factor"] == "nitrogen"
    assert payload["p_value"] < 0.001


def test_tool_errors_carry_the_actionable_message():
    client = llm.build_client(**DUMMY)

    try:
        client._tools["anova_effect"].func(response="grain", factor="nitrogen_id")
    except Exception as exc:
        assert "Available columns" in str(exc)
    else:  # pragma: no cover - the tool must reject an unknown column
        raise AssertionError("an unknown column should raise")


def test_settings_default_and_environment(monkeypatch):
    assert llm.settings() == (llm.DEFAULT_MODEL, llm.DEFAULT_BASE_URL, "")

    monkeypatch.setenv("AGENTIC_STATS_LLM_MODEL", "smol")
    monkeypatch.setenv("AGENTIC_STATS_LLM_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("AGENTIC_STATS_LLM_API_KEY", "secret")

    assert llm.settings() == ("smol", "http://example.test/v1", "secret")


def test_resolve_demo_uses_the_server_key(monkeypatch):
    monkeypatch.setenv("AGENTIC_STATS_LLM_API_KEY", "server-key")

    assert llm.resolve("demo", "", "", "") == (
        "llama-3.1-8b-instant",
        llm.GROQ_BASE_URL,
        "server-key",
    )


def test_resolve_demo_without_a_server_key_explains_itself(monkeypatch):
    monkeypatch.delenv("AGENTIC_STATS_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="AGENTIC_STATS_LLM_API_KEY"):
        llm.resolve("demo")


def test_resolve_custom_and_unknown_presets_pass_through():
    assert llm.resolve("custom", "m", "http://x/v1", "k") == ("m", "http://x/v1", "k")
    assert llm.resolve("nope", "m", "http://x/v1", "") == ("m", "http://x/v1", "")


def test_trace_records_a_request_result_pair():
    trace: list = []
    pending: dict = {}
    request = SimpleNamespace(id="1", name="describe_dataset", arguments={})
    llm._record_request(trace, pending, request)
    result = SimpleNamespace(request=request, error=None, get_model_value=lambda: '{"n_rows": 72}')
    llm._record_result(trace, pending, result)

    assert trace == [llm.TraceEvent("describe_dataset", {}, '{"n_rows": 72}')]


def test_trace_marks_errors():
    trace: list = []
    pending: dict = {}
    request = SimpleNamespace(id="1", name="anova_effect", arguments={"factor": "x"})
    llm._record_request(trace, pending, request)
    result = SimpleNamespace(
        request=request, error=ValueError("bad column"), get_model_value=lambda: ""
    )
    llm._record_result(trace, pending, result)

    assert trace[0].is_error
    assert trace[0].summary.startswith("ERROR")
