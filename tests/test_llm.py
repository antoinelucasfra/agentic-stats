"""The chatlas client: every registered tool, and JSON results."""

from __future__ import annotations

import inspect
import json

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
