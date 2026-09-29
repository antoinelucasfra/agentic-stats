"""Tests for the HTTP surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentic_stats.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_tools_endpoint_lists_three_tools():
    tools = client.get("/tools").json()
    assert {tool["name"] for tool in tools} == {
        "describe_dataset",
        "fit_mixed_model",
        "anova_effect",
    }


def test_dataset_endpoint():
    body = client.get("/dataset").json()
    assert body["n_rows"] > 0
    assert "assay_signal" in body["numeric"]


def test_mixed_model_endpoint():
    response = client.post(
        "/mixed-model",
        json={"response": "assay_signal", "fixed_effects": ["formulation", "dose"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["converged"] is True
    assert any(effect["term"] == "dose" for effect in body["fixed_effects"])


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        ({"response": "assay_signal", "factor": "nope"}, "Available columns"),
        ({"response": "formulation", "factor": "dose"}, "not numeric"),
    ],
)
def test_tool_errors_become_400_with_actionable_message(payload, expected_detail):
    response = client.post("/anova", json=payload)
    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]
