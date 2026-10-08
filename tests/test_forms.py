"""Form generation: schema to fields, field values back to arguments."""

from __future__ import annotations

import pytest

import forms
import registry
import stats_tools


def fields_for(tool: str) -> list[forms.Field]:
    spec = registry.get(tool)
    return forms.fields_for(spec.input_schema(), spec.input_model.model_fields)


def by_name(tool: str) -> dict[str, forms.Field]:
    return {field.name: field for field in fields_for(tool)}


def test_fields_follow_the_tool_schema():
    fields = by_name("anova_effect")

    assert list(fields) == ["response", "factor"]
    assert fields["response"].kind == "string"
    assert fields["response"].required
    assert fields["factor"].default is None


def test_list_and_optional_arguments_are_recognized():
    fields = by_name("marginal_means")

    assert fields["factors"].kind == "array"
    assert fields["factors"].required
    assert fields["covariates"].kind == "array"
    assert not fields["covariates"].required
    assert fields["confidence_level"].kind == "number"
    assert fields["confidence_level"].default == 0.95
    assert fields["pairwise"].kind == "boolean"


def test_enum_arguments_keep_their_choices():
    # No tool declares a Literal today, so the enum path is driven by the schema.
    schema = {"properties": {"model": {"enum": ["4pl", "quadratic"]}}, "required": ["model"]}

    fields = forms.fields_for(schema, {"model": None})

    assert fields[0].kind == "enum"
    assert fields[0].choices == ["4pl", "quadratic"]
    assert fields[0].required


def test_arguments_split_arrays_and_drop_empty_optionals():
    fields = fields_for("marginal_means")
    arguments = forms.arguments(
        fields,
        {
            "response": "grain",
            "factors": "variety, nitrogen",
            "covariates": "",
            "confidence_level": 0.9,
            "pairwise": True,
        },
    )

    assert arguments == {
        "response": "grain",
        "factors": ["variety", "nitrogen"],
        "confidence_level": 0.9,
        "pairwise": True,
    }


def test_arguments_complain_about_missing_required_fields():
    with pytest.raises(stats_tools.ToolError, match="Fill in: response, factor"):
        forms.arguments(fields_for("anova_effect"), {"response": "", "factor": ""})


def test_arguments_reject_a_non_numeric_number():
    with pytest.raises(stats_tools.ToolError, match="must be a number"):
        forms.arguments(
            fields_for("power_analysis"), {"response": "a", "factor": "b", "alpha": "x"}
        )


def test_suggestions_come_from_the_dataset_columns():
    description = stats_tools.describe_dataset()

    assert forms.suggestion("response", description) in description.numeric
    assert forms.suggestion("factor", description) == "variety"
    assert forms.suggestion("fixed_effects", description) == "variety"
    assert forms.suggestion("group", description) == "block"
    assert forms.suggestion("dose", description) == "nitrogen"
    assert forms.suggestion("covariates", description) != "grain"
    assert forms.suggestion("top_outliers", description) == ""
