"""Turn a tool's JSON Schema into form fields, and field values back into arguments.

The Playground form is generated from `registry`, so a tool added on the Python
side shows up with no change here. Kept free of Shiny so the coercion rules can
be unit-tested, which is what the browser app's `form.ts` used to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from stats.stats_tools import DatasetDescription, ToolError

FieldKind = Literal["string", "number", "integer", "boolean", "array", "enum"]

_OMIT = object()

DEFAULTS = {"response": "grain", "factor": "variety", "group": "block", "dose": "nitrogen"}


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind
    required: bool
    choices: list[str]
    default: Any
    description: str


def _unwrap(property_: dict[str, Any]) -> dict[str, Any]:
    """`list[str] | None` nests the real type one level down, next to the null."""
    options = [option for option in property_.get("anyOf", []) if option.get("type") != "null"]
    return options[0] if options else property_


def _kind(property_: dict[str, Any], inner: dict[str, Any]) -> FieldKind:
    if property_.get("enum"):
        return "enum"
    match inner.get("type"):
        case "array":
            return "array"
        case "boolean":
            return "boolean"
        case "number" | "integer":
            return inner["type"]
        case _:
            return "string"


def fields_for(input_schema: dict[str, Any], model_fields: dict[str, Any]) -> list[Field]:
    """Describe every argument the tool takes, in schema order."""
    required = set(input_schema.get("required", []))
    fields = []
    for name, property_ in input_schema.get("properties", {}).items():
        inner = _unwrap(property_)
        info = model_fields.get(name)
        # Required fields carry no usable default: pydantic stores a sentinel.
        default = None if info is None or info.is_required() else info.default
        fields.append(
            Field(
                name=name,
                kind=_kind(property_, inner),
                required=name in required,
                choices=[str(choice) for choice in property_.get("enum", [])],
                default=default,
                description=property_.get("description", ""),
            )
        )
    return fields


def suggestion(name: str, description: DatasetDescription) -> str:
    """A demo value for a familiar argument name, from the dataset's own columns."""
    numeric = description.numeric
    factors = description.factors
    response = DEFAULTS["response"] if DEFAULTS["response"] in numeric else _first(numeric, "")
    factor = DEFAULTS["factor"] if DEFAULTS["factor"] in factors else _first(factors, "batch")
    if name == "response":
        return response
    if name in ("factor", "factors", "fixed_effects"):
        return factor
    if name == "covariates":
        return next((column for column in numeric if column != response), "")
    if name == "group":
        return DEFAULTS["group"] if DEFAULTS["group"] in factors else factor
    if name == "dose":
        return DEFAULTS["dose"] if DEFAULTS["dose"] in numeric else response
    return ""


def _first(values: list[str], fallback: str) -> str:
    return values[0] if values else fallback


def coerce(field: Field, raw: Any) -> Any:
    """One field's raw input as a Python value, or `_OMIT` to leave it out."""
    if field.kind == "boolean":
        return bool(raw)
    if field.kind == "array":
        items = raw if isinstance(raw, list) else str(raw or "").split(",")
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return cleaned or _OMIT
    text = str(raw or "").strip()
    if not text:
        return _OMIT
    if field.kind in ("number", "integer"):
        try:
            return int(text) if field.kind == "integer" else float(text)
        except ValueError:
            raise ToolError(f"{field.name} must be a number, got {text!r}") from None
    return text


def arguments(fields: list[Field], raw: dict[str, Any]) -> dict[str, Any]:
    """Field values as tool keyword arguments, complaining about empty required ones."""
    values: dict[str, Any] = {}
    missing: list[str] = []
    for field in fields:
        value = coerce(field, raw.get(field.name))
        if value is _OMIT:
            if field.required:
                missing.append(field.name)
            continue
        values[field.name] = value
    if missing:
        raise ToolError(f"Fill in: {', '.join(missing)}")
    return values
