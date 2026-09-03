"""Shared helpers for domain models."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

JsonScalar = str | int | float | bool | None

ShortText = Annotated[str, Field(min_length=1, max_length=2000)]
"""Bounded free-text used for statements, descriptions, reasons."""

Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:\-]+$")]
"""Identifier for facts/hypotheses/artifacts/etc. Model-chosen ids are allowed but validated."""


def utcnow() -> datetime:
    """Timezone-aware current time (UTC)."""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a short random identifier with a type prefix, e.g. ``evt_3f9a...``."""
    return f"{prefix}_{secrets.token_hex(8)}"


def _drop_docstring_description(schema: dict[str, Any]) -> None:
    """Docstrings, field descriptions and titles are for humans (and live in the output contract prose, which is a cached
    prefix). Keeping them out of the per-call output schema roughly halves it."""
    schema.pop("description", None)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        _strip_prop(prop)


def _strip_prop(prop: Any) -> None:
    """Remove human-only keys and string length bounds (still enforced in Python; llama.cpp's grammar converter cannot
    expand ``maxLength: 2000``) from a property schema, including inside anyOf/items."""
    if not isinstance(prop, dict):
        return
    for key in ("description", "title", "maxLength", "minLength"):
        prop.pop(key, None)
    for sub in prop.get("anyOf", []) + prop.get("oneOf", []):
        _strip_prop(sub)
    _strip_prop(prop.get("items"))


class StrictModel(BaseModel):
    """Base for model-facing schemas: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, json_schema_extra=_drop_docstring_description)


class Record(BaseModel):
    """Base for runtime records: tolerant of extra keys on read, but never written with them."""

    model_config = ConfigDict(extra="ignore")
