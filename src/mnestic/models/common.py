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
    """Class docstrings are for humans; keeping them out of the output schema saves tokens on every call."""
    schema.pop("description", None)


class StrictModel(BaseModel):
    """Base for model-facing schemas: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, json_schema_extra=_drop_docstring_description)


class Record(BaseModel):
    """Base for runtime records: tolerant of extra keys on read, but never written with them."""

    model_config = ConfigDict(extra="ignore")
