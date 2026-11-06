"""Provider-neutral model transport boundary."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from yewo.models.records import ModelRequest, ModelResponse


class ModelClient(Protocol):
    """Complete provider-neutral model requests."""

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse: ...


__all__ = ["ModelClient"]
