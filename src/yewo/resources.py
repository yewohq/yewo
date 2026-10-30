"""Shared ownership and cleanup for runtime resources."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Self


class ResourceOwnership(StrEnum):
    """Whether lifecycle belongs to this run or an external owner."""

    EXTERNAL = "external"
    RUN = "run"


class ResourceLease:
    """An idempotent release operation for one runtime resource."""

    def __init__(
        self,
        ownership: ResourceOwnership,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.ownership = ownership
        self._release = release
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this lease was released."""
        return self._closed

    def close(self) -> None:
        """Release a resource once when the current run owns it."""
        if self._closed:
            return
        if self.ownership is ResourceOwnership.RUN and self._release is not None:
            self._release()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()
