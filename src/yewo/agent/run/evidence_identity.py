"""Stable identities for observations and execution compatibility."""

from __future__ import annotations

from pydantic import BaseModel, Field

from yewo.identity import derived_uuid, new_uuid


class ObservationIdentity(BaseModel):
    """Identity of one exact capture, independent of screen classification."""

    observation_id: str = Field(default_factory=new_uuid)


def new_observation_id() -> str:
    """Create one exact-capture identity through its typed owner."""
    return ObservationIdentity().observation_id


class ExecutionContextIdentity(BaseModel):
    """Versioned compatibility identity for one execution context."""

    context_id: str
    app_id: str
    platform: str
    actor: str
    setup: str
    contract_version: str
    identity_version: str = "execution-context-v1"

    @classmethod
    def build(
        cls,
        *,
        app_id: str,
        platform: str,
        actor: str,
        setup: str,
        instruction_ids: tuple[str, ...],
    ) -> "ExecutionContextIdentity":
        contract_version = derived_uuid("instruction-contract-set-v1", *instruction_ids)
        identity_version = "execution-context-v1"
        return cls(
            context_id=derived_uuid(
                identity_version,
                app_id,
                platform,
                actor,
                setup,
                contract_version,
            ),
            app_id=app_id,
            platform=platform,
            actor=actor,
            setup=setup,
            contract_version=contract_version,
            identity_version=identity_version,
        )
