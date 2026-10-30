"""Select one device without depending on inventory order."""

from __future__ import annotations

from collections.abc import Sequence

from yewo.devices.errors import AmbiguousDeviceError, DeviceNotFoundError
from yewo.devices.inventory import DeviceDescriptor, DeviceQuery


def _query_text(query: DeviceQuery) -> str:
    criteria: list[str] = []
    if query.platform is not None:
        criteria.append(f"platform={query.platform.value}")
    if query.target_id is not None:
        criteria.append(f"target_id={query.target_id}")
    if query.display_name is not None:
        criteria.append(f"display_name={query.display_name}")
    return ", ".join(criteria) if criteria else "no criteria"


def _candidate_text(devices: Sequence[DeviceDescriptor]) -> str:
    if not devices:
        return "none"
    return ", ".join(
        f"{device.platform.value}:{device.target_id} ({device.display_name})"
        for device in sorted(
            devices,
            key=lambda candidate: (
                candidate.platform.value,
                candidate.target_id,
                candidate.display_name,
            ),
        )
    )


def select_device(query: DeviceQuery, devices: Sequence[DeviceDescriptor]) -> DeviceDescriptor:
    """Return the sole target that satisfies every query criterion."""
    candidates = tuple(devices)
    matches = candidates
    if query.platform is not None:
        matches = tuple(device for device in matches if device.platform is query.platform)
    if query.target_id is not None:
        matches = tuple(device for device in matches if device.target_id == query.target_id)
    if query.display_name is not None:
        matches = tuple(device for device in matches if device.display_name == query.display_name)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceNotFoundError(
            f"No device matched {_query_text(query)}. "
            f"Available targets: {_candidate_text(candidates)}."
        )
    raise AmbiguousDeviceError(
        f"The query {_query_text(query)} matched {len(matches)} devices. "
        f"Matching targets: {_candidate_text(matches)}."
    )
