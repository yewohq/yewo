"""Close automation and extension resources without abandoning later cleanup."""

from __future__ import annotations

from yewo.automation import AppiumServerLease, AutomationSession
from yewo.devices import DeviceLease
from yewo.extensions import ExtensionRegistry
from yewo.run_narration import RunNarrator
from yewo.runs.failure_diagnostics import error_chain

_NARRATION_SOURCE = "run-coordinator"


def close_automation_resources(
    session: AutomationSession | None,
    server: AppiumServerLease | None,
    device: DeviceLease | None,
    narrator: RunNarrator,
) -> tuple[Exception | None, bool]:
    """Close every automation resource and preserve the first failure."""
    errors: list[Exception] = []
    interrupted = False
    resources = (
        ("automation-session", session),
        ("appium-server", server),
        ("device", device),
    )
    for resource_name, resource in resources:
        if resource is None:
            continue
        narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=starting")
        try:
            resource.close()
            narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=completed")
        except KeyboardInterrupt:
            interrupted = True
            narrator.trace(_NARRATION_SOURCE, f"cleanup={resource_name} state=interrupted")
            try:
                resource.close()
            except KeyboardInterrupt:
                interrupted = True
            except Exception as error:  # noqa: BLE001 - cleanup must finish
                errors.append(error)
                narrator.trace(
                    _NARRATION_SOURCE,
                    f"cleanup={resource_name} state=failed chain={error_chain(error)}",
                )
        except Exception as error:  # noqa: BLE001 - cleanup must finish
            errors.append(error)
            narrator.trace(
                _NARRATION_SOURCE,
                f"cleanup={resource_name} state=failed chain={error_chain(error)}",
            )
    return (errors[0] if errors else None), interrupted


def close_extension_resources(
    extensions: ExtensionRegistry | None,
    narrator: RunNarrator,
) -> tuple[Exception | None, bool]:
    """Close extension resources after their automation consumers."""
    if extensions is None:
        return None, False
    try:
        extensions.close_run_resources()
    except KeyboardInterrupt:
        narrator.trace(_NARRATION_SOURCE, "cleanup=extension-resources state=interrupted")
        return None, True
    except Exception as error:  # noqa: BLE001 - cleanup must become a terminal fact
        narrator.trace(
            _NARRATION_SOURCE,
            f"cleanup=extension-resources state=failed type={type(error).__name__}",
        )
        return error, False
    narrator.trace(_NARRATION_SOURCE, "cleanup=extension-resources state=completed")
    return None, False


__all__ = ["close_automation_resources", "close_extension_resources"]
