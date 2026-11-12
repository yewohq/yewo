"""When a run says what it cannot do, and when it stays quiet."""

from __future__ import annotations

from pathlib import Path

from yewo.cli.extension_notice import (
    CLOUD_URL,
    RunDeclarations,
    failure_notice,
    missing_notice,
)
from yewo.devices.logs.reader import DeviceLogReader
from yewo.devices.recording.recorder import ScreenRecorder
from yewo.extensions import DeclaredValues, ExtensionRegistry, RunContext
from yewo.observations.service import ObservationSource


class _Source:
    def screenshot(self) -> bytes:
        return b""

    def screen_calibration(self):
        raise NotImplementedError

    def active_application_id(self):
        raise NotImplementedError

    def application_state(self):
        raise NotImplementedError

    def application_context(self):
        raise NotImplementedError

    def keyboard_visible(self):
        raise NotImplementedError

    def device_orientation(self):
        raise NotImplementedError

    def system_regions(self):
        raise NotImplementedError


class _Recorder:
    @property
    def capture_method(self):
        raise NotImplementedError

    @property
    def can_start_after_failure(self) -> bool:
        raise NotImplementedError

    def start(self, maximum_duration_seconds, maximum_stop_seconds=10.0, maximum_pull_seconds=30.0):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _Reader:
    def begin(self, log_type):
        raise NotImplementedError

    def read(self, log_type):
        raise NotImplementedError

    def captures_checkpoint(self, log_type, checkpoint_kind) -> bool:
        raise NotImplementedError

    def acknowledge(self, log_type, batch) -> None:
        raise NotImplementedError


def _registry(*, filled: bool = False, failure: str | None = None) -> ExtensionRegistry:
    registry = ExtensionRegistry()
    if filled:
        registry.provide(ObservationSource, _Source())
        registry.provide(ScreenRecorder, _Recorder())
        registry.provide(DeviceLogReader, _Reader())
    if failure is not None:
        registry.record_failure(failure)
    return registry


def test_a_run_with_nothing_declared_is_told_nothing():
    """The condition that keeps this message worth reading.

    Printed on every run it becomes boilerplate, and by the time it matters nobody reads it.
    """
    assert missing_notice(_registry(), RunDeclarations()) is None


def test_declared_secrets_without_an_extension_say_what_happens_to_them():
    """The consequence is stated before the product, in its own sentence."""
    notice = missing_notice(_registry(), RunDeclarations(secret_labels=("card_number", "cvv")))

    assert notice is not None
    consequence, _, pitch = notice.partition("Yewo Cloud")
    assert "2 secret values" in consequence
    assert "as the device produced them" in consequence
    assert "Yewo Cloud" not in consequence
    assert CLOUD_URL in pitch


def test_the_notice_separates_what_is_kept_from_what_the_model_reads():
    """The recording and the device log are retained as artifacts.

    The message said all three were sent to the model. Only the screenshot reaches one, so the
    sentence overstated the exposure of the two artifacts that stay on disk.
    """
    notice = missing_notice(_registry(), RunDeclarations(secret_labels=("card_number",)))

    assert notice is not None
    kept, _, sent_to_model = notice.partition("Each screenshot also goes to")
    assert "the recording, and the device log" in kept
    assert "recording" not in sent_to_model
    assert "device log" not in sent_to_model


def test_an_installed_extension_silences_it():
    """Nothing to warn about once something is filling the slot."""
    assert (
        missing_notice(_registry(filled=True), RunDeclarations(secret_labels=("card_number",)))
        is None
    )


def test_an_extension_that_failed_to_load_is_not_silent():
    """A capability missing because its provider crashed looks like one never installed."""
    notice = failure_notice(_registry(failure="extended: boom"))

    assert notice is not None
    assert "extended: boom" in notice
    assert failure_notice(_registry()) is None


def test_the_notice_never_carries_a_declared_value():
    """Labels name what was declared. The values stay out of anything printed."""
    values = DeclaredValues()
    values.add("card_number", "4242424242424242")
    RunContext(artifact_root=Path("."), run_id="r1", declared_values=values)

    notice = missing_notice(
        _registry(), RunDeclarations(secret_labels=[v.label for v in values.all()])
    )

    assert notice is not None
    assert "4242424242424242" not in notice


def test_partial_install_is_not_silent():
    """Covering the screen and not the recording is not covering the run.

    The package this ships with claims all three together, so this state needs a package that
    claims fewer. The message stays the same either way: the caller installs something.
    """
    registry = ExtensionRegistry()
    registry.provide(ObservationSource, _Source())

    notice = missing_notice(registry, RunDeclarations(secret_labels=("card_number",)))

    assert notice is not None
    assert CLOUD_URL in notice
