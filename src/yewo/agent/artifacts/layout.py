"""Where one run's files go: its directories, and the two logs it writes."""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass(frozen=True)
class RunLayout:
    """The directories one run owns, under one session id."""

    root: Path
    session_id: str

    @classmethod
    def under(cls, base: Path, session_id: str) -> "RunLayout":
        return cls(root=base / session_id, session_id=session_id)

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def screenshots(self) -> Path:
        return self.root / "screenshots"

    @property
    def recordings(self) -> Path:
        return self.root / "recordings"

    @property
    def telemetry(self) -> Path:
        return self.root / "telemetry"

    @property
    def manifest(self) -> Path:
        """The run's own record, at the root rather than inside telemetry.

        Every other file under here is evidence this one points at.
        """
        return self.root / "run.json"

    @property
    def prompts(self) -> Path:
        """What each model call was asked, where the run keeps them."""
        return self.root / "prompts"

    @property
    def appium_log(self) -> Path:
        return self.logs / f"appium_session_{self.session_id}.log"

    @property
    def run_log(self) -> Path:
        return self.logs / f"yewo_session_{self.session_id}.log"

    def device_log(self, stem: str, segment: int = 1) -> Path:
        """Named for the log rather than the platform, because a reader of
        `logs/` is asking what the device said. Numbered like a recording,
        because session recovery reopens on the same session id."""
        return self.logs / f"device_{stem}_{self.session_id}_{segment:03d}.log"

    def create(self) -> None:
        """Makes every directory, so nothing downstream has to check."""
        for directory in (
            self.logs,
            self.screenshots,
            self.recordings,
            self.telemetry,
            self.prompts,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def attach_run_log(path: Path) -> logging.Handler:
    """Sends everything logged anywhere to this run's own file."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler) -> None:
    """Stops writing to it, so the next run does not inherit the last one's file."""
    try:
        logging.getLogger().removeHandler(handler)
        handler.close()
    except Exception as exc:  # noqa: BLE001 - closing a log must not fail a run
        logger.warning("Failed to close the run log: %s", exc)
