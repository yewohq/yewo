"""What each step leaves behind on disk.

Every file a step writes starts with that step and the moment it was read, so
one step's files sit together and one glob finds them all. What kind of file
it is comes last. The step is padded, because a listing orders names by
character and would otherwise put step 10 before step 2.

Both writers go through the media manager rather than straight to disk. These
files outlive the run, so the manager covers a declared secret on screen first.
The scan dump goes beside them, which keeps a step's screenshots and the
reading they came from together.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.artifacts.scan import (
    build_screen_scan_record,
    write_screen_scan_dump,
)
from yewo.agent.step.observation import PerceptionSnapshot
from yewo.utils.secrets import SecretScrubber

logger = logging.getLogger(__name__)


class StepArtifacts:
    """Writes a step's files through the media manager that redacts them."""

    def __init__(self, images: RedactedImages, scrubber: SecretScrubber):
        self.images = images
        self.scrubber = scrubber

    @staticmethod
    def _redaction_evidence(
        snapshot: PerceptionSnapshot,
        *,
        planner_view: bool = False,
    ) -> list:
        frame = (
            getattr(snapshot, "planner_perception_frame", None)
            if planner_view
            else getattr(snapshot, "perception_frame", None)
        )
        elements = getattr(snapshot, "planner_elements", None) if planner_view else snapshot.gui_elements
        elements = elements if elements is not None else snapshot.gui_elements
        ocr_evidence = frame.ocr_redaction_evidence() if frame is not None else ()
        return [*elements, *ocr_evidence]

    def save_step(
        self, session_dir: Path, step: int, snapshot: PerceptionSnapshot
    ) -> dict[str, str]:
        """Writes the screen, and names the files it wrote.

        The names are returned rather than rebuilt by a reader. A manifest
        that derived them would be a second place the convention lives.
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        base = f"step{step + 1:03d}_{snapshot.timestamp}"
        annotated_path = session_dir / f"{base}.png"
        raw_path = session_dir / f"{base}_raw.png"
        scan_path = session_dir / f"{base}_scan.json"

        self.images.save_redacted_pair(
            snapshot.annotated_img,
            snapshot.raw_img_bytes,
            self._redaction_evidence(snapshot),
            annotated_path,
            raw_path,
        )

        written = {"raw": raw_path.name, "annotated": annotated_path.name}
        if snapshot.scan_info is None:
            return written

        record = build_screen_scan_record(
            step=step + 1,
            timestamp=snapshot.timestamp,
            scan_info=snapshot.scan_info,
            gui_elements=snapshot.gui_elements,
            screenshot_paths={
                "raw": raw_path.name,
                "annotated": annotated_path.name,
                "scan": scan_path.name,
            },
            prev_phash=snapshot.prev_phash,
            scrubber=self.scrubber,
            perception_frame=getattr(snapshot, "perception_frame", None),
        )
        write_screen_scan_dump(scan_path, record)
        logger.debug("Perception scan dump saved to %s", scan_path)
        return {**written, "scan": scan_path.name}

    def save_planner_view(
        self, session_dir: Path, step: int, snapshot: PerceptionSnapshot
    ) -> Optional[str]:
        """The numbered screenshot, which is the image the planner is shown.

        Written only in marks mode, where `snapshot.marks` is what numbering
        produced. The other two files are a raw screen and a detector overlay,
        and the planner sees neither, so only this one explains the badge it
        answered with.
        """
        planner_view = snapshot.planner_img_bytes
        if not snapshot.marks or not planner_view:
            return None
        session_dir.mkdir(parents=True, exist_ok=True)
        marked = session_dir / f"step{step + 1:03d}_{snapshot.timestamp}_marks.png"
        self.images.save_redacted_bytes(
            planner_view,
            self._redaction_evidence(snapshot, planner_view=True),
            marked,
        )
        return marked.name

    def save_refinement(
        self, session_dir: Path, step: int, snapshot: PerceptionSnapshot
    ) -> dict[str, str]:
        """Persist how a magnified planner view maps to the full screenshot."""
        if snapshot.refinement is None:
            return {}
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / (f"step{step + 1:03d}_{snapshot.timestamp}_refinement.json")
        path.write_text(
            json.dumps(snapshot.refinement, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"refinement": path.name}

    def save_action(
        self, session_dir: Path, step: int, snapshot: PerceptionSnapshot, action
    ) -> Optional[str]:
        """Where the action was aimed, when there was somewhere to draw.

        None where nothing was written. An action with no point on screen
        has no annotation to make. A wait and a key event are both such
        actions. Naming a file that does not exist puts a broken image in
        every report that lists the step's artifacts.
        """
        click_path = session_dir / f"step{step + 1:03d}_{snapshot.timestamp}_action.png"
        drawn = self.images.save_redacted_action_annotation(
            snapshot.annotated_img, self._redaction_evidence(snapshot), action, click_path
        )
        return click_path.name if drawn else None


def save_unmatched_screen(images: RedactedImages, path: Path, snapshot) -> None:
    """Keeps the screen a target was missing from, so the gap can be studied later.

    Through the media manager, not straight to disk: this file outlives the run
    like any other artifact, so declared secrets on screen get covered first.
    """
    try:
        images.save_redacted_bytes(
            snapshot.raw_img_bytes,
            StepArtifacts._redaction_evidence(snapshot),
            path,
        )
    except Exception as exc:  # noqa: BLE001 - an artifact must never fail a run
        logger.warning("Could not save the unmatched screen: %s", exc)
