"""Own the decision-model grounders used during one device run."""

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.grounding import GROUNDING_DESCRIBE
from yewo.agent.grounding.description import DecisionGrounder
from yewo.llm.jev_client import JevClient
from yewo.utils.secrets import SecretScrubber


class GroundingModels:
    """Create planner and replay grounders only when their modes need them."""

    def __init__(self, grounding: str, grounder, usage_callback):
        self._usage_callback = usage_callback
        self._prompt_store = None
        self.planner = grounder or self._build_planner(grounding)
        self._replay = self.planner

    def _build_planner(self, grounding: str):
        if grounding != GROUNDING_DESCRIBE:
            return None
        return DecisionGrounder(JevClient(usage_callback=self._usage_callback))

    def record_prompts(self, store) -> None:
        """Send later grounding prompts to the run's prompt store."""
        self._prompt_store = store
        if self.planner is not None:
            self.planner.client.prompt_store = store

    def replay(self):
        """Return the semantic replay grounder, creating it on first use."""
        if self._replay is None:
            self._replay = DecisionGrounder(
                JevClient(
                    usage_callback=self._usage_callback,
                    prompt_store=self._prompt_store,
                )
            )
        return self._replay


def ground_and_protect_typed_secret(
    action,
    snapshot,
    scrubber,
    images,
    grounding_models: GroundingModels,
    target_bounds=None,
):
    """Resolve and protect a detector-free field before typing a secret."""
    if not isinstance(action, TypeAction) or not isinstance(scrubber, SecretScrubber):
        return action, target_bounds
    if (
        target_bounds is None
        and not snapshot.gui_elements
        and action.target
        and scrubber.declared_labels_for_typed_text(action.text)
    ):
        safe_image = images.redact_model_image(snapshot.raw_img_bytes, snapshot.gui_elements)
        placement = grounding_models.replay().ground(safe_image, action.target)
        if placement is not None:
            target_bounds = placement.box
            x, y = placement.point
            action = action.model_copy(
                update={"coordinates": Coordinates(x=x, y=y)}
            )
    scrubber.protect_typed_region(
        action.text,
        action.coordinates,
        snapshot.gui_elements,
        bounds=target_bounds,
    )
    return action, target_bounds
