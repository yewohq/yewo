"""Strict metadata contract for detector artifacts supplied to Yewo."""

from pathlib import Path
from typing import Annotated, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class DetectorClass(BaseModel):
    """One raw class declared by a detector artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_id: int = Field(ge=0)
    label: str = Field(min_length=1)
    semantic_roles: frozenset[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=frozenset
    )


class PerceptionModelManifest(BaseModel):
    """Versioned mechanical metadata that travels with detector weights."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    weights_file: str = Field(min_length=1)
    inference_image_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    classes: tuple[DetectorClass, ...] = Field(min_length=1)

    @field_validator("weights_file")
    @classmethod
    def _validate_weights_filename(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("weights_file must be a plain filename")
        return value

    @model_validator(mode="after")
    def _validate_unique_classes(self) -> "PerceptionModelManifest":
        class_ids = [detector_class.class_id for detector_class in self.classes]
        labels = [detector_class.label for detector_class in self.classes]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("detector class IDs must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("detector class labels must be unique")
        return self

    def validate_model_classes(self, model_classes: Mapping[int, str] | Sequence[str]) -> None:
        """Require the loaded model and sidecar to declare the same class map."""
        class_labels_by_id = (
            dict(model_classes)
            if isinstance(model_classes, Mapping)
            else dict(enumerate(model_classes))
        )
        manifest_labels_by_id = {
            detector_class.class_id: detector_class.label for detector_class in self.classes
        }
        if class_labels_by_id != manifest_labels_by_id:
            raise ValueError(
                "detector classes do not match the perception manifest: "
                f"model={class_labels_by_id!r}, manifest={manifest_labels_by_id!r}"
            )

    def semantic_roles_by_class_id(self) -> dict[int, frozenset[str]]:
        """Return only the semantic roles declared by the model package."""
        return {
            detector_class.class_id: detector_class.semantic_roles
            for detector_class in self.classes
        }


def load_perception_model_manifest(path: Path) -> PerceptionModelManifest:
    """Load an untrusted model sidecar through the strict typed contract."""
    return PerceptionModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
