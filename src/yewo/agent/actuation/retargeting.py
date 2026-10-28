"""Build action variants that identify targets without raw coordinates."""

from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field, create_model


class RetargetedAction(BaseModel):
    """An action whose target resolves back to its coordinate form."""

    coordinate_form: ClassVar[type[BaseModel]]

    @property
    def count(self) -> int:
        """Model-facing taps are one action. Other variants ignore this value."""
        return 1

    @property
    def interval_ms(self) -> int:
        return 0


def _retargeted(
    model: type[BaseModel],
    name: str,
    target: str,
    definition: tuple[type, Any],
    coordinate_form: type[BaseModel] | None = None,
) -> type[RetargetedAction]:
    fields: dict[str, Any] = {
        field_name: (info.annotation, info)
        for field_name, info in model.model_fields.items()
        if field_name != "coordinates"
    }
    fields[target] = definition
    if target != "target":
        fields["target"] = (
            str,
            Field(
                ...,
                description=(
                    "What the numbered target is, described by its own visible "
                    "text, symbol, shape, or colour."
                ),
            ),
        )
    variant = cast(
        type[RetargetedAction],
        create_model(name, __base__=RetargetedAction, **fields),
    )
    variant.coordinate_form = coordinate_form or model
    return variant


def marks_variant(
    model: type[BaseModel],
    name: str,
    coordinate_form: type[BaseModel] | None = None,
) -> type[RetargetedAction]:
    """Replace coordinates with a required numbered screenshot mark."""
    return _retargeted(
        model,
        name,
        "mark",
        (int, Field(..., description="Number of the target element on the screenshot.")),
        coordinate_form,
    )


def describe_variant(
    model: type[BaseModel],
    name: str,
    coordinate_form: type[BaseModel] | None = None,
) -> type[RetargetedAction]:
    """Replace coordinates with a required visual target description."""
    return _retargeted(
        model,
        name,
        "target",
        (
            str,
            Field(
                ...,
                description=(
                    "What to act on, described as you would point it out to "
                    "someone looking at the screen."
                ),
            ),
        ),
        coordinate_form,
    )
