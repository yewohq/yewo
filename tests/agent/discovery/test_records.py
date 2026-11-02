"""Discovery request and evidence record tests."""

import pytest
from pydantic import ValidationError

from yewo.agent.discovery import DiscoveryObjective, DiscoveryRequest


def test_discovery_rejects_a_request_without_a_required_objective() -> None:
    with pytest.raises(ValidationError, match="at least one required objective"):
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="optional",
                    statement="Inspect optional behavior.",
                    required=False,
                ),
            ),
        )
