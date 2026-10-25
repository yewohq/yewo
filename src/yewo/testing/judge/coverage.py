"""Accounting for security rules across a whole journey.

Scoping a rule to the scenarios that can reach it stops one scenario from
failing because it cannot evidence an unrelated screen rule. Taken alone that
leaves a hole: a rule no scenario ever reaches disappears quietly. The suite
closes it. Every rule must be evidenced by at least one scenario or be named as
unevidenced.
"""

from typing import Iterable, List

from yewo.testing.judge import rules
from yewo.testing.models import ScenarioReport


def unevidenced_security_rules(
    security_rules: List[str], scenarios: Iterable[ScenarioReport]
) -> List[str]:
    """Returns the rules no scenario managed to check where they could apply."""
    evidenced = set()
    for scenario in scenarios:
        for verification in scenario.rule_verifications:
            if (
                rules.is_kind(verification.rule_id, rules.SECURITY)
                and verification.status == "passed"
                and verification.applicable
            ):
                evidenced.add(verification.rule_id)
    return [
        rule
        for index, rule in enumerate(security_rules)
        if rules.rule_id(rules.SECURITY, index) not in evidenced
    ]
