"""Settling the outcomes that are about where the run went, not what it saw.

"Tapping Continue reaches the confirmation screen" is true of a journey, not
of a screenshot, so the auditor is never asked it. What answers it is the
order of screens the run passed through.

The screen names come from the auditor, which wrote them without being told
the plan. A judge reading only those names inherits that independence: it can
report that a scenario never reached the screen the plan expected.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import SystemMessage, TextMessage
from yewo.testing.judge import rules
from yewo.testing.models import RuleVerification

logger = logging.getLogger(__name__)

PROMPT = """You judge claims about the order of screens in a test run.

### How to answer
1. `verdicts`: one entry per listed claim, copying its bracketed id into
   `rule_id`.
   - "passed" means the order of screens shows this claim holding. A claim
     that one screen is reached from another needs both, in that order.
   - "failed" means the order shows it not holding: the run went somewhere
     else instead, or never arrived at all.
   - "untested" means the order cannot say either way.
   - `reasoning`: name the screens you read it from.
2. Judge only what the order shows. You cannot see the screens themselves, so
   a claim about what a screen displayed is "untested" here, however plausible.
3. Screen names are written by a reader who did not know what was expected, so
   they will not match the words of a claim. Match them on what they describe,
   not on wording.
"""

RUN_CONTEXT = """### Screens visited, in order
{screens}

### Claims to judge
{claims}
"""


class TransitionVerdicts(BaseModel):
    """What the order of screens says about the claims made of it."""

    verdicts: List[RuleVerification] = Field(
        default_factory=list,
        description="One entry per claim listed, carrying that claim's bracketed id.",
    )


def _ordered_screens(assessments) -> List[str]:
    """The screens the run passed through, named once each time it arrived.

    A screen named twice in a row is one visit the auditor read twice, and
    repeating it would suggest the run left and came back.
    """
    names: List[str] = []
    for assessment in assessments:
        name = (assessment.screen_name or "").strip()
        if name and name != (names[-1] if names else None):
            names.append(name)
    return names


def settle(
    llm: LLMClient,
    expected_outcomes,
    assessments,
    scrubber=None,
) -> List[RuleVerification]:
    """Verdicts on the transition outcomes, or none when nothing can judge them.

    Returns the same evidence the auditor returns, under the same ids, so the
    gate that weighs it needs to know nothing about where it came from.
    """
    moved = rules.transitions_only(expected_outcomes)
    claims = rules.listing(moved, rules.EXPECTED)
    if claims == rules.NONE_SPECIFIED:
        return []
    asked = rules.listed_ids(moved, rules.EXPECTED)

    screens = _ordered_screens(assessments)
    if not screens:
        logger.info("No screen was named, so no transition can be settled.")
        return []

    numbered = "\n".join(f"{index}. {name}" for index, name in enumerate(screens, 1))
    logger.info("Judging %d screen(s) against the transitions the scenario expects.", len(screens))
    messages = [
        SystemMessage(content=PROMPT),
        TextMessage(content=RUN_CONTEXT.format(screens=numbered, claims=claims)),
    ]
    answer: TransitionVerdicts = llm.generate_response(messages, TransitionVerdicts)

    # This judge is shown expected outcomes and nothing else, so an answer
    # naming a bare index names one of them.
    for verdict in answer.verdicts:
        verdict.rule_id = rules.as_id(verdict.rule_id or verdict.rule, rules.EXPECTED)

    # Only the claims that were put to it. An answer naming anything else is
    # about a rule this judge never saw, and a fabricated security verdict
    # would fail the scenario outright.
    verdicts = [verdict for verdict in answer.verdicts if verdict.rule_id in asked]
    for verdict in verdicts:
        verdict.screen_name = _last_screen(verdict, screens)
        if scrubber is not None:
            verdict.rule = scrubber.scrub(verdict.rule)
            verdict.reasoning = scrubber.scrub(verdict.reasoning)
    return verdicts


def _last_screen(verdict: RuleVerification, screens: List[str]) -> Optional[str]:
    """Where to file this verdict, which is the screen the claim is about.

    A transition is settled by the whole order rather than by one screen, and
    a report groups evidence by screen. The screen the claim names last is the
    one it says the run arrived at.
    """
    named = [screen for screen in screens if screen.lower() in (verdict.reasoning or "").lower()]
    return named[-1] if named else (screens[-1] if screens else None)
