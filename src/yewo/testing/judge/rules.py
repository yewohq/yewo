"""The identifiers a verdict comes back under.

Two places depend on this format, and on both using the same format. The
auditor reads the rules. The gate matches a verdict to its rule by id. A
separate copy in each file would let one drift. Drift fails silently. The
verdicts stop matching, and every scenario reports missing evidence while its
assertions pass. So the scheme lives here, and nowhere else.
"""

import re
from typing import Iterable, List, Optional, Set

EXPECTED = "expected"
SECURITY = "security"
NONE_SPECIFIED = "None specified."

IN_TEXT = re.compile(rf"\b(?:{EXPECTED}|{SECURITY}):(\d+)\b")
"""An id anywhere inside a line of text.

Searched rather than matched, because a verdict comes back carrying the id on
its own, or still inside the prompt line it was read from.
"""


def rule_id(kind: str, index: int) -> str:
    """The id for one rule, e.g. `security:1`."""
    return f"{kind}:{index}"


def ids(rules: Iterable, kind: str) -> Set[str]:
    """Every id of this kind. The gate matches verdicts against these."""
    return {rule_id(kind, index) for index, _ in enumerate(rules or [])}


def id_in(text) -> Optional[str]:
    """The rule id a line of text carries, or None when it carries none.

    A verdict names its rule in whichever field the model chose, and the gate
    matches on the id alone. An id read out of the wrong field is still the
    right id; one that is never read discards the verdict in silence.
    """
    if not isinstance(text, str):
        return None
    match = IN_TEXT.search(text)
    return match.group(0) if match else None


def as_id(text, kind: str) -> Optional[str]:
    """The id an answer names, for a judge that was given one kind of rule.

    A judge shown only expected outcomes answers "expected:4", and sometimes
    answers "4". The bare number is unambiguous there, because one kind was
    listed. It is not unambiguous for a judge shown both kinds, which is why
    this takes the kind rather than assuming one.
    """
    embedded_id = id_in(text)
    if embedded_id:
        return embedded_id
    numeric_rule_id = (text or "").strip() if isinstance(text, str) else ""
    return rule_id(kind, int(numeric_rule_id)) if numeric_rule_id.isdigit() else None


def is_kind(identifier: Optional[str], kind: str) -> bool:
    """Whether an id belongs to this kind of rule.

    Takes an optional id because a verdict may carry none: the auditor raises
    assertions of its own, and those name no rule.
    """
    return (identifier or "").startswith(f"{kind}:")


def statement_of(rule) -> str:
    """The words of a rule, whether it arrived as a string or an outcome."""
    return rule if isinstance(rule, str) else rule.statement


def format_rules_for_prompt(rules: Iterable, kind: str) -> List[str]:
    """The rules as prompt lines. Each line carries the id its verdict must use."""
    return [
        f"- [{rule_id(kind, index)}] {statement_of(rule)}"
        for index, rule in enumerate(rules or [])
        if rule is not None
    ]


def map_statements_to_rule_ids(
    expected_outcomes: Iterable, security_rules: Iterable
) -> dict:
    """The id for each rule's exact words.

    A verdict names its rule by the id it was given. Some answers name it by
    quoting the rule instead, with the id left out. The words are the words
    this prompt listed, so the id is a lookup and not a guess.

    A text listed twice is dropped rather than resolved. Two rules reading
    alike cannot be told apart by their words, and settling the wrong one is
    worse than settling neither.
    """
    rule_ids_by_statement: dict = {}
    duplicate_statements = set()
    for kind, listed in ((EXPECTED, expected_outcomes), (SECURITY, security_rules)):
        for index, rule in enumerate(listed or []):
            if rule is None:
                continue
            text = statement_of(rule).strip()
            if text in rule_ids_by_statement:
                duplicate_statements.add(text)
            rule_ids_by_statement[text] = rule_id(kind, index)
    for text in duplicate_statements:
        rule_ids_by_statement.pop(text, None)
    return rule_ids_by_statement


def listed_ids(rules: Iterable, kind: str) -> Set[str]:
    """The ids actually put to a judge, blanks excluded.

    A verdict naming a rule the judge was never shown is not evidence about
    that rule. Accepting one lets an answer settle an outcome nothing looked
    at, or fail a scenario on a security rule that was never asked.
    """
    return {
        rule_id(kind, index)
        for index, rule in enumerate(rules or [])
        if rule is not None
    }


def listing(rules: Iterable, kind: str) -> str:
    """One prompt section: the rules of this kind, or a note that none exist."""
    return "\n".join(format_rules_for_prompt(rules, kind)) or NONE_SPECIFIED


def _scoped(outcomes: Iterable, wanted, default: bool) -> List:
    """The outcomes of one scope, keeping every index.

    An outcome a judge cannot answer is blanked rather than dropped, because
    the id a verdict carries is its position in this list. Removing one would
    renumber the rest and move every later verdict onto the wrong rule.

    `default` decides a bare string, which carries no scope of its own.
    """
    from yewo.testing.models import ExpectedOutcome

    kept = []
    for outcome in outcomes or []:
        mine = outcome.scope is wanted if isinstance(outcome, ExpectedOutcome) else default
        kept.append(outcome if mine else None)
    return kept


def on_screen_only(outcomes: Iterable) -> List:
    """The outcomes one screenshot could settle, keeping every index."""
    from yewo.testing.models import AssertionScope

    return _scoped(outcomes, AssertionScope.CURRENT_SCREEN, default=True)


def transitions_only(outcomes: Iterable) -> List:
    """The outcomes only the order of screens could settle, keeping every index.

    A bare string is a claim about one screen, so it is not one of these. The
    two selectors must not both claim it: an outcome answered twice is counted
    twice, and one claimed by neither can never be settled at all.
    """
    from yewo.testing.models import AssertionScope

    return _scoped(outcomes, AssertionScope.TRANSITION, default=False)


def every_rule(expected_outcomes: Iterable, security_rules: Iterable) -> str:
    """Both kinds in one list, for a prompt that keeps them together."""
    lines = format_rules_for_prompt(
        expected_outcomes, EXPECTED
    ) + format_rules_for_prompt(security_rules, SECURITY)
    return "\n".join(lines) or NONE_SPECIFIED
