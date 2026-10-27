"""Normalize equivalent structured-output labels at the model boundary."""

from collections.abc import Callable, Mapping
from typing import Optional

FINDING_FIELD_ALIASES = ("location", "subject", "name", "target")
OUTCOME_STATEMENT_ALIASES = (
    "description",
    "outcome",
    "expected_outcome",
    "assertion",
    "detail",
    "text",
    "claim",
    "check",
    "expected",
)
RULE_ID_ALIASES = ("id", "ruleId", "rule_identifier")
RULE_TEXT_ALIASES = ("statement", "rule_text", "description", "name", "rule_name")
REASONING_ALIASES = ("reason", "details", "explanation", "evidence", "justification", "notes")
STATUS_ALIASES = ("status", "verdict", "result", "state")
INAPPLICABLE_STATUSES = frozenset(("not_applicable", "n/a", "na"))


def _first_value(values: Mapping[str, object], names: tuple[str, ...]) -> object:
    return next((values[name] for name in names if values.get(name)), None)


def normalize_sensitive_finding(
    value: object, rendering_values: frozenset[str]
) -> object:
    if not isinstance(value, dict):
        return value
    finding = dict(value)
    type_value = finding.pop("type", None)
    if type_value in rendering_values:
        finding.setdefault("rendering", type_value)
    elif type_value is not None:
        finding.setdefault("field", type_value)
    finding.setdefault("rendering", _first_value(finding, ("state", "status")))
    finding["field"] = finding.get("field") or _first_value(
        finding, FINDING_FIELD_ALIASES
    ) or "unspecified"
    if finding.get("quoted") is None:
        finding["quoted"] = ""
    return finding


def normalize_expected_outcome(value: object, measured_scope: object) -> object:
    if isinstance(value, str):
        return {"statement": value}
    if not isinstance(value, dict):
        return value
    outcome = dict(value)
    outcome["statement"] = outcome.get("statement") or _first_value(
        outcome, OUTCOME_STATEMENT_ALIASES
    )
    if "scope" not in outcome:
        supplied_scope = _first_value(outcome, ("kind", "type"))
        if supplied_scope in {"current_screen", "transition", "measured"}:
            outcome["scope"] = supplied_scope
        elif outcome.get("metric") is not None:
            outcome["scope"] = measured_scope
    return outcome


def normalize_rule_verification(
    value: object,
    *,
    status_spellings: Mapping[str, str],
    resolve_rule_id: Callable[[object], Optional[str]],
) -> object:
    if not isinstance(value, dict):
        return value
    verdict = dict(value)
    verdict["rule_id"] = verdict.get("rule_id") or _first_value(
        verdict, RULE_ID_ALIASES
    )
    resolved_rule_id = resolve_rule_id(verdict.get("rule_id")) or resolve_rule_id(
        verdict.get("rule")
    )
    if resolved_rule_id:
        verdict["rule_id"] = resolved_rule_id
    verdict["rule"] = verdict.get("rule") or _first_value(
        verdict, RULE_TEXT_ALIASES
    ) or verdict.get("rule_id")
    if "reasoning" not in verdict or verdict["reasoning"] is None:
        alias_reasoning = _first_value(verdict, REASONING_ALIASES)
        if alias_reasoning is not None:
            verdict["reasoning"] = alias_reasoning
    raw_status = _first_value(verdict, STATUS_ALIASES)
    normalized_status = str(raw_status or "").strip().lower().replace(" ", "_")
    verdict["status"] = status_spellings.get(normalized_status, raw_status)
    if normalized_status in INAPPLICABLE_STATUSES:
        verdict.setdefault("applicable", False)
    return verdict
