"""Readable summaries of deterministic execution-contract receipts."""

from .source import ScenarioSource


def contract_evidence_rows(scenario: ScenarioSource) -> tuple[str, ...]:
    """Format settlement, input, and obligation evidence for a report."""
    rows = [
        f"Settlement {receipt.settlement_id}: {receipt.status.value} — {receipt.reason}"
        for receipt in scenario.settlement_receipts
    ]
    rows.extend(
        f"Input {receipt.transaction_id}: {receipt.status.value} — {receipt.reason}"
        for receipt in scenario.input_transactions
    )
    rows.extend(
        f"Obligation {receipt.obligation_id}: {receipt.state.value} — {receipt.reason}"
        for receipt in scenario.obligation_evidence.settlements
    )
    readiness = scenario.obligation_evidence.readiness
    if readiness is not None:
        rows.append(
            "Obligation routing: "
            f"{readiness.active} active, {readiness.inactive} inactive, "
            f"{readiness.unknown} unknown, {readiness.closed} closed"
        )
    return tuple(rows)
