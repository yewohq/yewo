from yewo.agent.memory.control_identity import ControlIdentity
from yewo.agent.run.settlement import SettlementController, SettlementStatus

SOURCE = "0" * 64
TEMPORARY = "f" * 64
DRAWER = "1" * 64
SCHEDULE_LIST = "2" * 64
UNEXPECTED_FORM = "e" * 64


def _control(phash: str, x: float = 0.5) -> ControlIdentity:
    return ControlIdentity("tap", phash, "relative", ((x, 0.5),), "1:100")


def _dispatched(controller: SettlementController) -> ControlIdentity:
    control = _control(SOURCE)
    assert controller.begin_action(control, SOURCE, "observation-source") is None
    assert (
        controller.record_dispatch(control, SOURCE, TEMPORARY, failed=False)
        is None
    )
    return control


def test_return_through_temporary_screen_is_transient() -> None:
    controller = SettlementController()
    _dispatched(controller)

    assert controller.observe(TEMPORARY, "observation-temporary") is None
    receipt = controller.observe(SOURCE, "observation-return")

    assert receipt is not None
    assert receipt.status is SettlementStatus.TRANSIENT_OSCILLATION
    assert receipt.observation_ids == (
        "observation-source",
        "observation-temporary",
        "observation-return",
    )


def test_destination_requires_two_independent_observations() -> None:
    controller = SettlementController()
    _dispatched(controller)

    assert controller.observe(TEMPORARY, "observation-1") is None
    receipt = controller.observe(TEMPORARY, "observation-2")

    assert receipt is not None
    assert receipt.status is SettlementStatus.DURABLE_PROGRESS


def test_different_control_from_destination_does_not_prove_progress() -> None:
    controller = SettlementController()
    _dispatched(controller)

    receipt = controller.begin_action(_control(TEMPORARY, 0.8), TEMPORARY)

    assert receipt is not None
    assert receipt.status is SettlementStatus.UNVERIFIED


def test_live_hash_jitter_still_detects_return_to_source() -> None:
    controller = SettlementController()
    source = "daa82d505aa82d5058a225545822b05c2d57f2af2d57f2af2d57d2af2d57d2ab"
    compatible_source = "daa82d505aa8255058aa21545822b05d2d57f2af2d57f2af2d57d2af2d57d2ab"
    snackbar = "d8a82d5058a021504840f2a82d56d2ab2d57d2ab2d57f22fb87fa55fdaafa55d"
    control = _control(source)
    controller.begin_action(control, source)
    assert controller.record_dispatch(control, source, snackbar, failed=False) is None

    receipt = controller.observe(compatible_source, "observation-return")

    assert receipt is not None
    assert receipt.status is SettlementStatus.TRANSIENT_OSCILLATION


def test_next_action_from_source_disproves_progress() -> None:
    controller = SettlementController()
    _dispatched(controller)

    receipt = controller.begin_action(_control(SOURCE, 0.8), SOURCE)

    assert receipt is not None
    assert receipt.status is SettlementStatus.TRANSIENT_OSCILLATION


def test_unchanged_dispatch_closes_without_progress() -> None:
    controller = SettlementController()
    control = _control(SOURCE)
    controller.begin_action(control, SOURCE)

    receipt = controller.record_dispatch(control, SOURCE, SOURCE, failed=False)

    assert receipt is not None
    assert receipt.status is SettlementStatus.NO_VISIBLE_CHANGE


def test_failed_and_incomplete_dispatches_are_explicit() -> None:
    failed = SettlementController()
    control = _control(SOURCE)
    failed.begin_action(control, SOURCE)
    failed_receipt = failed.record_dispatch(
        control, SOURCE, TEMPORARY, failed=True
    )

    incomplete = SettlementController()
    incomplete.begin_action(control, SOURCE)
    incomplete_receipt = incomplete.record_dispatch(
        control, SOURCE, None, failed=False
    )

    assert failed_receipt is not None
    assert failed_receipt.status is SettlementStatus.DISPATCH_FAILED
    assert incomplete_receipt is not None
    assert incomplete_receipt.status is SettlementStatus.UNVERIFIED


def test_same_control_before_settlement_is_unverified() -> None:
    controller = SettlementController()
    _dispatched(controller)

    receipt = controller.begin_action(_control(TEMPORARY), TEMPORARY)

    assert receipt is not None
    assert receipt.status is SettlementStatus.UNVERIFIED


def test_close_unverified_is_safe_without_active_transaction() -> None:
    controller = SettlementController()

    assert controller.close_unverified("run ended") is None


def test_guarded_navigation_requires_its_observed_destination() -> None:
    controller = SettlementController()
    control = _control(DRAWER)
    controller.begin_action(
        control,
        DRAWER,
        expected_destination_phash=SCHEDULE_LIST,
    )

    receipt = controller.record_dispatch(
        control, DRAWER, UNEXPECTED_FORM, failed=False
    )

    assert receipt is not None
    assert receipt.status is SettlementStatus.UNVERIFIED
    assert receipt.expected_destination_phash == SCHEDULE_LIST


def test_guarded_navigation_settles_after_destination_observation() -> None:
    controller = SettlementController()
    control = _control(DRAWER)
    controller.begin_action(
        control,
        DRAWER,
        expected_destination_phash=SCHEDULE_LIST,
    )
    assert (
        controller.record_dispatch(
            control, DRAWER, SCHEDULE_LIST, failed=False
        )
        is None
    )

    receipt = controller.observe(SCHEDULE_LIST, "observation-list")

    assert receipt is not None
    assert receipt.status is SettlementStatus.DURABLE_PROGRESS
