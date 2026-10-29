"""Walking the path an earlier run saved, before the planner takes over.

Its own module because the loop should not have to know what a trace is.
The loop asks how many steps it no longer has to plan, and starts there.
"""

import logging
from yewo.agent.actuation.actions import PlannerTapAction, TapAction, TypeAction
from yewo.agent.memory import replay_match, replay_walk
from yewo.agent.memory.history import ActionEntry, ActionResult
from yewo.agent.memory.replay_projection import project_actionable_elements
from yewo.agent.run.record import StepSource
from yewo.agent.run.outcomes import (
    GOAL_REACHED,
    BeforeActionHookFn,
    DeviceRunResult,
    ObservationDirective,
    flush_trail,
)
from yewo.agent.run.postconditions import (
    PostconditionVerification,
    pending_for,
    verify_context,
)
from yewo.agent.run.recovery import prepare_recovery_proposal, record_recovery_action
from yewo.agent.step import observation

logger = logging.getLogger(__name__)


def try_cached_step(
    agent,
    snapshot,
    device_control,
    ctx,
    settle,
    session_dir,
    step: int,
    before_action_hook: BeforeActionHookFn | None = None,
) -> bool:
    """Takes one step from the saved path, or says the planner must.

    Asked once per turn of the loop, on the screen the loop already read.
    A refusal costs this step only. The planner moves the app on, and the
    next turn asks again from wherever that left it. So one changed screen
    no longer abandons the rest of the path.
    """
    replay = agent.durable_replay
    if replay is None or not replay.has_path:
        return False

    if not _arrived_where_expected(agent, replay, snapshot, ctx, step):
        return False

    if replay.path_completed:
        return False

    expected_transition = (
        replay.trace.transitions[replay.position]
        if replay.position < len(replay.trace.transitions)
        else None
    )
    if expected_transition is not None and expected_transition.starts_segment:
        index = replay.position
        if snapshot.scan_info is None:
            index = None
        elif expected_transition.source.anchors:
            projected_elements = project_actionable_elements(
                snapshot.gui_elements,
                snapshot.scan_info,
                replay.trace.identity.grounding,
            )
            refusal = replay_match.screen_agrees(
                expected_transition.source,
                projected_elements,
                snapshot.scan_info,
                replay.names,
                replay.key,
            )
            if refusal is not None:
                index = None
    elif expected_transition is not None and not expected_transition.source.anchors:
        # Detector-free paths have only a semantic source description. The
        # grounder verifies that description before the action is resolved.
        index = replay.position
    else:
        index = replay_walk.select_transition(
            replay.trace, replay.position, snapshot, replay.names, replay.key
        )
    if index is None:
        logger.debug("No saved step starts on this screen. Planning it.")
        return False

    transition = replay.trace.transitions[index]
    needs_semantic_grounding = bool(
        transition.source.description
        and not (
            replay.is_same_build
            and transition.source.phash == snapshot.current_phash
            and (
                transition.action.kind not in replay_walk.TARGETED_KINDS
                or (not transition.action.sensitive and transition.action.target_bounds is not None)
            )
        )
    )
    semantic_grounder = agent.grounding_models.replay() if needs_semantic_grounding else None
    model_image = (
        agent.images.redact_model_image(snapshot.raw_img_bytes, snapshot.gui_elements)
        if needs_semantic_grounding
        else None
    )

    attempt = replay_walk.take_cached_step(
        transition=transition,
        screen=snapshot,
        device_control=device_control,
        settle=lambda: settle.settle(device_control, agent.settle_time),
        values=replay.values,
        names=replay.names,
        step_log=ctx.steps,
        step_number=step + 1,
        quick_phash=lambda: settle.quick_phash(device_control),
        scrubber=agent.scrubber,
        key=replay.key,
        semantic_grounder=semantic_grounder,
        same_build=replay.is_same_build,
        model_image=model_image,
        before_action=lambda action: _allow_cached_action(
            before_action_hook, action, step + 1, ctx, snapshot
        ),
        grounding=replay.trace.identity.grounding,
        instruction_controller=ctx.instruction_controller,
        context_id=(
            ctx.execution_context.context_id
            if ctx.execution_context is not None
            else None
        ),
    )
    if attempt.stopped:
        return False
    if attempt.step is None:
        if attempt.resolved_action is not None:
            nudge = ctx.loop_detector.record_stale(
                attempt.resolved_action,
                snapshot.current_phash,
                observation.screen_size(snapshot),
            )
            if nudge is not None:
                ctx.pending_loop_nudge = "\n".join(
                    part for part in (ctx.pending_loop_nudge, nudge) if part
                )
        logger.info("Saved step %d refused: %s. Planning it.", index + 1, attempt.refusal)
        if transition.source.description and not attempt.source_mismatch:
            # This saved step belongs to the screen that was just refused.
            # Planning replaces only this step. The next screen can still
            # rejoin the following saved transition.
            replay.position = index + 1
        return False

    _record_cached_step(
        agent, replay, transition, attempt.step, ctx, snapshot, session_dir, step
    )
    replay.position = index + 1
    replay.pending_destination = replay.trace.transitions[index].destination
    if attempt.refusal is not None:
        # It ran and the device refused it. There is no destination to
        # read, and what runs next is a recovery rather than a route.
        replay.pending_destination = None
        replay.close_learning_segment(snapshot, transition.source.description)
    return True


def _allow_cached_action(
    before_action_hook: BeforeActionHookFn | None,
    action,
    step: int,
    ctx,
    snapshot,
) -> bool:
    if before_action_hook is not None:
        directive = before_action_hook(action, step)
        if directive is ObservationDirective.STOP:
            ctx.observation_blocked = True
            return False
    ctx.pending_loop_nudge = ctx.loop_detector.record_action(
        action,
        snapshot.current_phash,
        observation.screen_size(snapshot),
        snapshot.observation_id,
    )
    return not (
        ctx.loop_detector.action_blocked
        or ctx.loop_detector.action_repetition_terminated
    )


def verify_pending_arrival(agent, snapshot, ctx, step: int) -> bool:
    """Verify the previous cached action before this screen is audited."""
    replay = agent.durable_replay
    if replay is None or not replay.has_path:
        return True
    return _arrived_where_expected(agent, replay, snapshot, ctx, step)


def _arrived_where_expected(agent, replay, snapshot, ctx, step: int) -> bool:
    """Whether the last cached step reached the screen it used to reach.

    Asked of this step's own reading. A cached step that landed elsewhere
    must not read as a step that worked. The path must not be walked on
    from a screen it never expected.
    """
    expected, replay.pending_destination = replay.pending_destination, None
    if expected is None:
        return True
    is_final_destination = replay.position >= len(replay.trace.transitions)
    if is_final_destination and not replay.is_same_build and expected.description:
        model_image = agent.images.redact_model_image(snapshot.raw_img_bytes, snapshot.gui_elements)
        if not agent.grounding_models.replay().matches_screen(model_image, expected.description):
            refusal = "the saved final screen description does not match"
            logger.info("A saved step did not arrive where it used to: %s.", refusal)
            _mark_last_step_failed(ctx, refusal)
            return False
    refusal = replay_walk.confirm_arrival(
        expected,
        snapshot,
        replay.names,
        replay.key,
        same_build=replay.is_same_build,
        grounding=replay.trace.identity.grounding,
    )
    if refusal is None:
        if replay.position >= len(replay.trace.transitions):
            replay.path_completed = True
        return True
    logger.info("A saved step did not arrive where it used to: %s.", refusal)
    _mark_last_step_failed(ctx, refusal)
    return False


def _mark_last_step_failed(ctx, refusal: str) -> None:
    """Writes the arrival refusal onto the step it belongs to.

    The step that acted is already recorded, because the screen it reached
    is only read on the turn after it. A report that left it successful
    would say a cached step worked when the run had to plan around it.
    """
    for record in reversed(ctx.steps.records):
        if record.source is StepSource.CACHE and record.outcome is not None:
            record.outcome.success = False
            record.outcome.observation = refusal
            return


def _record_cached_step(
    agent, replay, transition, cached, ctx, snapshot, session_dir, step
) -> None:
    """Count and remember one cached step as a planned one is."""
    ctx.steps_executed += 1
    replay.cached_step_count += 1
    ctx.steps.files(agent.artifacts.save_step(session_dir, step, snapshot))
    ctx.history.append(
        ActionEntry(
            step_num=cached.record.n,
            action_desc=(
                cached.record.action.description if cached.record.action else "A saved step"
            ),
            result=ActionResult(
                success=cached.record.outcome.success if cached.record.outcome else False,
                observation=(cached.record.outcome.observation if cached.record.outcome else ""),
            ),
        )
    )
    ctx.steps.record_settlement_id(ctx.loop_detector.active_settlement_id)
    ctx.loop_detector.record_delivery_outcome(
        cached.action,
        snapshot.current_phash,
        cached.record.phash,
        observation.screen_size(snapshot),
        failed=not cached.is_executed,
    )
    if cached.is_executed:
        if isinstance(cached.action, TypeAction) and cached.entry_receipt is not None:
            specification = (
                transition.action.postconditions[0]
                if transition.action.postconditions
                else None
            )
            ctx.pending_entered_value = pending_for(
                cached.action,
                replay.names,
                replay.values,
                cached.entry_receipt.observed,
                specification,
                replay.verification_policies,
                focus=cached.entry_receipt.focus,
                source_observation_id=cached.record.observation_id,
                source_context_id=cached.record.context_id,
                source_phash=snapshot.current_phash,
            )
        replay.record_step(
            cached.action,
            cached.source,
            cached.arrival_ms,
            target_description=transition.action.described,
            screen_description=transition.source.description,
        )
    ctx.prev_phash = snapshot.current_phash


def reject_last_entered_value(
    agent,
    ctx,
    reason: str,
) -> None:
    """Exclude a failed entry from learning and require planner recovery."""
    _mark_last_action_failed(ctx, reason)
    replay = agent.durable_replay
    if replay is not None:
        replay.reject_last_step()
    ctx.pending_loop_nudge = "\n".join(filter(None, (ctx.pending_loop_nudge, reason)))
    ctx.postcondition_recovery_required = False


def verify_entered_value(
    agent, device_control, snapshot, ctx
) -> PostconditionVerification | None:
    """Verify and reject the previous entry before another action can run."""
    verification = verify_context(
        snapshot,
        ctx,
        read_focused_field=(
            device_control.read_focused_field
            if ctx.pending_entered_value is not None
            else None
        ),
        protect_focused_region=agent.scrubber.protect_typed_bounds,
    )
    if verification is not None and ctx.postcondition_recovery_required:
        reject_last_entered_value(agent, ctx, verification.reason)
    prepare_recovery_proposal(ctx, snapshot, device_control)
    return verification


def _mark_last_action_failed(ctx, refusal: str) -> None:
    for record in reversed(ctx.steps.records):
        if record.action is not None and record.outcome is not None:
            record.outcome.success = False
            record.outcome.observation = refusal
            break
    ctx.history.mark_last_failed(refusal)


def record_planned_step(agent, action, snapshot, ctx) -> None:
    """Keeps a planned step for the path this run may save.

    Only a step the device took. An action can fail after it is sent, and
    the loop catches that rather than raising. So a scenario that recovers
    and passes would otherwise write the failed action into the file.

    The failed action closes one segment. A later successful action starts a
    new segment from a newly observed screen.
    """
    record_recovery_action(ctx, snapshot)
    replay = agent.durable_replay
    if replay is None:
        return
    record = ctx.steps.current
    outcome = record.outcome if record is not None else None
    if outcome is not None and not outcome.success:
        replay.close_learning_segment(snapshot, None)
        return
    target_description = (
        action.target
        if isinstance(action, (TapAction, PlannerTapAction, TypeAction))
        else None
    )
    arrival_ms = round(record.act_ms or 0) if record is not None else 0
    replay.record_step(
        action,
        snapshot,
        arrival_ms=arrival_ms,
        target_description=agent.scrubber.scrub(target_description),
        screen_description=None,
    )


def record_arrival(
    agent,
    device_control,
    perception,
    session_trail,
    ctx,
    observation_hook=None,
) -> bool:
    """Reads the screen the last action produced.

    A step's destination is the next step's source, and the last step has
    no next step. Without one more reading, every scenario drops its last
    action from the path it learns. A scenario of one action then learns
    nothing at all.
    """
    replay = agent.durable_replay
    learning_arrival = bool(
        replay is not None and not replay.path_completed and replay.is_worth_saving
    )
    if not learning_arrival and ctx.pending_entered_value is None:
        return False
    try:
        arrived = observation.capture(perception, session_trail, ctx.prev_phash)
    except Exception as exc:  # noqa: BLE001 - learning must not fail a run
        logger.debug("No final reading, so the last step is not learned: %s", exc)
        return False
    expected_description = None
    if replay is not None:
        expected_description = (
            replay.pending_destination.description
            if replay.pending_destination is not None
            else None
        )
        if replay.pending_destination is not None:
            _arrived_where_expected(agent, replay, arrived, ctx, ctx.steps_executed)
    verify_entered_value(agent, device_control, arrived, ctx)
    if observation_hook is not None:
        observation_hook(arrived, ctx.steps_executed + 1)
    if learning_arrival and replay is not None:
        replay.record_arrival(arrived, expected_description)
        return replay.path_completed
    return False


def finish_cached_path(agent, snapshot, ctx, session_trail) -> DeviceRunResult | None:
    """Ends a completely matched path without asking a planner to say done."""
    replay = agent.durable_replay
    if replay is None or not replay.path_completed:
        return None

    current = ctx.steps.current
    if current is not None and current.action is None and current.decision is None:
        files = dict(current.screenshots)
        ctx.steps.records.pop()
        if ctx.steps.current is not None:
            ctx.steps.current.screenshots.update(files)

    replay.record_arrival(snapshot)
    return cached_path_result(agent, ctx, session_trail, snapshot.current_phash)


def cached_path_result(agent, ctx, session_trail, phash: str | None) -> DeviceRunResult | None:
    """Returns the result earned by a fully verified saved path."""
    replay = agent.durable_replay
    if replay is None or not replay.path_completed:
        return None

    flush_trail(session_trail, phash)
    logger.info("Saved path completed the scenario without a planner call.")
    return DeviceRunResult(
        status=GOAL_REACHED,
        steps_executed=ctx.steps_executed,
        failure_reason=None,
        history=ctx.history,
        capability_receipts=ctx.capability_receipts,
        steps=list(ctx.steps.records),
        postcondition_verifications=list(ctx.postcondition_verifications),
    )
