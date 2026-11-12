"""A traced failure has to name the reason, not only the exception class."""

from yewo.extensions import ExtensionDiagnosticError
from yewo.models import ModelArtifactError
from yewo.runs.failure_diagnostics import authored_detail


def test_a_yewo_failure_traces_the_sentence_yewo_raised_it_with() -> None:
    # One Yewo error class is raised from many places, so the class name alone cannot say which
    # check refused the run.
    detail = authored_detail(ModelArtifactError("The model request exceeds the artifact policy."))

    assert detail == " said=The model request exceeds the artifact policy."


def test_a_dependency_message_stays_out_of_the_trace() -> None:
    # A message from outside Yewo may quote whatever the failing call was handling.
    assert authored_detail(ValueError("field contents 4242424242424242")) == ""


def test_an_extension_can_mark_its_own_message_safe_for_the_trace() -> None:
    error = ExtensionDiagnosticError("The classifier dependency is unavailable.")
    error.__cause__ = ValueError("dependency handled 4242424242424242")

    detail = authored_detail(error)

    assert detail == " said=The classifier dependency is unavailable."


def test_a_wrapped_cause_contributes_only_its_yewo_links() -> None:
    cause = ValueError("provider said 4242424242424242")
    error = ModelArtifactError("Could not persist model-call evidence.")
    error.__cause__ = cause

    detail = authored_detail(error)

    assert detail == " said=Could not persist model-call evidence."
