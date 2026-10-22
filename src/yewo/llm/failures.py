"""Which provider failures are worth another attempt, and which are final.

A model call fails for reasons that want opposite answers. A rate limit or a
dropped connection says to wait and ask again. A malformed request or a
rejected key says the next call will fail the same way. Asking again then
spends a device the suite is holding.

Read by type first and by status second. A type is what the client promises,
and a status is what the service said. Neither is read from the message text,
which is written for a person and changes between versions.
"""

import logging
import time
from typing import Callable, Optional, TypeVar

from litellm import exceptions

logger = logging.getLogger(__name__)

Answer = TypeVar("Answer")

ATTEMPTS = 3
"""How many times one call is made before it is given up on.

Three, because the first retry covers a dropped connection and the second
covers a rate limit that needs a pause. More would hold the device longer than
a scenario is worth.
"""

BACKOFF_SECONDS = (2.0, 8.0)
"""How long to wait before the second attempt and the third.

One entry fewer than `ATTEMPTS`, because nothing waits before the first.
"""

_RETRYABLE = (
    exceptions.Timeout,
    exceptions.RateLimitError,
    exceptions.APIConnectionError,
    exceptions.InternalServerError,
    exceptions.ServiceUnavailableError,
    # The decision model has its own SDK, and its transport errors subclass
    # these two. Naming the builtins rather than that SDK keeps it an
    # optional install, and `is_retryable` is only asked at a provider call.
    ConnectionError,
    TimeoutError,
)
"""Failures of the service or the transport, rather than of the request.

Listed by type as well as by status, because a transport failure can arrive
carrying no status at all. A connection that dropped is the clearest case for
asking again, and a status list alone would read it as terminal.

There is no list of the refusals. Every one a client raises carries a status
the list below does not hold. Naming them would decide nothing, and the list
would go stale on its own.
"""

RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
"""Statuses where the same request could be answered later.

408 and 425 are the request's own timing. 409 is a conflict a retry resolves.
429 is a rate limit. The 5xx entries are the service, and 529 is what some
gateways send when a model is overloaded.
"""


STATUS_FIELDS = ("status_code", "status")
"""What each client calls the HTTP status it carries.

LiteLLM says `status_code` and the decision model's SDK says `status`. One
name would read the rate limit of one provider and miss the other's.
"""


def _status_of(exc: BaseException) -> Optional[int]:
    """The HTTP status an exception carries, or None where it carries none.

    Read dynamically because the attribute is the client's, not ours, and it
    is absent on a plain transport error. The value is narrowed to an int
    here, so nothing below this reads an unknown object.
    """
    for name in STATUS_FIELDS:
        status = getattr(exc, name, None)
        if isinstance(status, int):
            return status
    return None


def is_retryable(exc: BaseException) -> bool:
    """Whether another attempt at the same call could answer differently.

    Terminal unless something says otherwise. A failure nothing here
    recognises is not retried: a retry costs the device the suite is holding,
    and a request that was wrong stays wrong.
    """
    if isinstance(exc, _RETRYABLE):
        return True
    return _status_of(exc) in RETRYABLE_STATUSES


def reportable(error: BaseException) -> str:
    """One failure as a log line or a report may carry it.

    A model service quotes the request back in its message. The request is
    the prompt: the screen, the journey, and whatever the app had on it. So a
    failure the client raised is reduced to its type and its status.

    A failure Yewo raised keeps its message, because Yewo wrote it for a
    reader. `ProviderUnavailable` already says what failed and how many
    attempts it took, and reducing it would throw that away.
    """
    from yewo.errors import Cause, YewoError, cause_of

    if cause_of(error) is Cause.PROVIDER and not isinstance(error, YewoError):
        return describe(error)
    return str(error) or type(error).__name__


def call_with_retry(
    operation: Callable[[], Answer], task_label: str, wait=time.sleep
) -> Answer:
    """One provider call, made again while the service could still answer it.

    The same policy as the completion client, over a different vendor. A
    decision model is reached through its own SDK, and a rate limit there
    ends a scenario exactly as one from a completion would.

    Raises:
        ProviderUnavailable: The attempts are spent.
        BaseException: Whatever the service raised, where asking again could
            not have changed it.
    """
    from yewo.errors import ProviderUnavailable

    last: BaseException | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - provider boundary
            if not is_retryable(exc):
                raise
            last = exc
            if attempt == ATTEMPTS:
                break
            pause = BACKOFF_SECONDS[attempt - 1]
            logger.warning(
                "%s did not answer (%s). Attempt %d of %d, waiting %.0fs.",
                task_label,
                describe(exc),
                attempt,
                ATTEMPTS,
                pause,
            )
            wait(pause)

    raise ProviderUnavailable(
        f"{task_label} did not answer after {ATTEMPTS} attempts: "
        f"{describe(last) if last else 'no attempt was made'}"
    ) from last


def describe(exc: BaseException) -> str:
    """One provider failure, as a reader needs it.

    The type and the status, and not the message. A provider message can
    quote the prompt back, and this reaches a report.
    """
    status = _status_of(exc)
    exception_name = type(exc).__name__
    return (
        f"{exception_name} (HTTP {status})"
        if status is not None
        else exception_name
    )
