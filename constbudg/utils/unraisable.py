"""Filter noisy teardown exceptions that cannot be caught with try/except."""

from __future__ import annotations

import sys
import traceback


def _is_multiprocess_resource_tracker_rlock_noise(unraisable) -> bool:
    """
    multiprocess/resource_tracker.py can raise on shutdown (Python 3.12+):
    AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
    """
    exc = unraisable.exc_value
    if not isinstance(exc, AttributeError):
        return False
    if "_recursion_count" not in str(exc):
        return False
    tb = unraisable.exc_traceback
    if tb is None:
        return False
    for frame in traceback.extract_tb(tb):
        if "resource_tracker" in frame.filename:
            return True
    return False


def install_multiprocess_resource_tracker_noise_filter() -> None:
    """
    Install sys.unraisablehook to silence known multiprocess ResourceTracker
    teardown noise. Other unraisable exceptions are passed to the default hook
    (there is no way to "re-raise" into normal control flow from this hook).
    """
    default = getattr(sys, "__unraisablehook__", None)

    def hook(unraisable) -> None:
        if _is_multiprocess_resource_tracker_rlock_noise(unraisable):
            return
        if default is not None:
            default(unraisable)
        else:
            traceback.print_exception(
                unraisable.exc_type,
                unraisable.exc_value,
                unraisable.exc_traceback,
                file=sys.stderr,
            )

    sys.unraisablehook = hook
