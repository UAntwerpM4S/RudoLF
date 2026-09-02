from contextlib import contextmanager
from typing import Callable


@contextmanager
def fh_sim_cleanup(get_env: Callable[[], object], dispose: bool = True):
    """
    Ensure the FH Simulator bridge on an environment is disposed on exit.

    Wraps a block of code (training or visualization) so that
    `env.fh_sim.dispose()` is always called when the block finishes -
    whether it completes normally, breaks early via a stop condition,
    or exits via any exception (including `KeyboardInterrupt`).

    `env.fh_sim` is looked up only on exit, not on entry, so it reflects
    whichever environment/bridge is active by the time the block finishes
    (relevant for pipeline stages that may swap the active environment
    mid-flow).

    `FhSimBridge.dispose()` and `SimSyncBridge.dispose()` are both
    idempotent, safe to call on an environment with no bridge (`fh_sim`
    is `None` for pure-Python dynamics models), and already swallow their
    own internal exceptions - so this context manager does not add
    another layer of exception handling around `dispose()` itself. If a
    future duck-typed bridge's `dispose()` raises, that exception is left
    to propagate rather than being silently caught here.

    Args:
        get_env: Zero-arg callable returning the environment (or
            env-like object) to check for an `fh_sim` attribute.
        dispose: Whether to actually dispose `fh_sim` on exit. Defaults to
            True. Pass False when the caller knows another `fh_sim_cleanup`
            block immediately downstream (on the same environment) will
            perform the real, final dispose - e.g. a pipeline stage that
            trains and then immediately visualizes the same environment.
            Skipping the intermediate dispose avoids tearing down and then
            immediately reinitializing the native FH math model. The
            environment is still not touched at all when False - this is a
            pure no-op on exit, not a deferred/queued dispose.

    Yields:
        None.
    """

    try:
        yield
    finally:
        if dispose:
            env = get_env()
            fh_sim = getattr(env, "fh_sim", None)
            if fh_sim is not None:
                fh_sim.dispose()
