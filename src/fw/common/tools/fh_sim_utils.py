from typing import Callable
from contextlib import contextmanager


@contextmanager
def fh_sim_cleanup(get_env: Callable[[], object], dispose: bool = True):
    """
    Dispose the FH Simulator bridge on an environment when the block exits.

    Runs on normal completion, early break and any exception (including
    KeyboardInterrupt). `env.fh_sim` is looked up on exit, so it reflects the
    bridge active at that moment. Both bridges' `dispose()` are idempotent,
    tolerate `fh_sim is None` (pure-Python dynamics) and swallow their own
    errors, so no extra exception handling is added here.

    Args:
        get_env: Zero-arg callable returning the environment to check for `fh_sim`.
        dispose: If False, exit is a no-op. Use when a downstream
            `fh_sim_cleanup` on the same environment does the final dispose
            (e.g. train then visualize), avoiding a needless FH re-init.

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
