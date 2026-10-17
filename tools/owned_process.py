"""Bounded POSIX command ownership for file-logged experiment supervisors.

Normal exits, failures, timeouts, SIGINT and SIGTERM clean up the owned process
group before returning. Workers have no cleanup grace; the one enclosing
supervisor explicitly gives their cooperative parent five seconds to clean up.
Arbitrary nested daemons or processes escaping their group are not supported.
No user-space supervisor can guarantee cleanup after its own SIGKILL or power loss.
"""
import math
import os
import signal
import subprocess
import threading


def _signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _cleanup(process, grace):
    # Finish ownership transfer even if cancellation is pressed a second time.
    handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if grace == 0:
            _signal_group(process, signal.SIGKILL)
            process.wait()
        else:
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
            finally:
                _signal_group(process, signal.SIGKILL)
                process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def run(command, *, timeout, stdout, stderr=subprocess.STDOUT, cwd=None, cleanup_grace=0):
    """File-logged POSIX command; only the enclosing supervisor uses grace=5.

Its directly supervised commands must use the default zero-grace worker
cleanup. Equal deadlines at nested levels are not a supported contract.
"""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        raise ValueError("owned process supervision requires the POSIX main thread")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if stdout == subprocess.PIPE or stderr == subprocess.PIPE:
        raise ValueError("owned process logs must not use undrained PIPEs")
    if type(cleanup_grace) not in (int, float) or cleanup_grace not in (0, 5):
        raise ValueError("cleanup grace must be 0 for workers or 5 for their supervisor")
    spawning, pending = True, None

    def cancel(signum, _frame):
        nonlocal pending
        if spawning:
            pending = pending or signum
        elif signum == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        spawning = False
        if pending is not None:
            cancel(pending, None)
        code = process.wait(timeout=timeout)
        return subprocess.CompletedProcess(command, code)
    finally:
        try:
            if process is not None:
                _cleanup(process, cleanup_grace)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
