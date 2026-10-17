import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("owned_process", Path(__file__).with_name("owned_process.py"))
owned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owned)


@unittest.skipUnless(os.name == "posix", "POSIX process groups")
class OwnershipTests(unittest.TestCase):
    def command(self, root, outcome):
        child = ("from pathlib import Path; import time; "
                 f"Path({str(root / 'ready')!r}).touch(); time.sleep(1); "
                 f"Path({str(root / 'leak')!r}).touch()")
        parent = ("import subprocess, sys, time; from pathlib import Path; "
                  f"subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', {child!r}]); "
                  f"ready=Path({str(root / 'ready')!r})\n"
                  "deadline=time.monotonic()+5\n"
                  "while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                  "assert ready.exists()\n" + outcome)
        return [sys.executable, "-I", "-S", "-B", "-c", parent]

    def test_success_and_failure_clean_descendants(self):
        for code in (0, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = owned.run(self.command(root, f"sys.exit({code})"),
                                   timeout=10, stdout=subprocess.DEVNULL)
                self.assertEqual(result.returncode, code)
                time.sleep(1.1)
                self.assertTrue((root / "ready").exists())
                self.assertFalse((root / "leak").exists())

    def test_timeout_cleans_descendants_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                owned.run(self.command(root, "time.sleep(10)"), timeout=.4, stdout=subprocess.DEVNULL)
            time.sleep(1.1)
            self.assertTrue((root / "ready").exists())
            self.assertFalse((root / "leak").exists())
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_keyboard_interrupt_cleans_descendants(self):
        original = subprocess.Popen.wait
        calls = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def interrupted(process, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    deadline = time.monotonic() + 5
                    while not (root / "ready").exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue((root / "ready").exists())
                    raise KeyboardInterrupt
                return original(process, *args, **kwargs)

            with patch.object(subprocess.Popen, "wait", interrupted), self.assertRaises(KeyboardInterrupt):
                owned.run(self.command(root, "time.sleep(10)"), timeout=10, stdout=subprocess.DEVNULL)
            time.sleep(1.1)
            self.assertFalse((root / "leak").exists())

    def test_nested_supervisor_forwards_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = self.command(root, "time.sleep(10)")
            nested = ("import importlib.util, subprocess; "
                      f"s=importlib.util.spec_from_file_location('owned', {owned.__file__!r}); "
                      "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                      f"m.run({worker!r},timeout=10,stdout=subprocess.DEVNULL)")
            with self.assertRaises(subprocess.TimeoutExpired):
                owned.run([sys.executable, "-I", "-S", "-B", "-c", nested],
                          timeout=.5, stdout=subprocess.DEVNULL, cleanup_grace=5)
            time.sleep(1.1)
            self.assertTrue((root / "ready").exists())
            self.assertFalse((root / "leak").exists())

    def test_nested_supervisor_kills_term_resistant_worker_before_escalation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = ("import signal,time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      f"Path({str(root / 'ready')!r}).touch(); time.sleep(2); "
                      f"Path({str(root / 'leak')!r}).touch(); time.sleep(10)")
            command = [sys.executable, "-I", "-S", "-B", "-c", worker]
            nested = ("import importlib.util,subprocess; "
                      f"s=importlib.util.spec_from_file_location('owned',{owned.__file__!r}); "
                      "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                      f"m.run({command!r},timeout=10,stdout=subprocess.DEVNULL)")
            with self.assertRaises(subprocess.TimeoutExpired):
                owned.run([sys.executable, "-I", "-S", "-B", "-c", nested],
                          timeout=.5, stdout=subprocess.DEVNULL, cleanup_grace=5)
            time.sleep(2.1)
            self.assertTrue((root / "ready").exists())
            self.assertFalse((root / "leak").exists())

    def test_spawn_registration_defers_sigterm_and_sigint_until_owned(self):
        original = subprocess.Popen
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(sig=sig), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                previous = signal.getsignal(sig)

                def interrupted(*args, **kwargs):
                    process = original(*args, **kwargs)
                    deadline = time.monotonic() + 5
                    while not (root / "ready").exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    signal.raise_signal(sig)
                    return process

                exception = SystemExit if sig == signal.SIGTERM else KeyboardInterrupt
                with patch.object(owned.subprocess, "Popen", interrupted), self.assertRaises(exception):
                    owned.run(self.command(root, "time.sleep(10)"), timeout=10, stdout=subprocess.DEVNULL)
                time.sleep(1.1)
                self.assertTrue((root / "ready").exists())
                self.assertFalse((root / "leak").exists())
                self.assertIs(signal.getsignal(sig), previous)

    def test_invalid_timeout_or_pipes_reject_before_spawn(self):
        with patch.object(owned.subprocess, "Popen") as spawn:
            for timeout in (0, -1, float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    owned.run(["unused"], timeout=timeout, stdout=subprocess.DEVNULL)
            with self.assertRaises(ValueError):
                owned.run(["unused"], timeout=1, stdout=subprocess.PIPE)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
