import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_byte_corpus_segments.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CompatibilityTests(unittest.TestCase):
    def test_run_delegates_to_installed_package(self):
        package = ModuleType("spiraltorch")
        package.run_byte_corpus_segments = Mock(return_value=[{"stop_after": 4}])
        with patch.dict("sys.modules", spiraltorch=package):
            self.assertEqual(runner.run("request", "output", [4], resume_after=2),
                             [{"stop_after": 4}])
        package.run_byte_corpus_segments.assert_called_once_with(
            "request", "output", [4], resume_after=2)

    def test_cli_delegates_to_installed_package(self):
        package = ModuleType("spiraltorch.byte_corpus_runs")
        package.main = Mock()
        with patch.dict("sys.modules", {"spiraltorch.byte_corpus_runs": package}):
            runner.main()
        package.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
