"""Keep native QKV import isolation effective under unittest discovery too."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = r'''
import importlib.abc
import json
from pathlib import Path
import sys
import unittest

forbidden = {"torch", "numpy", "transformers", "safetensors", "accelerate"}
attempts = []
class RejectForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            attempts.append(fullname)
            raise AssertionError("foreign ML import: " + fullname)

def check():
    loaded = [name for name in sys.modules if name.split(".")[0] in forbidden]
    if attempts or loaded:
        raise AssertionError("foreign ML isolation failed: " + repr((attempts, loaded)))

check()
sys.meta_path.insert(0, RejectForeignML())
package_root = Path(sys.argv[1])
sys.path.insert(0, str(package_root))
import spiraltorch
assert Path(spiraltorch.__file__).resolve() == package_root / "spiraltorch/__init__.py"
check()
count = 0
if len(sys.argv) == 3:
    sys.path.insert(0, sys.argv[2])
    names = ["test_native_causal_lm_qkv", "test_native_qkv_generation_example",
             "test_native_qkv_corpus_example"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    expected = suite.countTestCases()
    result = unittest.TextTestRunner(verbosity=1, failfast=True).run(suite)
    count = result.testsRun
    assert result.wasSuccessful() and not result.skipped and count == expected and count > 0
check()
print(json.dumps({"isolated": True, "tests": count, "attempts": attempts}))
'''


def probe(package_root, test_directory=None):
    args = [sys.executable, "-I", "-B", "-S", "-c", PROBE, str(package_root)]
    if test_directory is not None:
        args.append(str(test_directory))
    return subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)


class NativeQkvImportIsolationTests(unittest.TestCase):
    def test_native_qkv_suite_in_fresh_process_blocks_foreign_ml(self):
        spec = importlib.util.find_spec("spiraltorch")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.origin)
        root = Path(spec.origin).resolve().parent.parent
        result = probe(root, Path(__file__).resolve().parent)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(record["isolated"])
        self.assertGreater(record["tests"], 0)
        self.assertEqual(record["attempts"], [])

    def test_probe_rejects_swallowed_import_and_preloaded_module(self):
        cases = (
            ("pass\n", True),
            ("try:\n import torch\nexcept Exception:\n pass\n", False),
            ("import sys, types\nsys.modules['numpy'] = types.ModuleType('numpy')\n", False),
        )
        for source, accepted in cases:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                package = root / "spiraltorch"
                package.mkdir()
                (package / "__init__.py").write_text(source)
                result = probe(root)
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                if not accepted:
                    self.assertIn("foreign ML isolation failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
