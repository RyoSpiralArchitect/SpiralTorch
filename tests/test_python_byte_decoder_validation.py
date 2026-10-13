"""Regression checks for the Python parity transport and wheel identity gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("byte_parity", ROOT / "tools/validate_python_byte_decoder.py")
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


class ExternalReferences(unittest.TestCase):
    def test_flat_order_and_absent_routes(self):
        vectors = [[1.], [2.], [3.]]
        case = {"biases": [{"z": [0.], "pair": [0.]}, {"z": None, "pair": [0.]},
                           {"z": None, "pair": None}], "bias_gradients": vectors}
        self.assertEqual(PARITY.external_gradient_references(case), [
            {"z": vectors[0], "pair": vectors[1]}, {"z": None, "pair": vectors[2]},
            {"z": None, "pair": None}])
        self.assertEqual(case["bias_gradients"], vectors)

    def test_missing_and_extra_routes_fail(self):
        for references in ([], [[1.], [2.]]):
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "external bias derivative"):
                PARITY.external_gradient_references({
                    "biases": [{"z": [0.], "pair": None}], "bias_gradients": references})


class WheelIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        package = self.root / "installed/spiraltorch"
        package.mkdir(parents=True)
        self.facade = package / "__init__.py"
        self.native = package / "spiraltorch.so"
        self.members = {"spiraltorch/__init__.py": b"# facade\n",
                        "spiraltorch/spiraltorch.so": b"native-test-fixture",
                        "spiraltorch-0.4.30.dist-info/METADATA": b"Name: spiraltorch\nVersion: 0.4.30\n"}
        self.facade.write_bytes(self.members["spiraltorch/__init__.py"])
        self.native.write_bytes(self.members["spiraltorch/spiraltorch.so"])
        self.wheel = self.root / "candidate.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for name, data in self.members.items():
                archive.writestr(name, data)
        distribution = SimpleNamespace(version="0.4.30", locate_file=lambda p: package.parent / p)
        for patch in (
            mock.patch.object(PARITY.importlib.metadata, "distribution", return_value=distribution),
            mock.patch.object(PARITY.importlib, "import_module", return_value=SimpleNamespace(__file__=str(self.native))),
            mock.patch.object(PARITY.importlib.machinery, "EXTENSION_SUFFIXES", [".so"]),
            mock.patch.object(PARITY.st, "__file__", str(self.facade)),
            mock.patch.object(PARITY.st, "__version__", "0.4.30"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_exact_candidate_identity_is_recorded_without_local_paths(self):
        report = PARITY.installed_identity(self.wheel)
        self.assertEqual(report["version"], "0.4.30")
        self.assertEqual(report["wheel_sha256"], hashlib.sha256(self.wheel.read_bytes()).hexdigest())
        self.assertEqual(report["installed_files"], {
            n: hashlib.sha256(b).hexdigest() for n, b in self.members.items() if n.startswith("spiraltorch/")})
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_native_mutation_is_rejected(self):
        self.native.write_bytes(b"different extension")
        with self.assertRaisesRegex(ValueError, "differs from candidate"):
            PARITY.installed_identity(self.wheel)

    def test_checkout_shadowing_is_rejected_even_with_same_bytes(self):
        shadow = self.root / "checkout/__init__.py"
        shadow.parent.mkdir()
        shadow.write_bytes(self.facade.read_bytes())
        with mock.patch.object(PARITY.st, "__file__", str(shadow)), \
                self.assertRaisesRegex(ValueError, "shadowing"):
            PARITY.installed_identity(self.wheel)

    def test_version_mismatch_is_rejected(self):
        with mock.patch.object(PARITY.st, "__version__", "0.4.29"), \
                self.assertRaisesRegex(ValueError, "version mismatch"):
            PARITY.installed_identity(self.wheel)


if __name__ == "__main__":
    unittest.main()
