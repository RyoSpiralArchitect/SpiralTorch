"""Reproduce the actual alias omissions, without depending on an installed wheel."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

root = Path('/Users/ryospiralarchitect/\U0001f300SpiralReality\U0001f300/_wt/spiraltorch-resident-nn-integration-v1')
spec = importlib.util.spec_from_file_location('runtime_tests', root / 'tests/test_runtime_imports.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
current = module.TOP_LEVEL_STUB_PATH
name = 'test_wgpu_stub_preserves_facade_and_wildcard_aliases'
with tempfile.TemporaryDirectory() as directory:
    previous = Path(directory) / 'before.pyi'
    previous.write_bytes(subprocess.check_output(['git', 'show',
        '6475973ac3f03566b3dfa025bff0993f9aa431b1:bindings/st-py/spiraltorch/__init__.pyi'], cwd=root))
    module.TOP_LEVEL_STUB_PATH = previous
    before = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([module.RuntimeImportsTest(name)]))
    assert not before.wasSuccessful() and len(before.failures) == 2 and not before.errors
module.TOP_LEVEL_STUB_PATH = current
after = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([module.RuntimeImportsTest(name)]))
assert after.wasSuccessful()
print('Expected two pre-fix alias failures reproduced; fixed stub passed.')
