"""Keep public geometry imports available when optional PyTorch is absent."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import spiraltorch as st


class GeometryOptionalDependencyTests(unittest.TestCase):
    def test_public_star_import_and_actionable_errors_without_torch(self):
        code = textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            sys.modules['torch'] = None
            import spiraltorch as st
            namespace = {}
            exec('from spiraltorch import *', namespace)
            from spiraltorch import geometry_autograd as geometry
            assert geometry.torch is None
            for name in geometry.__all__:
                assert name in st.__all__ and name in namespace, name
                assert namespace[name] is getattr(geometry, name), name
                try:
                    if name.endswith('Adapter'):
                        namespace[name](8)
                    elif name in ('wave_gate_autograd', 'elliptic_gated_causal_autograd', 'elliptic_anchored_autograd'):
                        namespace[name](None, None, None)
                    else:
                        namespace[name](None, None)
                except RuntimeError as error:
                    assert 'PyTorch is required' in str(error), (name, error)
                else:
                    raise AssertionError(name + ' did not reject missing PyTorch')
        """)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(Path(st.__file__).resolve().parent.parent),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
