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
            assert 'GeometryAdapterStack' in st.__all__ and 'GeometryAdapterStack' in namespace
            try:
                namespace['GeometryAdapterStack']({})
            except RuntimeError as error:
                assert 'PyTorch is required' in str(error)
            else:
                raise AssertionError('geometry stack did not reject missing PyTorch')
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
            from spiraltorch import fractional_autograd as fractional
            assert fractional.torch is None
            for name in fractional.__all__:
                assert name in st.__all__ and name in namespace, name
                assert namespace[name] is getattr(fractional, name), name
                try:
                    if name.endswith('Adapter'):
                        namespace[name](8)
                    elif name == 'fractional_gl_history_log_gain_autograd':
                        namespace[name](None, None, None, axis=1)
                    elif name == 'fractional_gl_angle_autograd':
                        namespace[name](None)
                    else:
                        namespace[name](None, None, axis=1)
                except RuntimeError as error:
                    assert 'PyTorch is required' in str(error), (name, error)
                else:
                    raise AssertionError(name + ' did not reject missing PyTorch')
            # Native snapshots remain usable without the optional AD client.
            from array import array
            wave = st.WaveGateKernel().forward_buffer(
                array('f', [1.]), array('f', [0.]), array('f', [0.]), 1, 1)
            assert wave.output_buffer() == array('f', [0.]).tobytes()
            assert wave.vjp_buffer(array('f', [1.]))[1] == array('f', [1.]).tobytes()
            chart = st.FractionalGlAngleChart(0.)
            assert chart.alpha == 1. and chart.alpha_derivative == 2.
            assert chart.vjp(.5) == chart.jvp(.5) == 1.
            buffered = st.FractionalGlKernel().forward_buffer(array('f', [1.0]), [1], 0, 0.5)
            assert buffered.output_buffer() == array('f', [1.0]).tobytes()
            assert buffered.vjp_alpha_buffer(array('f', [2.0])) == 0.0
            full = st.FractionalGlKernel().forward([1.0], [1], 0, 0.5)
            assert full.output == [1.0]
            assert full.vjp_input([2.0]) == [2.0] and full.vjp_alpha([2.0]) == 0.0
            history = st.FractionalGlKernel().forward_history([1.0], [1], 0, 0.5)
            assert history.output == [0.0]
            assert history.vjp_input([2.0]) == [0.0] and history.vjp_alpha([2.0]) == 0.0
            normalized = st.FractionalGlKernel(kernel_len=2).forward_history_l2_buffer(
                array('f', [1., 2.]), [2], 0, .5, 1.5)
            assert normalized.output == [0., -1.5]
            assert normalized.vjp_alpha([1., 1.]) == 0.
            learned = st.FractionalGlKernel(kernel_len=2).forward_history_log_gain_buffer(
                array('f', [1., 2.]), [2], 0, .5, 0.)
            assert isinstance(learned, st.FractionalGlGainLearningBatch)
            assert learned.output == [0., -1.]
            assert learned.vjp_parameters([1., 1.]) == (0., -1.)
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
