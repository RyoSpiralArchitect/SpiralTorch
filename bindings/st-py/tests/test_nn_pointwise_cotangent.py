"""The same Rust pointwise seed/VJP contract, through both public graph owners."""
import ast
import json
import os
from pathlib import Path
import unittest

import spiraltorch as st


def plan():
    model = st.nn.Sequential()
    model.add(st.nn.Scaler.from_gain("gain", st.Tensor(1, 1, [2.])))
    model.add(st.nn.Relu())
    return model.inference_plan([2, 2, 1]).fuse_pointwise()


def inputs(*tensors):
    result = st.wgpu.WgpuPointwiseInputs()
    for tensor in tensors:
        result.add(tensor)
    return result


def read(tensor):
    return tensor.snapshot().read_values()


class Surface(unittest.TestCase):
    def test_graph_owners_declare_the_shared_pointwise_types(self):
        stub = Path(__file__).resolve().parents[1] / "spiraltorch/__init__.pyi"
        classes = {n.name: n for n in ast.parse(stub.read_text()).body if isinstance(n, ast.ClassDef)}
        for name in ("_NnResidentGraphLearner", "_NnResidentGraphAutograd"):
            method = next(n for n in classes[name].body
                          if isinstance(n, ast.FunctionDef) and n.name == "backward_pointwise")
            self.assertEqual([ast.unparse(a.annotation) for a in method.args.args[1:]],
                             ["_NnGraphForward", "WgpuPointwisePlan", "WgpuPointwiseInputs"])
            self.assertEqual(ast.unparse(method.returns), "_NnGraphGradients")


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_materialized_equivalence_ownership_and_recovery(self):
        for learner in (False, True):
            owner = (plan().compile_graph_learner_wgpu(gradient_policy="exact") if learner
                     else plan().compile_graph_autograd_wgpu())
            self.assertNotEqual(owner.adapter_info()["device_type"], "Cpu")
            owner.upload_values([1.] * 4)
            device = owner.tensor_device()
            source = device.upload([2, 2, 1], [1., 2., 3., 4.]).permute([1, 0, 2])
            scalar = device.upload([], [.5])
            bound = inputs(source, scalar)
            program = bound.compile([("multiply", 1), ("add", 0)])
            f = owner.forward()
            before = owner.submitted_backwards
            with self.assertRaises(ValueError):
                owner.backward_pointwise(f, program, inputs(source))
            self.assertEqual(owner.submitted_backwards, before)
            materialized = owner.backward(f, program.run(bound))
            direct = owner.backward_pointwise(f, program, bound)
            held = direct.input_gradient_tensor()
            self.assertEqual(read(held), [3., 9., 6., 12.])
            self.assertEqual(read(held), read(materialized.input_gradient_tensor()))
            self.assertEqual(read(direct.parameter_gradient_tensor(0)), [15.])
            self.assertEqual(read(materialized.parameter_gradient_tensor(0)), [15.])
            maximum = device.upload([], [float.fromhex("0x1.fffffep+127")])
            negative = device.upload([], [-2.])
            bad_inputs = inputs(source, maximum, negative)
            bad_plan = bad_inputs.compile([("multiply", 1), ("multiply", 2), ("relu", None)])
            bad = owner.backward_pointwise(f, bad_plan, bad_inputs)
            good = owner.backward_pointwise(f, program, bound)
            self.assertEqual(read(good.parameter_gradient_tensor(0)), [15.])
            if learner:
                batch = st.nn.GraphGradientBatch()
                batch.add(good, 1.)
                batch.add(bad, 0.)
                owner.sgd_weighted(batch, .1)
                with self.assertRaises(ValueError):
                    owner.update_snapshot().read()
                saved = json.loads(owner.parameter_snapshot().read_plan().to_json())
                self.assertEqual(saved["parameters"][0]["values"], [2.])
            else:
                owner.forward()
            with self.assertRaises(ValueError):
                owner.backward_pointwise(f, program, bound)
            del owner, bound, program, source, scalar, bad_inputs, bad_plan, device
            self.assertEqual(read(held), [3., 9., 6., 12.])
            for tensor in (bad.input_gradient_tensor(), bad.parameter_gradient_tensor(0)):
                with self.assertRaises(ValueError):
                    read(tensor)


if __name__ == "__main__":
    unittest.main()
