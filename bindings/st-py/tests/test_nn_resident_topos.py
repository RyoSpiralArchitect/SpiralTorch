"""Thin Python construction, Rust graph semantics, and opt-in real-GPU parity."""
import json
import math
import os
from pathlib import Path
import sys
import unittest

import spiraltorch as st


def model(porosity=0.3, **options):
    kernel = st.ToposResonatorKernel(coupling=0.2, iterations=5, porosity=porosity, **options)
    module = st.nn.Sequential()
    module.add_topos_resonator("topos", st.Tensor(1, 3, [.8, -.4, 1.1]), kernel)
    return module, kernel


class Surface(unittest.TestCase):
    def test_shared_gate_host_vjp_and_versioned_plan(self):
        module, kernel = model(max_values=12)
        values = [-2., .3, 1., -.5, .8, 2.]
        gate = json.loads(module.inference_plan([2, 3]).to_json())["parameters"][0]["values"]
        expected = kernel.forward(values, gate * 2, 2, 3)
        observed = [v for row in module(st.Tensor(2, 3, values)).tolist() for v in row]
        self.assertEqual(observed, expected)
        seed = [.25] * 6
        dx, _ = kernel.backward(values, gate * 2, seed, 2, 3)
        actual_dx = module.backward(st.Tensor(2, 3, values), st.Tensor(2, 3, seed))
        self.assertEqual([v for row in actual_dx.tolist() for v in row], dx)
        plan = module.inference_plan([2, 2, 3])
        payload = json.loads(plan.to_json())
        self.assertEqual(payload["schema"], "spiraltorch.nn.inference_plan.v5")
        self.assertEqual(payload["stages"][0]["kind"], "topos_resonator")
        self.assertEqual(payload["parameters"][0]["role"], "gate")
        self.assertEqual(payload["parameters"][0]["shape"], [3])
        self.assertEqual(st.nn.InferencePlan.from_json(plan.to_json()).to_json(), plan.to_json())
        for version in (2, 3, 4):
            payload["schema"] = f"spiraltorch.nn.inference_plan.v{version}"
            with self.assertRaises(ValueError):
                st.nn.InferencePlan.from_json(json.dumps(payload))
        with self.assertRaises(ValueError):
            module.inference_plan([5, 3])

    def test_failed_add_leaves_existing_module_unchanged(self):
        module, kernel = model()
        before = module.inference_plan([2, 3]).to_json()
        for gate in (st.Tensor(2, 3, [1.] * 6), st.Tensor(1, 3, [1., math.nan, 1.])):
            with self.assertRaises((ValueError, RuntimeError)):
                module.add_topos_resonator("bad", gate, kernel)
            self.assertEqual(module.inference_plan([2, 3]).to_json(), before)
        with self.assertRaises(TypeError):
            module.add_topos_resonator("bad", [1., 1., 1.], kernel)
        with self.assertRaises(TypeError):
            module.add_topos_resonator("bad", st.Tensor(1, 3, [1.] * 3), None)
        self.assertEqual(module.inference_plan([2, 3]).to_json(), before)


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def test_shared_gate_training_matches_independent_torch_and_handoffs(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from benchmark_topos_learning import torch_reference

        maximum = dict(output=0., dx=0., dg=0., gate=0., loss=0.)

        def compare(name, actual, expected):
            actual = torch.tensor(actual, dtype=torch.float32, device="cpu").reshape(expected.shape)
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5)
            maximum[name] = max(maximum[name], float((actual - expected).abs().max()))

        for porosity in (0., .3):
            for policy in ("exact", "module_compatible"):
                module, kernel = model(porosity, max_values=12)
                base = module.inference_plan([2, 2, 3])
                graph = base.compile_graph_training_wgpu(gradient_policy=policy)
                self.assertNotEqual(graph.adapter_info()["device_type"], "Cpu")
                initial = json.loads(base.to_json())["parameters"][0]["values"]
                gate = torch.tensor(initial, dtype=torch.float32, device="cpu").requires_grad_()
                config = json.loads(kernel.configuration_json())
                for step in range(25):
                    values = [((i * 17 + step * 7) % 31) / 7. - 2. for i in range(12)]
                    x = torch.tensor(values, dtype=torch.float32, device="cpu").reshape(2, 2, 3).requires_grad_()
                    target = (x.detach() * .13).clone()
                    expected = torch_reference(x, gate, config)
                    loss = (expected - target).square().mean()
                    dx, dg = torch.autograd.grad(loss, (x, gate))
                    graph.upload_batch_values(x.detach().flatten().tolist(), target.flatten().tolist())
                    graph.step(.03)
                    state = graph.state_snapshot().read_state()
                    self.assertEqual(state.parameter_role(0), "gate")
                    compare("output", state.prediction_values(), expected.detach())
                    compare("dx", state.input_gradient_values(), dx)
                    compare("dg", state.parameter_gradient_values(0), dg)
                    compare("dg", state.effective_gradient_values(0), dg)
                    compare("loss", [state.loss], loss.reshape(1).detach())
                    gate = (gate.detach() - .03 * dg).requires_grad_()
                    compare("gate", state.parameter_values(0), gate.detach())
                updated = st.nn.InferencePlan.from_json(state.to_plan().to_json())
                self.assertEqual(base.apply_parameters_to(module, updated), 1)
                host_x = st.Tensor(4, 3, x.detach().flatten().tolist())
                expected = torch_reference(x, gate, config).detach()
                compare("output", [v for row in module(host_x).tolist() for v in row], expected)
                resident = st.WgpuTensorDevice.create().upload([2, 2, 3], x.detach().flatten().tolist())
                compare("output", module(resident).snapshot().read_values(), expected)
        print(json.dumps(dict(scope="100 synthetic graph updates vs independent CPU Torch, not speed/quality", max_abs_error=maximum)))

    def test_resident_gate_handoff_rebuilds_without_overwriting_old_output(self):
        module, _ = model()
        base = module.inference_plan([2, 3])
        device = st.WgpuTensorDevice.create()
        x = device.upload([3, 2], [-2., -.5, .3, .8, 1., 2.]).permute([1, 0])
        held = module(x)
        before = held.snapshot().read_values()
        self.assertEqual(module(x).snapshot().read_values(), before)
        self.assertEqual(module.resident_cache_info()["compilations"], 1)
        payload = json.loads(base.to_json())
        payload["parameters"][0]["values"] = [.2, .3, .4]
        updated = st.nn.InferencePlan.from_json(json.dumps(payload))
        self.assertEqual(base.apply_parameters_to(module, updated), 1)
        after = module(x).snapshot().read_values()
        self.assertNotEqual(before, after)
        self.assertEqual(module.resident_cache_info()["compilations"], 2)
        module.clear_resident_cache()
        del module
        self.assertEqual(held.snapshot().read_values(), before)


if __name__ == "__main__":
    unittest.main()
