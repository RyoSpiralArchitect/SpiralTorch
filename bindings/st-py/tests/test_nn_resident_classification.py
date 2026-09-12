"""Real class-last Loss -> VJP -> update via the public Python API."""
import gc
import math
import os
import unittest
import spiraltorch as st


def read(tensor):
    return tensor.snapshot().read_values()


class Surface(unittest.TestCase):
    def test_classification_surface_preserves_host_and_cpu_only_boundaries(self):
        from spiraltorch.nn import CrossEntropyWithLogits
        objective = CrossEntropyWithLogits(label_smoothing=.1)
        host, labels = st.Tensor(1, 2, [0., 0.]), st.Tensor(1, 1, [0.])
        gpu = st.wgpu_kernel_reports_available()
        with self.assertRaises(TypeError if gpu else NotImplementedError):
            objective.evaluate_resident(host, labels)
        if not gpu:
            self.assertAlmostEqual(objective.forward(host, labels).tolist()[0][0], math.log(2), places=6)
        for kwargs in [dict(reduction="invalid"), dict(label_smoothing=float("nan")), dict(label_smoothing=1.1)]:
            with self.assertRaises(ValueError): CrossEntropyWithLogits(**kwargs)


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def near(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isfinite(a) and abs(a-b) <= 2e-5+2e-4*abs(b), (a,b))

    def test_nd_views_and_exact_ignore_transport(self):
        device = st.WgpuTensorDevice.create()
        x = device.upload([3, 2, 7], [0.] * 42).permute([1, 0, 2])
        target = device.upload([], [0.]).broadcast_to([2, 3])
        objective = st.nn.CrossEntropyWithLogits(reduction="none", label_smoothing=.2)
        pair = objective.evaluate_resident(x, target)
        self.assertEqual(tuple(pair.loss_tensor().shape), (2, 3, 1))
        self.near(read(pair.loss_tensor()), [math.log(7)]*6)
        with self.assertRaises(TypeError): objective.evaluate_resident(x, st.Tensor(6, 1, [0.]*6))
        with self.assertRaises(ValueError): objective.evaluate_resident(x, device.upload([6, 1], [0.]*6))
        huge = device.upload([1,2], [float.fromhex("0x1.fffffep+127"), -float.fromhex("0x1.fffffep+127")])
        failures = []
        for label, ignore in [(0.5,-100), (-100.,-100), (16777216.,16777217), (2.**63,2**63-1)]:
            loss = st.nn.CrossEntropyWithLogits(ignore_index=ignore)
            failures.append(loss.evaluate_resident(huge, device.upload([1], [label])))
        zero = device.upload([1,2], [0.,0.])
        poisoned = huge.add(huge).mul(zero)
        failures.append(objective.evaluate_resident(poisoned, device.upload([1], [0.])))
        del objective, device, x, target, huge, zero, poisoned
        gc.collect()
        self.near(read(pair.loss_tensor()), [math.log(7)]*6)
        for failure in failures:
            for tensor in (failure.loss_tensor(), failure.prediction_gradient_tensor()):
                with self.assertRaises(ValueError): read(tensor)

    def test_classification_learns_for_all_reductions_and_recovers_atomically(self):
        for policy, scale in [("exact", 1.), ("module_compatible", .25)]:
            for reduction in ["none", "sum", "mean"]:
                model = st.nn.Sequential()
                model.add(st.nn.Scaler.from_gain("logits", st.Tensor(1,2,[0.,0.])))
                baseline = model.inference_plan([2,2,2])
                learner = baseline.compile_graph_learner_wgpu(gradient_policy=policy)
                device = learner.tensor_device()
                x, target = device.upload([2,2,2],[1.]*8), device.upload([2,2],[0.,-100.,0.,0.])
                learner.set_input_tensor(x)
                objective = st.nn.CrossEntropyWithLogits(reduction=reduction,label_smoothing=.1)
                held, expected = [], [0.,0.]
                rate = .1 if reduction=="mean" else .1/3
                for step in range(32):
                    forward = learner.forward()
                    pair = objective.evaluate_resident(forward.prediction_tensor(),target)
                    gradient = learner.backward(forward,pair.prediction_gradient_tensor())
                    learner.sgd(gradient,rate)
                    p = 1/(1+math.exp(expected[1]-expected[0]))
                    loss = -.95*math.log(p)-.05*math.log1p(-p)
                    held.append((pair,learner.update_snapshot(),loss))
                    expected[0] -= .1*scale*(p-.95)
                    expected[1] -= .1*scale*((1-p)-.05)
                before = learner.parameter_snapshot().read_plan().to_json()
                f = learner.forward()
                bad = objective.evaluate_resident(f.prediction_tensor(),device.upload([2,2],[0.,.5,0.,0.]))
                learner.sgd(learner.backward(f,bad.prediction_gradient_tensor()),0.)
                with self.assertRaises(ValueError): learner.update_snapshot().read()
                self.assertEqual(learner.parameter_snapshot().read_plan().to_json(),before)
                f = learner.forward(); pair = objective.evaluate_resident(f.prediction_tensor(),target)
                learner.sgd(learner.backward(f,pair.prediction_gradient_tensor()),0.)
                self.assertEqual(learner.update_snapshot().read(),34)
                self.assertEqual(baseline.apply_parameters_to(model,learner.parameter_snapshot().read_plan()),1)
                self.near(read(model(x)),expected*4)
                final_values = read(pair.loss_tensor())
                final_loss = sum(final_values)/(1 if reduction=="mean" else 3)
                self.assertLess(final_loss,math.log(2))
                del learner, model, device, objective
                gc.collect()
                for i,(pair,receipt,loss) in enumerate(held,1):
                    self.assertEqual(receipt.read(),i)
                    expected_loss = [loss,0.,loss,loss] if reduction=="none" else [loss*(1 if reduction=="mean" else 3)]
                    self.near(read(pair.loss_tensor()),expected_loss)


if __name__ == "__main__": unittest.main()
