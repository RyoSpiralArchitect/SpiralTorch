"""Public fusion uses the existing Rust operations, guards and learner."""
import gc
import os
import unittest
import spiraltorch as st


class Surface(unittest.TestCase):
    def test_aliases_and_cpu_gate(self):
        for name in ("WgpuPointwiseInputs", "WgpuPointwisePlan"):
            self.assertIs(getattr(st, name), getattr(st.wgpu, name))
            self.assertIn(name, st.__all__)
            self.assertIn(name, st.wgpu.__all__)
        with self.assertRaises(TypeError): st.WgpuPointwisePlan()
        if not st.wgpu_kernel_reports_available():
            with self.assertRaises(NotImplementedError): st.WgpuPointwiseInputs()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def inputs(self, *values):
        result = st.WgpuPointwiseInputs()
        for value in values: result.add(value)
        return result

    def test_views_three_modes_reuse_and_ownership(self):
        d = st.WgpuTensorDevice.create()
        self.assertNotEqual(d.adapter_info()["device_type"], "Cpu")
        x = d.upload([2,3,2], [float(i)-3 for i in range(12)]).permute([1,0,2]).narrow(0,1,2)
        inputs = self.inputs(x, d.upload([2], [2.,3.]))
        plan = inputs.compile([("multiply",1),("add",0),("relu",None)])
        expected = [max(v*([2.,3.][i%2])+v, 0.) for i,v in enumerate(x.snapshot().read_values())]
        outputs = [plan.run(inputs, mode) for mode in ("sequential","batched","fused")]
        replacement = d.upload([2,3,2], [1.]*12).permute([1,0,2]).narrow(0,1,2)
        inputs.set(0, replacement)
        changed = plan.run(inputs)
        del plan, inputs, x, replacement, d
        gc.collect()
        for out in outputs:
            self.assertEqual(out.shape, (2,2,2))
            self.assertEqual(out.snapshot().read_values(), expected)
        self.assertEqual(changed.snapshot().read_values(), [3.,4.]*4)

    def test_atomic_replacement_and_recipe_rejection(self):
        d = st.WgpuTensorDevice.create()
        inputs = self.inputs(d.upload([2], [1.,2.]), d.upload([], [2.]))
        plan = inputs.compile([("multiply",1)])
        for bad in (True,-1,1.5,2**65,"0"):
            with self.assertRaises((TypeError,ValueError,OverflowError)): inputs.set(bad,d.upload([2],[9.,9.]))
        for steps in ([],[("mul",1)],[("relu",1)],[("multiply",None)],[("add",True)],[("add",2)],[("identity",None)],[("add",1)]*257):
            with self.assertRaises((TypeError,ValueError,OverflowError)): inputs.compile(steps)
        with self.assertRaises(ValueError): inputs.set(0,d.upload([1,2],[9.,9.]))
        with self.assertRaises(ValueError): plan.run(inputs,"auto")
        self.assertEqual(plan.run(inputs).snapshot().read_values(), [2.,4.])
        full = self.inputs(*([d.upload([], [1.])]*16))
        with self.assertRaises(ValueError): full.add(d.upload([], [1.]))
        self.assertEqual(len(full),16)

    def test_finite_guards_scalar_empty_and_recovery(self):
        d = st.WgpuTensorDevice.create()
        maxf = float.fromhex("0x1.fffffep+127")
        inputs = self.inputs(d.upload([1],[-maxf]),d.upload([],[2.]))
        plan = inputs.compile([("multiply",1),("relu",None)])
        for mode in ("sequential","batched","fused"):
            bad = plan.run(inputs,mode)
            with self.assertRaises(ValueError): bad.snapshot().read_values()
            empty = self.inputs(bad.broadcast_to([0]))
            with self.assertRaises(ValueError): empty.compile([("identity",None)]).run(empty,mode).snapshot().read_values()
        inputs.set(0,d.upload([1],[3.]))
        self.assertEqual(plan.run(inputs).snapshot().read_values(),[6.])
        for shape, data in (([],[-0.]),([0,3],[])):
            values = self.inputs(d.upload(shape,data))
            self.assertEqual(values.compile([("identity",None)]).run(values).snapshot().read_values(),data)

    def test_compiled_cotangent_updates_resident_learner(self):
        model = st.nn.Sequential()
        model.add(st.nn.Scaler.from_gain("gain",st.Tensor(1,1,[2.])))
        gpu = model.inference_plan([2,1]).compile_graph_learner_wgpu(gradient_policy="exact")
        d = gpu.tensor_device()
        gpu.upload_values([1.,2.])
        f = gpu.forward()
        bindings = self.inputs(f.prediction_tensor(),d.upload([],[0.5]))
        cube = bindings.compile([("multiply",0),("multiply",0),("multiply",1)])
        seed = cube.run(bindings)
        gradients = gpu.backward(f,seed)
        raw = gradients.parameter_gradient_tensor(0).snapshot().read_values()[0]
        initial = f.prediction_tensor().snapshot().read_values()[0]
        self.assertAlmostEqual(raw,8.5*initial**3,places=5)
        gpu.sgd(gradients,0.01)
        self.assertEqual(gpu.update_snapshot().read(),1)
        gpu.upload_values([1.,2.])
        after = gpu.forward().prediction_tensor().snapshot().read_values()
        self.assertAlmostEqual(after[0],initial-0.01*raw,places=5)


if __name__ == "__main__": unittest.main()
