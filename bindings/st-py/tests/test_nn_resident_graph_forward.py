"""Existing NN modules compose with public GPU tensors via the Rust compiler."""
import gc
import os
import unittest
import spiraltorch as st


def model(gain=2.):
    net = st.nn.Sequential()
    net.add(st.nn.Scaler.from_gain("scale", st.Tensor(1,1,[gain])))
    net.add(st.nn.Relu())
    return net


class ForwardSurface(unittest.TestCase):
    def test_exports_and_cpu_boundary(self):
        from spiraltorch.nn import ResidentGraphInference, GraphInferenceSnapshot
        for cls in (ResidentGraphInference, GraphInferenceSnapshot):
            self.assertIs(cls, getattr(st.nn, cls.__name__))
            with self.assertRaises(TypeError): cls()
        if not st.wgpu_kernel_reports_available():
            with self.assertRaisesRegex(NotImplementedError, "wgpu"): model().inference_plan([1]).compile_graph_wgpu()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class ForwardGpu(unittest.TestCase):
    def test_direct_forward_interleaves_with_explicit_steps_without_aliases(self):
        graph = model().inference_plan([2,1]).compile_graph_wgpu()
        device = graph.tensor_device()
        view = device.upload([2,2], [99.,-1.,99.,2.]).narrow(1,1,1)
        first = graph.forward_tensor(view)
        snapshot = graph.snapshot()
        second = graph.forward_tensor(first)
        self.assertEqual((graph.generation, graph.submitted_dispatches), (2,2))
        self.assertEqual(graph.dispatch(), 3)
        self.assertEqual(graph.output_tensor().snapshot().read_values(), [0.,8.])
        for bad in (None, st.Tensor(2,1,[0.,0.]), device.upload([1,2],[0.,0.])):
            with self.assertRaises((TypeError,ValueError)): graph.forward_tensor(bad)
        self.assertEqual((graph.generation, graph.submitted_dispatches), (2,3))
        graph.upload_values([1.,1.]); graph.dispatch()
        self.assertEqual(graph.snapshot().read_values(), [2.,2.])
        graph.set_input_tensor(view); graph.dispatch()
        self.assertEqual(graph.snapshot().read_values(), [0.,4.])
        del graph, device, view
        gc.collect()
        self.assertEqual(snapshot.read_values(), [0.,4.])
        self.assertEqual(first.snapshot().read_values(), [0.,4.])
        self.assertEqual(second.snapshot().read_values(), [0.,8.])

    def test_direct_forward_preserves_failed_guards_after_valid_recovery(self):
        graph = model(float.fromhex("0x1.fffffep+127")).inference_plan([2,1]).compile_graph_wgpu()
        device = graph.tensor_device()
        bad = graph.forward_tensor(device.upload([2,1],[-2.,-2.]))
        failed = graph.snapshot()
        self.assertEqual(graph.forward_tensor(device.upload([2,1],[0.,0.])).snapshot().read_values(),[0.,0.])
        for read in (failed.read_values, bad.snapshot().read_values):
            with self.assertRaisesRegex(ValueError,"non-finite"): read()
        zero = model(0.).inference_plan([2,1]).compile_graph_wgpu()
        masked = zero.forward_tensor(bad)
        zero.forward_tensor(device.upload([2,1],[1.,1.]))
        with self.assertRaisesRegex(ValueError,"non-finite"): masked.snapshot().read_values()

    def test_graph_to_graph_owns_values_after_source_reuse(self):
        plan = model().inference_plan([2,1])
        graph, other = plan.compile_graph_wgpu(), plan.compile_graph_wgpu()
        self.assertEqual((graph.input_shape, graph.output_shape, graph.stage_count, graph.parameter_count), ((2,1),(2,1),2,1))
        self.assertNotEqual(graph.adapter_info()["device_type"], "Cpu")
        for call in (graph.dispatch, graph.snapshot, graph.output_tensor):
            with self.assertRaises(ValueError): call()
        device = graph.tensor_device()
        view = device.upload([2,2], [99.,-1.,99.,2.]).narrow(1,1,1)
        graph.set_input_tensor(view)
        for i in range(1,4): self.assertEqual(graph.dispatch(), i)
        snapshot = graph.snapshot()
        self.assertEqual((snapshot.shape,snapshot.generation,snapshot.submitted_dispatch), ((2,1),1,3))
        frozen = graph.output_tensor()
        other.set_input_tensor(frozen.add(device.upload([], [1.])))
        other.dispatch()
        chained = other.output_tensor().snapshot()
        graph.upload_values([0.,0.]); graph.dispatch()
        del graph, other, device, view
        gc.collect()
        self.assertEqual(snapshot.read_values(), [0.,4.])
        self.assertEqual(frozen.snapshot().read_values(), [0.,4.])
        self.assertEqual(chained.read_values(), [2.,10.])
        with self.assertRaisesRegex(RuntimeError,"consumed"): snapshot.read_values()

    def test_invalid_uploads_and_masked_failure_do_not_invalidate_old_captures(self):
        graph = model(float.fromhex("0x1.fffffep+127")).inference_plan([1]).compile_graph_wgpu()
        device = graph.tensor_device()
        graph.upload_values([0.]); graph.dispatch()
        for call in (lambda: graph.upload_values([]), lambda: graph.upload_values([float("nan")]),
                     lambda: graph.set_input_tensor(device.upload([1,1],[1.]))):
            with self.assertRaises(ValueError): call()
        self.assertEqual((graph.generation, graph.submitted_dispatches), (1,1))
        self.assertEqual(graph.snapshot().read_values(), [0.])
        graph.upload_values([-2.]); graph.dispatch()
        failed, frozen = graph.snapshot(), graph.output_tensor()
        graph.upload_values([0.]); graph.dispatch()
        self.assertEqual(graph.snapshot().read_values(), [0.])
        for read in (failed.read_values, frozen.snapshot().read_values):
            with self.assertRaisesRegex(ValueError,"non-finite"): read()
        consumer = model().inference_plan([1]).compile_graph_wgpu()
        consumer.set_input_tensor(frozen)
        for _ in range(2):
            consumer.dispatch()
            with self.assertRaisesRegex(ValueError,"non-finite"): consumer.snapshot().read_values()
        consumer.upload_values([1.]); consumer.dispatch()
        self.assertEqual(consumer.snapshot().read_values(), [2.])
        for options in (dict(kernel="bad"),dict(accumulation="bad"),dict(tile_mnk=[True,8,16])):
            with self.assertRaises((TypeError,ValueError)): model().inference_plan([1]).compile_graph_wgpu(**options)

    def test_training_and_specialized_dense_use_the_same_tensor_handle(self):
        device = st.WgpuTensorDevice.create()
        x, y = device.upload([2,1],[1.,2.]), device.upload([2,1],[0.,0.])
        plan = model().inference_plan([2,1])
        training = plan.compile_graph_training_wgpu(gradient_policy="exact")
        training.upload_batch_tensors(x,y)
        for invalid in (device.upload([1,2],[0.,0.]), device.upload([1],[0.])):
            with self.assertRaises(ValueError): training.upload_batch_tensors(x,invalid)
        self.assertEqual(training.batch_generation,1)
        training.step(.1)
        predicted, gradient, loss = training.prediction_tensor(), training.input_gradient_tensor(), training.loss_snapshot()
        graph = plan.compile_graph_wgpu()
        graph.set_input_tensor(predicted); graph.dispatch()
        result = graph.output_tensor().snapshot()
        training.upload_batch_tensors(y,y); training.step(0.)
        del training, graph
        gc.collect()
        self.assertEqual(loss.read(),10.)
        self.assertEqual(predicted.snapshot().read_values(),[2.,4.])
        self.assertEqual(gradient.snapshot().read_values(),[4.,8.])
        self.assertEqual(result.read_values(),[4.,8.])
        dense = st.nn.Sequential()
        dense.add(st.nn.Linear(1,1,name="linear"))
        dense.load_state_dict([("linear::weight",st.Tensor(1,1,[.5])),("linear::bias",st.Tensor(1,1,[.25]))])
        specialized = dense.inference_plan([2,1]).compile_wgpu()
        specialized.set_input_tensor(predicted); specialized.dispatch()
        self.assertEqual(specialized.tensor_snapshot(device).snapshot().read_values(),[1.25,2.25])


if __name__ == "__main__": unittest.main()
