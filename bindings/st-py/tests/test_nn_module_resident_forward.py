"""One original NN model, explicit host/resident inputs, no hidden transfers."""
import ast
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import unittest
import spiraltorch as st


def model():
    net = st.nn.Sequential()
    net.add(st.nn.Linear("up", 3, 4))
    net.add(st.nn.Gelu())
    net.add(st.nn.Scaler("gain", 4))
    net.add(st.nn.Relu())
    net.add(st.nn.Linear("down", 4, 3))
    return net


def flat(t):
    return [v for row in t.tolist() for v in row]


def mutable_legacy_export(tensor):
    capsule = tensor.__dlpack__(max_version=None, copy=False)
    pointer = ct.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ct.py_object, ct.c_char_p]
    pointer.restype = ct.c_void_p
    header = pointer(capsule, b"dltensor")
    # Legacy DLManagedTensor starts with DLTensor, whose first member is data.
    data = ct.cast(header, ct.POINTER(ct.c_void_p)).contents.value
    rows, cols = tensor.shape()
    return capsule, (ct.c_float * (rows * cols)).from_address(data)


class Surface(unittest.TestCase):
    def test_host_calls_and_type_errors_are_unchanged(self):
        x = st.Tensor(2, 3, [0.25]*6)
        for m in [model(), st.nn.Linear(3, 3), st.nn.Scaler("g", 3),
                  st.nn.Gelu(), st.nn.Relu()]:
            self.assertIsInstance(m(x), st.Tensor)
            self.assertEqual(m(x).tolist(), m.forward(x).tolist())
            with self.assertRaisesRegex(TypeError, "no implicit device transfer"):
                m([0.25]*6)
        if not st.wgpu_kernel_reports_available():
            for m in [model(), st.nn.Linear(3, 3), st.nn.Scaler("g", 3)]:
                with self.assertRaisesRegex(NotImplementedError, "wgpu"): m.resident_cache_info()
                with self.assertRaisesRegex(NotImplementedError, "wgpu"): m.clear_resident_cache()

    def test_stubs_keep_input_and_output_device_types(self):
        path = Path(__file__).resolve().parents[1] / "spiraltorch/__init__.pyi"
        classes = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        for name in ["Linear", "Sequential", "Scaler", "Gelu", "Relu"]:
            for method in ["forward", "__call__"]:
                overloads = [n for n in classes["_Nn"+name].body if isinstance(n, ast.FunctionDef) and n.name == method]
                self.assertEqual([(ast.unparse(n.args.args[1].annotation), ast.unparse(n.returns)) for n in overloads],
                                 [("Tensor", "Tensor"), ("WgpuTensor", "WgpuTensor")])


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class Gpu(unittest.TestCase):
    def setUp(self):
        self.d = st.WgpuTensorDevice.create()
        self.assertNotEqual(self.d.adapter_info()["device_type"], "Cpu")

    def close(self, gpu, host):
        values = gpu.snapshot().read_values()
        self.assertEqual(len(values), len(host))
        for a, b in zip(values, host): self.assertAlmostEqual(a, b, delta=2e-5)

    def test_late_dlpack_export_updates_cpu_pack_and_resident_graph(self):
        net = st.nn.Linear("late", 3, 5)
        host = st.Tensor(6, 3, [0.25]*18)
        gpu = self.d.upload([2, 3, 3], [0.25]*18)
        original_host = flat(net(host))
        original = net(gpu)
        self.close(net(gpu), original_host)
        state = dict(net.state_dict())
        weight = next(t for name, t in state.items() if name.endswith("::weight"))
        bias = next(t for name, t in state.items() if name.endswith("::bias"))
        weight_owner, weights = mutable_legacy_export(weight)
        bias_owner, biases = mutable_legacy_export(bias)
        self.close(net(gpu), original_host)
        self.assertEqual(net.resident_cache_info()["compilations"], 1)
        # Writes occur between calls, with live capsule owners, never concurrently.
        weights[-1] += 0.125
        changed_host = flat(net(host))
        self.assertNotEqual(changed_host, original_host)
        self.close(net(gpu), changed_host)
        self.assertEqual(net.resident_cache_info()["compilations"], 2)
        biases[-1] = float("nan")
        before = net.resident_cache_info()
        with self.assertRaises(ValueError): net(gpu)
        with self.assertRaises(ValueError): net(host)
        self.assertEqual(net.resident_cache_info(), before)
        biases[-1] = 0.
        self.close(net(gpu), changed_host)
        self.assertEqual(net.resident_cache_info()["compilations"], 2)
        self.close(original, original_host)
        del weights, biases, weight_owner, bias_owner

    def test_late_pointwise_stage_guards_survive_masking_and_valid_reuse(self):
        for bad in [0, 7, 11]:
            with self.subTest(bad=bad):
                net = st.nn.Sequential()
                for block in range(12):
                    gain = -float.fromhex("0x1.fffffep+127") if block == bad else 1.
                    net.add(st.nn.Scaler.from_gain(f"gain{block}", st.Tensor(1,1,[gain])))
                    net.add(st.nn.Relu())
                graph = net.inference_plan([2,1]).compile_graph_wgpu()
                invalid = graph.forward_tensor(self.d.upload([2,1],[2.,2.]))
                pending = graph.snapshot()
                safe = self.d.upload([2,1],[0.,0.])
                for _ in range(8):
                    graph.forward_tensor(safe)
                self.assertEqual(graph.snapshot().read_values(),[0.,0.])
                with self.assertRaisesRegex(ValueError,rf"stage {2*bad},"):
                    pending.read_values()
                with self.assertRaisesRegex(ValueError,"non-finite"):
                    invalid.snapshot().read_values()
                with self.assertRaisesRegex(ValueError,"non-finite"):
                    net(self.d.upload([2,1],[2.,2.])).snapshot().read_values()
                self.close(net(safe),[0.,0.])
                self.assertEqual(net.resident_cache_info()["compilations"],1)

    def test_live_versions_views_and_bound_consumers_survive_output_reuse(self):
        net = st.nn.Sequential()
        net.add(st.nn.Scaler.from_gain("gain",st.Tensor(1,1,[2.])))
        net.add(st.nn.Relu())
        outputs = [net(self.d.upload([2,1],[float(i+1)]*2)) for i in range(12)]
        for i,output in enumerate(outputs):
            self.assertEqual(output.snapshot().read_values(),[2.*(i+1)]*2)
            self.assertTrue(all(not output.shares_storage_with(other) for other in outputs[:i]))
        view = outputs[0].narrow(0,0,1)
        inputs = st.WgpuPointwiseInputs();inputs.add(outputs[0])
        pointwise = inputs.compile([("identity",None)])
        snapshot = outputs[1].snapshot()
        consumer = net.inference_plan([2,1]).compile_graph_wgpu()
        consumer.set_input_tensor(outputs[2])
        del outputs,output
        gc.collect()
        for i in range(20):
            latest = net(self.d.upload([2,1],[float(i+20)]*2))
        self.assertEqual(net.resident_cache_info(),dict(compilations=1,cache_hits=31,submitted_forwards=32))
        net.clear_resident_cache();del net
        gc.collect()
        self.assertEqual(latest.snapshot().read_values(),[78.,78.])
        self.assertEqual(view.snapshot().read_values(),[2.])
        self.assertEqual(snapshot.read_values(),[4.,4.])
        self.assertEqual(pointwise.run(inputs).snapshot().read_values(),[2.,2.])
        consumer.dispatch()
        self.assertEqual(consumer.snapshot().read_values(),[12.,12.])

    def test_twenty_ordinary_calls_chain_nd_views_and_own_outputs(self):
        net = model()
        values = [(i-7)/16 for i in range(24)]
        x = self.d.upload([2,4,3], values).permute([1,0,2])
        host = st.Tensor(8,3,x.snapshot().read_values())
        for i in range(20):
            x, host = net(x), net(host)
            if i == 0: first, expected_first = x, flat(host)
        self.assertIsInstance(x, st.WgpuTensor)
        self.assertEqual(x.shape, (4,2,3))
        self.assertEqual(net.resident_cache_info(), dict(compilations=1,cache_hits=19,submitted_forwards=20))
        net.clear_resident_cache()
        del net
        self.close(x, flat(host))
        self.close(first, expected_first)

    def test_state_load_and_checked_training_handoff_invalidate_cache(self):
        net = model()
        x = self.d.upload([2,3],[0.25]*6)
        host = st.Tensor(2,3,[0.25]*6)
        first = net(x)
        original = first.snapshot().read_values()
        state = net.state_dict()
        net.load_state_dict([(name, st.Tensor(*value.shape(), [v+0.125 for v in flat(value)]))
                             for name,value in state])
        self.close(net(x), flat(net(host)))
        self.assertEqual(net.resident_cache_info()["compilations"], 2)
        base = net.inference_plan([2,3])
        payload = json.loads(base.to_json())
        for p in payload["parameters"]: p["values"] = [v-0.25 for v in p["values"]]
        updated = st.nn.InferencePlan.from_json(json.dumps(payload))
        self.assertEqual(base.apply_parameters_to(net, updated), 5)
        self.close(net(x), flat(net(host)))
        self.assertEqual(net.resident_cache_info()["compilations"], 3)
        self.assertEqual(first.snapshot().read_values(), original)

    def test_standalone_layers_unsupported_modules_and_invalid_input(self):
        x = self.d.upload([2,3],[-1.,0.,2.,-1.,0.,2.])
        for net in [st.nn.Linear(3,3), st.nn.Scaler("g",3), st.nn.Gelu(), st.nn.Relu()]:
            self.close(net(x), flat(net(st.Tensor(2,3,[-1.,0.,2.,-1.,0.,2.]))))
        net = st.nn.Sequential()
        net.add(st.nn.LayerNorm("unsupported",3,-1.,1e-5))
        with self.assertRaises(ValueError): net(x)
        self.assertEqual(net.resident_cache_info()["compilations"], 0)
        large = self.d.upload([2,3],[float.fromhex("0x1.fffffep+127")]*6)
        invalid = large.mul(large)
        net = st.nn.Sequential()
        net.add(st.nn.Scaler.from_gain("zero",st.Tensor(1,3,[0.]*3)))
        net.add(st.nn.Relu())
        invalid = st.nn.Gelu()(net(invalid))
        with self.assertRaisesRegex(ValueError,"non-finite"): invalid.snapshot().read_values()
        self.close(net(x),[0.]*6)


if __name__ == "__main__": unittest.main()
