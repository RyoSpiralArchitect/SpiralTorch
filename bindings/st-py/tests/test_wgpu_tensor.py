"""Public immutable GPU tensors, not replacements for host Tensor storage."""
import gc
import os
import unittest
import spiraltorch as st


class TensorSurface(unittest.TestCase):
    def test_public_aliases_and_opaque_constructors(self):
        from spiraltorch.wgpu import WgpuTensorDevice, WgpuTensor, WgpuTensorSnapshot
        for cls in (WgpuTensorDevice, WgpuTensor, WgpuTensorSnapshot):
            self.assertIs(cls, getattr(st, cls.__name__))
            self.assertIn(cls.__name__, st.__all__)
            self.assertIn(cls.__name__, st.wgpu.__all__)
            with self.assertRaises(TypeError): cls()

    def test_cpu_build_explicitly_rejects_creation(self):
        if not st.wgpu_kernel_reports_available():
            with self.assertRaisesRegex(NotImplementedError, "wgpu"): st.WgpuTensorDevice.create()


@unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "real WGPU opt-in")
class TensorGpu(unittest.TestCase):
    def test_views_snapshots_and_shape_metadata(self):
        device = st.WgpuTensorDevice.create()
        self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
        values = [-1.,2.,3.,4.,5.,6.]
        x = device.upload([2,3], values)
        values[:] = [99.]*6
        view = x.permute([1,0]).narrow(0,1,2)
        self.assertEqual((view.shape,view.strides,view.offset,view.numel), ((2,2),(1,3),1,4))
        self.assertFalse(view.is_contiguous)
        self.assertTrue(x.shares_storage_with(view))
        packed = view.contiguous()
        self.assertTrue(packed.is_contiguous)
        self.assertFalse(packed.shares_storage_with(view))
        self.assertTrue(x.shares_storage_with(x.reshape([6])))
        broadcast = device.upload([1,2], [2.,3.]).broadcast_to([2,2])
        result = packed.mul(broadcast).add(device.upload([], [1.])).relu()
        snapshot = result.snapshot()
        del device, x, view, packed, result, broadcast
        gc.collect()
        self.assertEqual(snapshot.shape, (2,2))
        self.assertEqual(snapshot.read_values(), [5.,16.,7.,19.])
        self.assertEqual(snapshot.shape, (2,2))
        with self.assertRaisesRegex(RuntimeError, "consumed"): snapshot.read_values()

    def test_scalar_empty_and_shared_queue_interoperability(self):
        first, second = st.WgpuTensorDevice.create(), st.WgpuTensorDevice.create()
        scalar = first.upload([], [2.])
        self.assertEqual((scalar.shape,scalar.strides,scalar.offset,scalar.numel), ((),(),0,1))
        self.assertEqual(scalar.add(second.upload([], [3.])).snapshot().read_values(), [5.])
        self.assertEqual(first.upload([0,3], []).gelu().snapshot().read_values(), [])
        self.assertEqual(first.upload([1], [2.]).device().upload([1], [3.]).snapshot().read_values(), [3.])

    def test_strict_indices_layouts_and_immutable_properties(self):
        device = st.WgpuTensorDevice.create()
        x = device.upload([2,3], [0.]*6)
        for bad in (True,-1,1.5,2**65,"1"):
            for call in (lambda: device.upload([bad], []), lambda: x.narrow(bad,0,1),
                         lambda: x.narrow(0,bad,1), lambda: x.narrow(0,0,bad),
                         lambda: x.reshape([bad]), lambda: x.permute([bad,0])):
                with self.assertRaises((TypeError,ValueError,OverflowError)): call()
        for call in (lambda: device.upload([1], []), lambda: device.upload([1], [float("nan")]),
                     lambda: x.permute([0,0]), lambda: x.broadcast_to([2,2]),
                     lambda: x.reshape([5]), lambda: x.narrow(0,2,1),
                     lambda: x.add(device.upload([2,2],[0.]*4))):
            with self.assertRaises(ValueError): call()
        with self.assertRaises(AttributeError): x.shape = (999,)
        with self.assertRaises(TypeError): x.add(st.Tensor(2,3,[0.]*6))

    def test_deferred_failures_cannot_be_hidden_or_read_twice(self):
        device = st.WgpuTensorDevice.create()
        bad = device.upload([1], [-float.fromhex("0x1.fffffep+127")]).mul(device.upload([], [2.])).relu()
        snapshot = bad.reshape([1,1]).broadcast_to([2,1]).snapshot()
        del device, bad
        gc.collect()
        with self.assertRaisesRegex(ValueError, "non-finite"): snapshot.read_values()
        with self.assertRaisesRegex(RuntimeError, "consumed"): snapshot.read_values()


if __name__ == "__main__": unittest.main()
