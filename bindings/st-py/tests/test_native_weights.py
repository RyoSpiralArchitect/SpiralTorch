"""Checkpoint ingress, N-D views and native execution without optional ML packages."""

from array import array
import importlib.abc
import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "numpy", "safetensors", "accelerate"}:
            raise AssertionError("native weight loader must not import " + fullname)


if __name__ == "__main__":
    assert not any(name.split(".")[0] in {"torch", "transformers", "numpy", "safetensors", "accelerate"}
                   for name in sys.modules)
    sys.meta_path.insert(0, NoForeignML())

import spiraltorch as st  # noqa: E402 - guard optional imports before importing the package.


def encode_header(header, payload=b""):
    raw = header.encode()
    return struct.pack("<Q", len(raw)) + raw + payload


def fixture(dtype="F32", shape=(2, 2), values=(1., -2., .5, 4.)):
    if dtype == "F32":
        payload = struct.pack("<" + "f" * len(values), *values)
    elif dtype == "F16":
        payload = struct.pack("<" + "e" * len(values), *values)
    elif dtype == "BF16":
        payload = b"".join(struct.pack("<f", x)[2:] for x in values)
    elif dtype == "I64":
        payload = struct.pack("<" + "q" * len(values), *values)
    else:
        raise AssertionError(dtype)
    return encode_header(json.dumps({"w": {"dtype": dtype, "shape": shape, "data_offsets": [0, len(payload)]},
                                     "__metadata__": {"format": "pt", "scope": "synthetic_test"}}), payload)


class NativeWeightTests(unittest.TestCase):
    def test_three_dtypes_reach_native_matmul(self):
        for dtype in ("F32", "F16", "BF16"):
            with self.subTest(dtype=dtype):
                archive = st.SafeTensorArchive.from_bytes(fixture(dtype))
                self.assertEqual(archive.names(), ["w"])
                self.assertEqual(archive.metadata()["format"], "pt")
                self.assertEqual(archive.info("w"), {
                    "dtype": dtype, "shape": [2, 2], "elements": 4, "byte_len": 16 if dtype == "F32" else 8})
                weight = archive.tensor("w")
                self.assertEqual(weight.shape, [2, 2])
                self.assertEqual(len(weight), 4)
                self.assertEqual(weight.values(), [1., -2., .5, 4.])
                x = st.Tensor([[2., 3.]])
                self.assertEqual(x.matmul(weight.to_tensor(), backend="cpu").tolist(), [[3.5, 8.]])
        self.assertIn("SafeTensorArchive", st.__all__)
        self.assertIn("NdTensor", st.__all__)
        native = importlib.import_module("spiraltorch.spiraltorch")
        for name in ("SafeTensorArchive", "NdTensor"):
            self.assertIn(name, native.__all__)
            self.assertIs(getattr(st, name), getattr(native, name))

    def test_nd_views_do_not_flatten_silently(self):
        weight = st.SafeTensorArchive.from_bytes(fixture(shape=(1, 2, 2))).tensor("w")
        self.assertEqual(weight.shape, [1, 2, 2])
        with self.assertRaisesRegex(ValueError, "rank two"):
            weight.to_tensor()
        matrix = weight.reshape([2, 2]).permute([1, 0])
        self.assertEqual(matrix.values(), [1., .5, -2., 4.])
        self.assertEqual(matrix.to_tensor().tolist(), [[1., .5], [-2., 4.]])
        self.assertEqual(weight.values(), [1., -2., .5, 4.])
        for shape in ([5], [True, 4], [-1], [1] * 129):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                weight.reshape(shape)
        for axes in ([0, 0, 1], [3, 0, 1], [True, 0, 1]):
            with self.assertRaises((ValueError, TypeError)):
                weight.permute(axes)

    def test_scalar_empty_tensor_and_empty_archive(self):
        scalar = st.SafeTensorArchive.from_bytes(fixture(shape=(), values=(.5,))).tensor("w")
        self.assertEqual(scalar.shape, [])
        self.assertEqual(scalar.reshape([1, 1]).to_tensor().tolist(), [[.5]])
        empty = st.SafeTensorArchive.from_bytes(fixture(shape=(0, 2, 3), values=())).tensor("w")
        self.assertEqual(empty.shape, [0, 2, 3])
        self.assertEqual(empty.values_buffer(), b"")
        self.assertEqual(st.SafeTensorArchive.from_bytes(encode_header("{}")).names(), [])

    def test_owned_file_and_buffer_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(fixture())
            archive = st.SafeTensorArchive(path)
            path.write_bytes(b"")
            weight = archive.tensor("w")
            exported = weight.values_buffer()
            self.assertEqual(exported, array("f", [1., -2., .5, 4.]).tobytes())
            exported[:] = bytes(len(exported))
            metadata = archive.metadata()
            metadata["format"] = "changed"
            info = archive.info("w")
            info["shape"][0] = 99
            self.assertEqual(archive.metadata()["format"], "pt")
            self.assertEqual(archive.info("w")["shape"], [2, 2])
            self.assertEqual(weight.values(), [1., -2., .5, 4.])
            with self.assertRaises(ValueError):
                st.SafeTensorArchive(path)
            with self.assertRaises(OSError):
                st.SafeTensorArchive(path.with_name("missing"))
        with self.assertRaises(TypeError):
            st.SafeTensorArchive.from_bytes(bytearray(fixture()))

    def test_snapshot_from_existing_tensor(self):
        source = st.Tensor([[1., 2.], [3., 4.]])
        snapshot = st.NdTensor.from_tensor(source)
        source.add_row_inplace([10., 20.])
        self.assertEqual(snapshot.to_tensor().tolist(), [[1., 2.], [3., 4.]])

    def test_budgets_checked_and_zero_elements_allowed(self):
        payload = fixture()
        header_len = struct.unpack("<Q", payload[:8])[0]
        kwargs = dict(max_file_bytes=len(payload), max_header_bytes=header_len, max_tensors=1,
                      max_rank=2, max_tensor_elements=4)
        self.assertEqual(st.SafeTensorArchive.from_bytes(payload, **kwargs).tensor("w").values(), [1., -2., .5, 4.])
        for key, value in (("max_file_bytes", len(payload) - 1), ("max_header_bytes", header_len - 1),
                           ("max_tensors", 0), ("max_rank", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                st.SafeTensorArchive.from_bytes(payload, **{**kwargs, key: value})
        for key in kwargs:
            with self.subTest(key=key), self.assertRaises(TypeError):
                st.SafeTensorArchive.from_bytes(payload, **{key: True})
        limited = st.SafeTensorArchive.from_bytes(payload, max_tensor_elements=3)
        self.assertEqual(limited.info("w")["elements"], 4)
        with self.assertRaisesRegex(ValueError, "element budget"):
            limited.tensor("w")
        empty = st.SafeTensorArchive.from_bytes(fixture(shape=(0,), values=()), max_tensor_elements=0)
        self.assertEqual(empty.tensor("w").values(), [])

    def test_unsupported_and_nonfinite_do_not_silently_convert(self):
        archive = st.SafeTensorArchive.from_bytes(fixture("I64", values=(1, 2, 3, 4)))
        self.assertEqual(archive.info("w")["dtype"], "I64")
        with self.assertRaisesRegex(ValueError, "supported: F32, F16, BF16"):
            archive.tensor("w")
        with self.assertRaises(KeyError):
            archive.tensor("missing")
        with self.assertRaises(KeyError):
            archive.info("missing")
        for dtype in ("F32", "F16", "BF16"):
            for value in (float("nan"), float("inf"), -float("inf")):
                bad = st.SafeTensorArchive.from_bytes(fixture(dtype, (1,), (value,)))
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    bad.tensor("w")

    def test_hostile_headers_and_trailing_payload(self):
        tensor = '{"dtype":"F32","shape":[0],"data_offsets":[0,0]}'
        headers = [
            '{"w":' + tensor + ',"w":' + tensor + '}',
            '{"w":{"dtype":"F32","dtype":"F16","shape":[0],"data_offsets":[0,0]}}',
            '{"w":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}',
            '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,18446744073709551615]}}',
            '{"__metadata__":{"k":"one","k":"two"}}',
            '{}{}', ' {}',
        ]
        for header in headers:
            with self.subTest(header=header), self.assertRaises(ValueError):
                st.SafeTensorArchive.from_bytes(encode_header(header))
        for payload in (b"", struct.pack("<Q", 2**64 - 1), fixture() + b"x", fixture()[:-1]):
            with self.assertRaises(ValueError):
                st.SafeTensorArchive.from_bytes(payload)


if __name__ == "__main__":
    unittest.main()
