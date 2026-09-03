"""Exercise the installed native interchange contract without optional packages."""

import ctypes
import gc
import importlib

import spiraltorch as st


def main() -> None:
    native = importlib.import_module("spiraltorch.spiraltorch")
    source = st.Tensor(1, 2, [3., 5.])
    assert isinstance(source, native.Tensor), "DLPack smoke requires the native wheel"
    assert source.__dlpack_device__() == (1, 0)
    capsule_name = ctypes.pythonapi.PyCapsule_GetName
    capsule_name.argtypes = [ctypes.py_object]
    capsule_name.restype = ctypes.c_char_p

    for version in (None, (1, 0)):
        for copy in (None, False, True):
            capsule = source.__dlpack__(max_version=version, copy=copy, dl_device=(1, 0))
            assert capsule_name(capsule) == (
                b"dltensor" if version is None else b"dltensor_versioned"
            )
            restored = st.from_dlpack(capsule)
            assert restored.tolist() == [[3., 5.]]
            try:
                st.from_dlpack(capsule)
            except ValueError:
                pass
            else:
                raise AssertionError("a DLPack capsule was consumed twice")
            restored.add_row_inplace([1., 2.])
            assert restored.tolist() == [[4., 7.]]
            assert source.tolist() == [[3., 5.]]

    shared = st.from_dlpack(source)
    del source
    gc.collect()
    assert shared.tolist() == [[3., 5.]]
    print("native DLPack legacy/versioned, copy policies, lifetime, and single-use smoke passed")


if __name__ == "__main__":
    main()
