# Rust-owned DLPack interoperability

The native Tensor bridge accepts **contiguous 2D CPU float32** tensors. It
supports legacy `DLManagedTensor` and the DLPack 1.0 versioned ABI. Python passes
the protocol and copy request to Rust; Rust owns descriptor validation, foreign
storage lifetime, read-only metadata, and copy-on-write mutation.

This is storage interchange, not a transfer of an autograd graph. Detach a
PyTorch tensor before importing it if it requires gradients. GPU buffers and
implicit dtype/layout conversions are not accepted. A tensor imported from CPU
memory can subsequently run SpiralTorch operations on an explicitly selected
backend, but that does not make the original DLPack buffer GPU-resident.

## Python

With an installed native wheel and NumPy 2.x:

```python
import numpy as np
import spiraltorch as st

weights = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
weights.flags.writeable = False
tensor = st.from_dlpack(weights)
shared = np.from_dlpack(tensor, copy=False)
assert np.shares_memory(weights, shared)

copied = np.from_dlpack(tensor, copy=True)
assert copied.flags.writeable
assert not np.shares_memory(weights, copied)

# Rust mutation materializes owned storage without changing the producer.
tensor.add_row_inplace([10., 20.])
assert tensor.tolist() == [[11., 22.], [13., 24.]]
assert weights.tolist() == [[1., 2.], [3., 4.]]
assert shared.tolist() == weights.tolist()
```

`Tensor.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`:

| Argument | Contract |
| --- | --- |
| `max_version=None` or `< (1, 0)` | Export a legacy `dltensor` capsule. |
| `max_version >= (1, 0)` | Export a `dltensor_versioned` capsule with version `(1, 0)`. |
| `copy=None` | Share unless a legacy consumer needs a copy of read-only storage. |
| `copy=False` | Never copy; raise `BufferError` if legacy export cannot preserve read-only storage. |
| `copy=True` | Export independent writable storage; versioned exports set `IS_COPIED`. |
| `dl_device` | Only `None` or `(1, 0)` is accepted; no silent device transfer. |
| `stream` | Only `None` or `0` is accepted for this CPU bridge. |

`Tensor.to_dlpack()` and `st.to_dlpack(tensor)` keep their legacy capsule API.
Both `st.from_dlpack(...)` and `Tensor.from_dlpack(...)` accept an array object
or either capsule format. Array imports request version `(1, 0)`; old producers
that reject the new keyword before entering Python code retain a narrow legacy
compatibility path. Exceptions from inside a Python producer are not retried.
NumPy 1.26's writable arrays remain supported through the legacy path.

Capsules are single-use, **including when shape/dtype/device validation fails**.
An unconsumed capsule releases its producer on destruction; a consumed capsule
transfers that responsibility to the receiving Tensor. Views and exports retain
the producer until the last dependent object is dropped.

Ordinary writable PyTorch CPU tensors keep the zero-copy path:

```python
import torch
import spiraltorch as st

source = torch.tensor([[1., 2.]], dtype=torch.float32)
tensor = st.from_dlpack(source)
shared = torch.from_dlpack(tensor)
assert source.data_ptr() == shared.data_ptr()
```

Avoid mutating shared foreign storage while Rust is reading or computing with
it, including from another thread. For isolation, ask for an explicit copy.
The source-only NumPy fallback is a separate compatibility implementation; the
native lifetime and zero-copy guarantees here require the installed Rust wheel.

`tensor.snapshot()` isolates pre-existing mutable aliases. Snapshot exports
share read-only versioned storage; legacy exports copy instead of exposing a
writable alias. `tensor.is_snapshot()` reports this protection. Autograd leaves
and saved gradients use snapshots automatically so a later NumPy/PyTorch write
cannot silently change a prior graph. Ordinary Tensor interchange still shares
storage when allowed. See the [autograd contract](autograd_contract.md).

## Rust

Direct Rust callers need no raw pointers for an internal handoff:

```rust
use st_tensor::dlpack::{DlpackCopyPolicy, DlpackExportOptions};
use st_tensor::{PureResult, Tensor};

fn roundtrip(source: &Tensor) -> PureResult<Tensor> {
    let handle = source.export_dlpack(DlpackExportOptions {
        copy: DlpackCopyPolicy::Never,
        ..Default::default()
    })?;
    Tensor::from_managed_dlpack(handle)
}
```

`ManagedTensor` owns one release obligation and calls the producer's deleter
when dropped, including failed imports. `into_raw()` transfers that obligation
to an FFI consumer. `as_ptr()` only borrows it. External raw imports remain
`unsafe`: the producer must supply live, aligned descriptors and initialized
storage of the declared extent, retain storage until release, and synchronize
access. A deleter that touches Python must acquire the GIL itself; a null
deleter requires the producer to keep storage alive independently.

Unknown major versions are rejected using only the stable version/deleter
prefix. Newer minor versions are accepted only when their fields are understood;
unsupported flags, non-f32 dtypes, non-CPU devices, non-contiguous layouts,
misalignment, and overflowing byte spans are rejected. Empty axes and irrelevant
singleton-axis strides are supported. The C ABI uses a 64-bit byte offset even
on WASM32. DLPack's 1.3 C Exchange API is not implemented here.

Run the safe Rust example with:

```sh
cargo run -p st-tensor --example dlpack_interop
```

Reference: [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html)
and [ABI definitions](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h).
