# Public resident vision training clients

This is synthetic correctness and handoff evidence, not real-image quality,
training throughput, PyTorch parity or a general cross-device bitwise guarantee.
Python and browser clients wrap the Rust-owned ConvNeXt classifier; no model,
loss, VJP, update or checkpoint semantics are reimplemented in either client.

## Recipe And Results

- One two-stage classifier: input `[2,1,8,8]`, stage dimensions `[2,4]`, depths
  `[1,1]`, patch `[2,2]`, epsilon `0.001`, two classes, initialization seed 17.
- Two synthetic 12x14 images pass through resident Normalize, resize, seeded
  horizontal flip and crop. CE uses mean reduction; plain-SGD rate is `0.01`.
- Four consecutive accepted updates, a frozen checkpoint at revision 2 read
  after the third update, and bitwise within-runtime continuation through step 4.
- Invalid labels reject attempted update 5 without changing any weights;
  valid retry is accepted at revision 6. Stale/foreign tokens are rejected.
  Retained prediction/checkpoint handles survive model destruction.
- Chrome `154.0.8037.58` reports the Rust `BrowserWebGpu` backend, device type
  `Other`, unnamed adapter. The native Python runtime reports Apple M4, Metal,
  `IntegratedGpu`; the browser descriptor is not hardware attestation.
- The revision-6 browser checkpoint roundtrips byte-for-byte through Python.
  Native continuation accepts revision 7 and compares all 24 parameter tensors
  (590 scalar values), loss and logits with the browser continuation.

| Browser-to-Python comparison | Maximum absolute error | Maximum scaled error |
| --- | ---: | ---: |
| Restored resident logits | 0 | 0 |
| Ordinary host inference logits | 7.46e-8 | 6.01e-8 |
| Continued CE loss | 5.97e-8 | 3.53e-8 |
| Every continued parameter | 0 | 0 |
| Continued logits | 0 | 0 |

Values in the table are rounded upward; [handoff.json](handoff.json) retains
the exact values and hashes. The bound is scaled error `< 2e-4`; matching
checkpoint hashes on this run do not strengthen that general contract.

## Verification

| Check | Result |
| --- | --- |
| Fresh default Python wheel, six targeted test files, real WGPU enabled | 22 passed, no skips |
| Native `st-vision --features wgpu --lib`, real WGPU enabled | 89 passed, one manual diagnostic ignored |
| CPU-only `st-vision --no-default-features --features nn --lib` | 72 passed |
| Native `st-nn --features wgpu --lib global_pool` | 3 passed, including four-thread shared pooling-cache use |
| Python default and CPU-only `python-default` feature checks | Passed |
| Full WASM `webgpu` release build and `nn`-only compile | Passed |
| Generated and shipped resident TypeScript contracts | Passed |
| Real Chrome public classifier fixture and browser-to-Python continuation | Passed |
| Handoff verifier negative tests | 3 passed, including truncation/non-finite/metadata/value mismatches |
| Runnable Python example in the new client guide | Passed, accepted revisions 1 through 4 |
| Workspace rustfmt and whitespace checks | Passed |

Strict Clippy attempts are not clean: dependency-inclusive Python lint stops in
unchanged `spiral-opt::ops::block`, and explicit `st-nn` lint stops on existing
warnings outside this patch. Python-binding-only strict lint also reports
existing argument-count/type-complexity warnings, including the prior vision
annotation methods. WASM-binding-only strict lint reports 19 existing warnings
in COBOL, cosmology, fractal, scale-stack, FFT and Mellin modules, not the new
classifier binding. These failures are retained locally rather than hidden or
mixed into this feature. CI's supported lint scope remains authoritative.

The first local browser launch failed because the fixture's initial result
element lacked the harness's running-state attribute. That harness integration
was fixed before the recorded successful run; the failed record is retained.
The first native binding check also revealed that the resident pooling cache
was not `Sync`. A synchronized Rust cache and concurrent regression test fixed
the cause, rather than making the Python model thread-affine.

## Replay And Evidence

Follow [the public client replay commands](../../../docs/resident_vision_training_clients.md#replay-the-public-client-checks).
Native test commands use Rust 1.98.0; formatting uses nightly. The installed
wheel is a local build of 0.4.27 with `nn,wgpu`, not a new PyPI release.
The source parent is `491aa1b4859c40b8157bfe471d6e8cf8549c1a61` plus this patch.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked \
  -p st-vision --features wgpu --lib
cargo +1.98.0 test --locked -p st-vision --no-default-features --features nn --lib
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test --locked \
  -p st-nn --features wgpu --lib global_pool
```

[browser-clients.json](browser-clients.json) publishes the complete condition,
checks, asset hashes and outputs, excluding the two raw checkpoint payloads and
normalized input. [handoff.json](handoff.json) records those checkpoint hashes,
the original browser-report hash and every cross-runtime comparison.
Original reports, checkpoints, build/test logs, wheel and WASM assets are kept
in the local validation archive; no large raw archives are added to Git.
[Source hashes](source-sha256.txt) and [validation log hashes](validation-log-sha256.txt)
identify the exercised code and retained logs, including failed attempts.
The log manifest also includes the CPU-only CI surface check, where the three
runtime GPU tests are explicitly skipped; this is separate from the 22-test
GPU-enabled run above.

Remaining gates: real-image matched PyTorch learning and transfer-inclusive
performance, data cursor/augmentation RNG/schedule restart, and resident
optimizer policy beyond plain SGD. Fixed batch shape is explicit; incompatible
tail batches fail rather than being silently dropped or padded.
