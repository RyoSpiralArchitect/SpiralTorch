# Pinned WebGPU compatibility backport

Source: the crates.io `wgpu 0.20.1` release, archive SHA-256
`90e37c7b9921b75dfd26dd973fdcbce36f13dfa6e2dc82aece584e0ed48c355c`.
Upstream MIT and Apache-2.0 licenses are retained unchanged.

This backports the two limit-mapping changes in
[gfx-rs/wgpu#6377](https://github.com/gfx-rs/wgpu/pull/6377):

- Do not read the removed `maxInterStageShaderComponents` browser property;
  report the legacy Rust field's default, as upstream does.
- Do not send that removed limit to `GPUAdapter.requestDevice`.

Without this fix, Chrome 152 rejects device creation before any compute is
submitted. It does not change native Vulkan/Metal/DX12 code, the public Rust
API, kernel precision, or device-selection policy. There is no global browser
monkeypatch and no CPU fallback. This vendored release is excluded from workspace
membership so project lint/format rules do not rewrite upstream code.

Remove this vendor patch as part of a separately tested workspace-wide WGPU
upgrade, not by silently substituting a new major dependency during a kernel
benchmark. The browser integration test exercises actual device creation and
compute using the patched dependency.

## Queue Completion

`ContextWebGpu::queue_on_submitted_work_done` in the pinned release was
`unimplemented!()`. It now registers the existing owning promise closures on
`GPUQueue.onSubmittedWorkDone()`. A resolved promise invokes the callback once;
a rejected promise drops it instead of certifying success. An owning oneshot
receiver therefore observes disconnection, not a successful completion or an
indefinite wait. The closures release each other after either outcome.

This is exercised by the resident graph profiling fixture's actual completions
and a deliberately rejected browser completion promise. The latter replacement
is restricted to the isolated test page, not production bindings or browser
globals used by a client. Native queue implementations are unchanged.
