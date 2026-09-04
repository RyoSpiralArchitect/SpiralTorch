# Pinned WebGPU compatibility backport

Source: the crates.io `wgpu 0.20.1` release, archive SHA-256
`90e37c7b9921b75dfd26dd973fdcbce36f13dfa6e2dc82aece584e0ed48c355c`.
Upstream MIT and Apache-2.0 licenses are retained unchanged. Only
`src/backend/webgpu.rs` differs from the released Rust sources.

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
