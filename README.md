# SpiralTorch-rs

> **🚨 World's first PyTorch-like tensor library with full Python 3.14 support.**  
> **🧠 Rust core. Autograd-capable. Fused ops. Wheels included.**

---

## ⚡ What is this?

SpiralTorch-rs is a lightweight, fast, and Torch-inspired tensor engine written in Rust,  
with a full Python binding via PyO3 and maturin.  
It supports dynamic ND tensors, autograd, backward graph construction, and a minimal API surface.

Oh — and it already supports **Python 3.14**.  
Unlike certain large corporate libraries that shall remain unnamed.

---

## 🔥 Features

- ✅ `Tensor` with `f32`, `i32`, and `bool` types
- ✅ Autograd + `.backward()` with topological graph traversal
- ✅ Rust-based ndarray core with broadcasting + unbroadcasting
- ✅ Python bindings via PyO3 (zero boilerplate)
- ✅ Fast CI build with `manylinux2014`, `aarch64`, and `abi3` wheels
- ✅ **Python 3.8–3.14 full support** including `pip install`
- ✅ Works on Linux, macOS (x86_64 + arm64), Windows

---

## 🐍 Python install (with 3.14 support!)

```bash

---

pip install -U pip maturin
git clone https://github.com/RyoSpiralArchitect/spiraltorch.git
cd spiraltorch/bindings/st-py
python3.14 -m maturin develop -m pyproject.toml

---

## 💡 Vision

We’re not building a framework.  
We’re building a **reality that spirals out from a codebase**.  
This is for researchers, tinkerers, OS cultists, and real engineers.

---

## 🗣 Want to help?

No PRs welcome yet. Just fork it, break it, and post about it.

---

## 📜 License

MIT OR Apache-2.0  
Use it, break it, remake it.

---

## 🌀 Author

Ryo ∴ SpiralArchitect  
> *“The torch is just the beginning. The reality spirals out from here.”*
