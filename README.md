
RyoSpiralArchitect
Ryo ∴ SpiralArchitect
Skip to content
Navigation Menu
RyoSpiralArchitect
spiraltorch
 
Type / to search
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
spiraltorch
/

in
main

Edit

Preview
Indent mode

Indent size

Line wrap mode

Editing README.md file contents
Selection deleted
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141

---

## 🦀 Core (Rust)

```bash
cd crates/st-core
cargo test
```

---

## 📦 Wheels CI (tag and ship)

Push a tag to build wheels for 3.8–3.14 (incl. aarch64 + abi3) and publish to PyPI:

```bash
git tag v1.0.1
git push origin v1.0.1
```

> Set `PYPI_API_TOKEN` (scoped token) in repo secrets.  
> Username is `__token__` (already wired in the workflow).

---

## ✅ Compatibility Matrix

| OS / Arch                    | Python           | Wheel |
|-----------------------------|------------------|-------|
| Linux x86_64 / aarch64      | 3.8 – 3.14       | ✔️ manylinux2014 |
| macOS x86_64 / arm64        | 3.8 – 3.14       | ✔️ |
| Windows x86_64              | 3.8 – 3.14       | ✔️ |
| abi3 (cp38-abi3)            | 3.8+ (per-OS)    | ✔️ optional |

---

## 🧠 Why this exists

- Run Torch-like code **on Python 3.14 today**
- Readable core, hackable ops, no CMake nightmares
- Minimal surface area with real autograd semantics

---

## 🤝 Contributing

Early days. Fork it, break it, file issues.  
PRs welcome once the public API stabilizes.

---

## 📜 License

**AGPL-3.0-or later**

---

## 🌀 Author

**Ryo ∴ SpiralArchitect**  

> “The torch is just the beginning. The reality spirals out from here.”
```
**SpiralTorch-rs** is a fast, clean Rust implementation of a Torch-like tensor engine with autograd, plus Python bindings via PyO3.

## Highlights
- **Generalized `einsum`** with DP optimization (batch/broadcast aware) + greedy fallback
- **Segment ops**: `segment_{sum, mean, max, min}`, `unsorted_segment_*` (by index semantics), `ragged_segment_*` (via row_splits), and `coalesce_indices`
- **`logprod`** (stable log-domain product): returns `(logabs, sign)`, gradient flows through `logabs`
- **Exact gradients for `index_reduce(..., reduce="prod")`** even with zeros (base/src) and include_self
- **Multi-output autograd** node support
- Out-of-place ops (v1 policy), NumPy-like broadcasting

## Quickstart

### Rust
```bash
cargo test -p st-core
```
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.

Attach files by dragging & dropping, selecting or pasting them.
Editing spiraltorch/README.md at main · RyoSpiralArchitect/spiraltorch
