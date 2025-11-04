# Getting Started with SpiralTorch

Welcome to SpiralTorch! This guide will help you get up and running with SpiralTorch, whether you're coming from PyTorch, JAX, TensorFlow, or starting fresh with deep learning.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Your First Model](#your-first-model)
5. [Training Basics](#training-basics)
6. [Z-Space Introduction](#z-space-introduction)
7. [Next Steps](#next-steps)

## Installation

### From PyPI (Recommended for Python Users)

```bash
pip install spiraltorch==0.3.0
```

The wheel is pre-built with `abi3` compatibility, supporting Python ≥ 3.8.

### From Source (For Rust Development)

Prerequisites:
- Rust stable (`rustup`)
- Cargo
- macOS: Xcode Command Line Tools / Linux: build-essentials

```bash
# Clone the repository
git clone https://github.com/RyoSpiralArchitect/SpiralTorch
cd SpiralTorch

# Build the workspace
cargo build --workspace --release

# Run tests
cargo test --workspace
```

### Building Python Wheel from Source

```bash
# CPU-only
maturin build -m bindings/st-py/Cargo.toml --release \
  --no-default-features --features cpu

# Metal (macOS, via WGPU)
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu

# CUDA (NVIDIA)
maturin build -m bindings/st-py/Cargo.toml --release --features cuda

# Install the wheel
pip install --force-reinstall --no-cache-dir target/wheels/spiraltorch-*.whl
```

## Quick Start

### Python: Your First Tensor

```python
import spiraltorch as st

# Create a tensor
x = st.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Shape: {x.shape()}")
print(f"Data: {x.tolist()}")

# Basic operations
y = x.scale(2.0)  # Multiply by scalar
z = x.hadamard(x)  # Element-wise multiplication

# Label your dimensions for clarity
labeled = st.tensor(
    [[0.2, 0.8], [0.4, 0.6]],
    axes=[st.Axis("batch", 2), st.Axis("feature", 2)],
)
print(f"Axis names: {labeled.axis_names()}")
```

### Rust: Pure Tensor Operations

```rust
use st_tensor::Tensor;

fn main() -> st_tensor::PureResult<()> {
    // Create a 2x3 tensor
    let x = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Scale by 2
    let y = x.scale(2.0)?;
    
    // Element-wise operations
    let z = x.hadamard(&x)?;
    
    println!("Shape: {:?}", x.shape());
    println!("Data: {:?}", x.data());
    
    Ok(())
}
```

## Core Concepts

### 1. Tensors and Shapes

SpiralTorch tensors are row-major 2D arrays (like NumPy and PyTorch). Multi-dimensional shapes are handled through labeled axes.

```python
import spiraltorch as st

# Simple 2D tensor
x = st.Tensor(3, 4)  # 3 rows, 4 columns (zero-initialized)

# From data
x = st.tensor([[1, 2, 3], [4, 5, 6]], dtype='f32')

# Shape information
rows, cols = x.shape()
print(f"{rows} × {cols}")
```

### 2. Zero-Copy Interop (DLPack)

SpiralTorch supports zero-copy tensor exchange with PyTorch, JAX, and other DLPack-compatible frameworks:

```python
import spiraltorch as st
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack

# SpiralTorch → PyTorch (zero-copy)
st_tensor = st.Tensor(2, 3, [1, 2, 3, 4, 5, 6])
capsule = st_tensor.to_dlpack()
torch_tensor = torch_from_dlpack(capsule)

# Mutations are visible in both (shared memory)
torch_tensor += 10
print("SpiralTorch sees changes:", st_tensor.tolist())

# PyTorch → SpiralTorch
pt_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
st_from_torch = st.Tensor.from_dlpack(pt_tensor)
```

### 3. Backends

SpiralTorch supports multiple backends:

- **CPU**: Always available, portable fallback
- **WGPU**: Metal (macOS), Vulkan, DirectX 12 (cross-platform GPU)
- **CUDA**: NVIDIA GPUs
- **HIP/ROCm**: AMD GPUs (Linux)

The framework automatically selects the best available backend or you can specify one explicitly.

## Your First Model

### Python: Simple Neural Network

```python
import spiraltorch as st

# Create a sequential model
from spiraltorch.nn import Sequential, Linear, Relu

model = Sequential()
model.add(Linear("layer1", in_features=10, out_features=20))
model.add(Relu())
model.add(Linear("layer2", in_features=20, out_features=5))

# Forward pass
x = st.Tensor(32, 10)  # Batch of 32 samples, 10 features each
output = model.forward(x)
print(f"Output shape: {output.shape()}")
```

### Rust: Neural Network Module

```rust
use st_nn::{Linear, Module, Relu, Sequential, Tensor};
use st_tensor::PureResult;

fn main() -> PureResult<()> {
    // Build a sequential model
    let mut model = Sequential::new();
    model.push(Linear::new("encoder", 10, 20)?);
    model.push(Relu::new());
    model.push(Linear::new("head", 20, 5)?);
    
    // Forward pass
    let input = Tensor::from_vec(1, 10, vec![0.1; 10])?;
    let output = model.forward(&input)?;
    
    println!("Output shape: {:?}", output.shape());
    
    Ok(())
}
```

## Training Basics

### Python: Complete Training Loop

```python
import spiraltorch as st

# Create model
model = st.nn.Sequential()
model.add(st.nn.Linear("encoder", 4, 8))
model.add(st.nn.Relu())
model.add(st.nn.Linear("head", 8, 2))

# Prepare training data
train_data = [
    (st.Tensor(1, 4, [0.1, -0.2, 0.3, -0.4]), st.Tensor(1, 2, [0.0, 1.0])),
    (st.Tensor(1, 4, [0.2, 0.1, -0.3, 0.5]), st.Tensor(1, 2, [1.0, 0.0])),
]

# Create trainer
from spiraltorch import ModuleTrainer

trainer = ModuleTrainer(
    device="cpu",
    curvature=-1.0,  # Hyperbolic geometry
    learning_rate=0.05,
    regularization=0.01
)
trainer.prepare(model)

# Train for one epoch
from spiraltorch.nn import MeanSquaredError
loss_fn = MeanSquaredError()

stats = trainer.train_epoch(
    model, 
    loss_fn, 
    train_data, 
    schedule=trainer.roundtable(rows=1, cols=2)
)

print(f"Average loss: {stats.average_loss:.6f}")
```

### Rust: Complete Training Loop

```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    Linear, MeanSquaredError, ModuleTrainer, Relu, RoundtableConfig, 
    Sequential, Tensor
};

fn main() -> st_nn::PureResult<()> {
    // Build model
    let mut model = Sequential::new();
    model.push(Linear::new("encoder", 4, 8)?);
    model.push(Relu::new());
    model.push(Linear::new("head", 8, 2)?);
    
    // Create trainer
    let mut trainer = ModuleTrainer::new(
        DeviceCaps::cpu(), 
        -1.0,   // curvature
        0.05,   // learning_rate
        0.01    // regularization
    );
    trainer.prepare(&mut model)?;
    
    // Prepare data
    let dataset = vec![
        (
            Tensor::from_vec(1, 4, vec![0.1, -0.2, 0.3, -0.4])?,
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        ),
        (
            Tensor::from_vec(1, 4, vec![0.2, 0.1, -0.3, 0.5])?,
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
        ),
    ];
    
    // Train one epoch
    let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
    let mut loss = MeanSquaredError::new();
    let stats = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;
    
    println!("Average loss: {:.6}", stats.average_loss);
    
    Ok(())
}
```

## Z-Space Introduction

Z-Space is SpiralTorch's unique geometric framework for training in hyperbolic space. This enables better representation of hierarchical structures and provides natural regularization.

### Why Z-Space?

- **Hierarchical Data**: Trees, graphs, taxonomies naturally live in hyperbolic space
- **Better Generalization**: Curvature provides implicit regularization
- **Richer Gradients**: Hyperbolic geometry allows more expressive optimization paths

### Using Hypergradient Tapes

```python
import spiraltorch as st

# Initialize weights with Z-space
weights = st.Tensor(1, 3, [0.1, 0.2, 0.3])

# Create hypergrad tape
tape = st.hg[weights](
    curvature=-0.9,      # Hyperbolic curvature
    learning_rate=0.02,
)

# Accumulate gradients
prediction = st.Tensor(1, 3, [0.25, 0.25, 0.25])
target = st.Tensor(1, 3, [0.0, 1.0, 0.0])
tape.accumulate_pair(prediction, target)

# Apply updates to weights
tape.apply(weights)

print("Tape shape:", tape.shape())
print("Learning rate:", tape.learning_rate())
```

### Z-Space Encoding

```python
import spiraltorch as st

# Encode text into Z-space
z_vec = st.z["Initialize the Z-space roundtable", 0.4]

# Create metrics bundle
metrics = st.z.metrics(
    speed=0.55,
    memory=0.12,
    stability=0.78,
    drs=0.05,
    gradient=[0.1, -0.2, 0.05],
)

# Train Z-space model
trainer = st.ZSpaceTrainer(z_dim=z_vec.shape()[1])
loss = trainer.step(metrics)

print(f"Z-space shape: {z_vec.shape()}, loss: {loss}")
```

## Next Steps

### Explore More Features

- **[SpiralTorchVision](../docs/spiraltorchvision.md)**: Computer vision with Z-space geometry
- **[SpiralTorchRL](../README.md#spiraltorchrl)**: Reinforcement learning with hypergradient policies
- **[SpiralTorchRec](../README.md#spiraltorchrec)**: Recommendation systems with topos guards
- **[Plugin System](../docs/plugin_system.md)**: Extend SpiralTorch with custom components

### Tutorials

- **[Module Discovery](../docs/ecosystem_integration.md)**: Build dynamic pipelines
- **[Custom Operators](../ECOSYSTEM_ENHANCEMENTS.md)**: Register custom tensor operations
- **[Backend Selection](../docs/backend_matrix.md)**: Optimize for your hardware

### Advanced Topics

- **[Z-Space Theory](../docs/zspace_intro.md)**: Deep dive into hyperbolic geometry
- **[SpiralK DSL](../README.md#heuristics-spiralk)**: Customize kernel heuristics
- **[Self-Rewrite](../README.md#self-rewrite)**: Automated performance tuning

### Community

- [GitHub Issues](https://github.com/RyoSpiralArchitect/SpiralTorch/issues): Report bugs or request features
- [Discussions](https://github.com/RyoSpiralArchitect/SpiralTorch/discussions): Ask questions and share ideas
- Email: kishkavsesvit@icloud.com

## Troubleshooting

### Common Issues

**Q: `AttributeError: module 'spiraltorch' has no attribute 'XYZ'`**  
A: Check that you're using the correct import. The Python wheel forwards many symbols dynamically. See `spiraltorch.pyi` for the supported API.

**Q: CUDA/ROCm link errors**  
A: Verify `CUDA_HOME`, driver/toolkit versions, or ROCm installation. On CI, add toolkit paths to `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`.

**Q: Wheel contains stale symbols after code changes**  
A: `pip uninstall -y spiraltorch && pip cache purge` → reinstall the freshly built wheel with `--no-cache-dir`.

**Q: Build fails with fontconfig errors**  
A: This is from optional plotting dependencies. Install fontconfig or build without visualization features.

### Getting Help

If you're stuck:

1. Check the [README](../README.md) for comprehensive examples
2. Browse [examples/](../examples/) for working code
3. Search [issues](https://github.com/RyoSpiralArchitect/SpiralTorch/issues) for similar problems
4. Open a new issue with a minimal reproducible example

---

**Welcome to the SpiralTorch community!** 🌀

We're excited to see what you'll build with Z-space geometry and portable GPU acceleration.
