# Migrating from PyTorch to SpiralTorch

This guide helps PyTorch users transition to SpiralTorch while leveraging your existing knowledge. SpiralTorch maintains a PyTorch-like API while adding Z-space geometry and portable GPU support.

## Table of Contents

1. [Quick Comparison](#quick-comparison)
2. [Tensor Operations](#tensor-operations)
3. [Neural Network Modules](#neural-network-modules)
4. [Training Loops](#training-loops)
5. [Autograd and Gradients](#autograd-and-gradients)
6. [Zero-Copy Integration](#zero-copy-integration)
7. [Advanced Features](#advanced-features)
8. [Migration Checklist](#migration-checklist)

## Quick Comparison

| Feature | PyTorch | SpiralTorch |
|---------|---------|-------------|
| **Tensor Creation** | `torch.tensor([...])` | `st.tensor([...])` |
| **Device Selection** | `.to('cuda')` | Automatic (WGPU/CUDA/CPU) |
| **Forward Pass** | `model(x)` | `model.forward(x)` |
| **Gradient Computation** | `loss.backward()` | Hypergrad tape |
| **Optimizer** | `torch.optim.Adam` | `ModuleTrainer` |
| **Data Loading** | `DataLoader` | `Dataset.loader()` |
| **Geometry** | Euclidean only | Euclidean + Hyperbolic (Z-space) |
| **Backends** | CUDA-centric | WGPU-first (Metal/Vulkan/DX12/CUDA) |

## Tensor Operations

### Creating Tensors

**PyTorch:**
```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 5)
```

**SpiralTorch:**
```python
import spiraltorch as st

x = st.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
zeros = st.Tensor(3, 4)  # Zero-initialized by default
ones = st.Tensor(2, 5, [1.0] * 10)  # Explicit values
```

### Basic Operations

**PyTorch:**
```python
# Element-wise operations
y = x * 2
z = x + y
w = x * y

# Matrix multiplication
result = torch.matmul(x, y.T)

# Reductions
mean = x.mean()
sum_vals = x.sum()
```

**SpiralTorch:**
```python
# Element-wise operations
y = x.scale(2.0)
z = x.add(y)
w = x.hadamard(y)

# Matrix multiplication
result = x.matmul(y.transpose())

# Reductions
mean = x.mean()
sum_vals = x.sum()
```

### Reshaping and Views

**PyTorch:**
```python
x = torch.randn(2, 3, 4)
x_flat = x.view(-1)
x_reshape = x.reshape(6, 4)
```

**SpiralTorch:**
```python
# SpiralTorch uses 2D tensors with labeled axes
x = st.Tensor(6, 4)  # Directly create desired shape

# For higher dimensions, use labeled axes
time = st.Axis("time", 2)
height = st.Axis("height", 3)
width = st.Axis("width", 4)
x = st.tensor(..., axes=[time, height, width])
```

## Neural Network Modules

### Defining a Model

**PyTorch:**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MyModel()
```

**SpiralTorch:**
```python
import spiraltorch as st
from spiraltorch.nn import Sequential, Linear, Relu

model = Sequential()
model.add(Linear("fc1", 784, 128))
model.add(Relu())
model.add(Linear("fc2", 128, 10))

# Or using the builder pattern:
from spiraltorch.nn import ModuleDiscoveryRegistry, ModulePipelineBuilder

registry = ModuleDiscoveryRegistry()
# ... register custom modules ...

model = ModulePipelineBuilder(registry) \
    .add_module("linear", {"in": "784", "out": "128"}) \
    .add_module("relu", {}) \
    .add_module("linear", {"in": "128", "out": "10"}) \
    .build()
```

### Custom Modules

**PyTorch:**
```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias
```

**SpiralTorch (Rust):**
```rust
use st_nn::{Module, Parameter, Tensor};
use st_tensor::PureResult;

pub struct CustomLayer {
    weight: Parameter,
    bias: Parameter,
}

impl CustomLayer {
    pub fn new(name: &str, in_features: usize, out_features: usize) -> PureResult<Self> {
        let weight = Parameter::new(
            format!("{}.weight", name),
            Tensor::randn(in_features, out_features)?,
        );
        let bias = Parameter::new(
            format!("{}.bias", name),
            Tensor::zeros(1, out_features)?,
        );
        Ok(Self { weight, bias })
    }
}

impl Module for CustomLayer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let out = input.matmul(self.weight.value())?;
        out.add(self.bias.value())
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }
}
```

## Training Loops

### Basic Training Loop

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

**SpiralTorch:**
```python
import spiraltorch as st
from spiraltorch import ModuleTrainer
from spiraltorch.nn import MeanSquaredError

model = st.nn.Sequential()
# ... build model ...

trainer = ModuleTrainer(
    device="auto",  # Automatically selects best backend
    curvature=-1.0,
    learning_rate=0.001,
    regularization=0.0001
)
trainer.prepare(model)

loss_fn = MeanSquaredError()
schedule = trainer.roundtable(rows=1, cols=10)  # Output dimensions

for epoch in range(10):
    stats = trainer.train_epoch(model, loss_fn, train_data, schedule)
    print(f"Epoch {epoch}, Loss: {stats.average_loss:.6f}")
```

### With Custom Loss

**PyTorch:**
```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()

criterion = CustomLoss()
```

**SpiralTorch (Rust):**
```rust
use st_nn::{Loss, Tensor};
use st_tensor::PureResult;

pub struct CustomLoss;

impl Loss for CustomLoss {
    fn forward(&mut self, pred: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        let diff = pred.sub(target)?;
        let squared = diff.hadamard(&diff)?;
        Ok(squared.mean())
    }
}
```

## Autograd and Gradients

### Computing Gradients

**PyTorch:**
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum() ** 2
y.backward()
print(x.grad)  # Gradients accumulated in .grad
```

**SpiralTorch:**
```python
# SpiralTorch uses hypergrad tapes for gradient management
x = st.Tensor(1, 3, [1.0, 2.0, 3.0])
tape = st.hg[x](curvature=-1.0, learning_rate=0.01)

# Accumulate gradients
prediction = st.Tensor(1, 3, [2.0, 3.0, 4.0])
target = st.Tensor(1, 3, [1.5, 2.5, 3.5])
tape.accumulate_pair(prediction, target)

# Apply gradient updates
tape.apply(x)

# For Euclidean gradients (no curvature):
from spiraltorch import Realgrad
realgrad = Realgrad(x, learning_rate=0.01)
realgrad.accumulate_pair(prediction, target)
realgrad.apply(x)
```

### Custom Backward Pass

**PyTorch:**
```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2

custom_fn = CustomFunction.apply
```

**SpiralTorch:**
```python
# Register custom operator with backward pass
from spiraltorch.ops import register_operator, OperatorBuilder

def forward_fn(inputs):
    return [inputs[0].scale(2.0)]

def backward_fn(inputs, grad_outputs):
    return [grad_outputs[0].scale(2.0)]

op = OperatorBuilder("custom_double", 1, 1) \
    .with_backend("CPU") \
    .with_forward(forward_fn) \
    .with_backward(backward_fn) \
    .build()

register_operator(op)
```

## Zero-Copy Integration

One of SpiralTorch's most powerful features is zero-copy integration with PyTorch through DLPack:

```python
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import spiraltorch as st

# PyTorch → SpiralTorch (zero-copy)
pt_tensor = torch.randn(5, 10)
st_tensor = st.Tensor.from_dlpack(pt_tensor)

# Modifications are visible in both
pt_tensor += 1.0
print("SpiralTorch sees update:", st_tensor.tolist()[0][0])

# SpiralTorch → PyTorch (zero-copy)
st_tensor2 = st.Tensor(3, 3, [float(i) for i in range(9)])
capsule = st_tensor2.to_dlpack()
pt_tensor2 = from_dlpack(capsule)

# Use PyTorch operations on SpiralTorch data
pt_result = torch.nn.functional.relu(pt_tensor2)

# Convert back to SpiralTorch
st_result = st.Tensor.from_dlpack(pt_result)
```

### Mixed Training Pipeline

You can use PyTorch for data preprocessing and SpiralTorch for training:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import spiraltorch as st

# Prepare data with PyTorch
pt_x = torch.randn(1000, 784)
pt_y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(pt_x, pt_y)
loader = DataLoader(dataset, batch_size=32)

# Train with SpiralTorch
model = st.nn.Sequential()
# ... build model ...

for pt_batch_x, pt_batch_y in loader:
    # Zero-copy conversion
    st_x = st.Tensor.from_dlpack(pt_batch_x)
    st_y = st.Tensor.from_dlpack(pt_batch_y)
    
    # Train in Z-space
    # ... training code ...
```

## Advanced Features

### 1. Hyperbolic Geometry (Unique to SpiralTorch)

PyTorch operates in Euclidean space. SpiralTorch adds hyperbolic geometry through Z-space:

```python
# Hyperbolic cross-entropy for hierarchical classification
from spiraltorch.nn import HyperbolicCrossEntropy

loss_fn = HyperbolicCrossEntropy(curvature=-0.95)
loss = loss_fn.forward(logits, targets)

# Z-space projector for hierarchical embeddings
from spiraltorch.nn import ZSpaceProjector, LanguageWaveEncoder
from spiraltorch.topos import OpenCartesianTopos

encoder = LanguageWaveEncoder(curvature=-0.9, temperature=0.7)
topos = OpenCartesianTopos(curvature=-0.9, tolerance=1e-6, 
                          saturation=1e4, max_volume=512, max_depth=16384)
projector = ZSpaceProjector(topos, encoder)

# Encode hierarchical text
z_embedding = projector.encode_text("Deep learning in hyperbolic space")
```

### 2. Multi-Backend Support

PyTorch is CUDA-centric. SpiralTorch supports WGPU (Metal/Vulkan/DX12), CUDA, and CPU:

```python
# Automatic backend selection
trainer = st.ModuleTrainer(device="auto")

# Explicit backend
trainer = st.ModuleTrainer(device="wgpu")  # Metal on macOS, Vulkan/DX12 elsewhere
trainer = st.ModuleTrainer(device="cuda")  # NVIDIA GPUs
trainer = st.ModuleTrainer(device="cpu")   # CPU fallback
```

### 3. SpiralK Kernel Heuristics

SpiralTorch includes a DSL for optimizing kernel selection:

```python
# Enable soft heuristics
import os
os.environ['SPIRAL_HEUR_SOFT'] = '1'
os.environ['SPIRAL_HEUR_K'] = '''
  mk: sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  tile: sel(log2(c)>15.0, 2048, sel(log2(c)>13.0, 1024, 512));
'''

# The runtime will now optimize kernel selection automatically
```

## Migration Checklist

### Phase 1: Setup and Exploration

- [ ] Install SpiralTorch: `pip install spiraltorch`
- [ ] Run basic tensor operations to verify installation
- [ ] Test zero-copy DLPack integration with existing PyTorch code
- [ ] Identify which parts of your model can benefit from Z-space geometry

### Phase 2: Model Migration

- [ ] Convert tensor creation calls (`torch.tensor` → `st.tensor`)
- [ ] Rewrite custom modules using SpiralTorch's `Module` trait
- [ ] Update forward passes (`model(x)` → `model.forward(x)`)
- [ ] Replace activation functions with SpiralTorch equivalents

### Phase 3: Training Loop Migration

- [ ] Replace `optimizer.zero_grad()` with hypergrad tape management
- [ ] Convert loss functions to SpiralTorch losses
- [ ] Update data loading (or keep PyTorch DataLoader with DLPack bridge)
- [ ] Replace `optimizer.step()` with `trainer.train_epoch()`

### Phase 4: Optimization

- [ ] Benchmark different backends (WGPU, CUDA, CPU)
- [ ] Enable SpiralK heuristics for kernel optimization
- [ ] Explore Z-space features for hierarchical data
- [ ] Fine-tune hypergradient tape parameters

### Phase 5: Advanced Features

- [ ] Add PSI telemetry for runtime observability
- [ ] Integrate with SpiralTorchVision for computer vision
- [ ] Use SpiralTorchRL for reinforcement learning
- [ ] Explore Canvas Transformer for visualization

## Common Pitfalls

### 1. Forward Pass Syntax

❌ **Don't:**
```python
output = model(x)  # PyTorch style
```

✅ **Do:**
```python
output = model.forward(x)  # SpiralTorch explicit forward
```

### 2. Gradient Accumulation

❌ **Don't:**
```python
loss.backward()  # PyTorch autograd
optimizer.step()
```

✅ **Do:**
```python
tape = st.hg[weights](curvature=-1.0, learning_rate=0.01)
tape.accumulate_pair(prediction, target)
tape.apply(weights)
```

### 3. Tensor Reshaping

❌ **Don't:**
```python
x = x.view(-1, 784)  # PyTorch view
```

✅ **Do:**
```python
# SpiralTorch uses 2D tensors; reshape data before creating tensor
# Or use labeled axes for semantic dimensions
x = st.tensor(data, axes=[st.Axis("batch"), st.Axis("features", 784)])
```

## Getting Help

- **Documentation**: [README.md](../README.md) and [docs/](../docs/)
- **Examples**: Browse [examples/](../examples/) for migration patterns
- **Community**: [GitHub Discussions](https://github.com/RyoSpiralArchitect/SpiralTorch/discussions)
- **Issues**: Report bugs at [GitHub Issues](https://github.com/RyoSpiralArchitect/SpiralTorch/issues)

---

**Ready to migrate?** Start with a small module, verify it works with DLPack, then gradually expand. The zero-copy integration means you can migrate incrementally without rewriting everything at once.

Good luck! 🌀
