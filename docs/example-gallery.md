# SpiralTorch Example Gallery

Welcome to the SpiralTorch Example Gallery! This collection demonstrates the framework's capabilities across different domains and skill levels.

## Table of Contents

- [Getting Started](#getting-started)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Reinforcement Learning](#reinforcement-learning)
- [Graph Neural Networks](#graph-neural-networks)
- [Recommendation Systems](#recommendation-systems)
- [Time Series](#time-series)
- [Advanced Z-Space](#advanced-z-space)
- [Plugin Development](#plugin-development)
- [Performance Optimization](#performance-optimization)

---

## Getting Started

### 01: Basic Tensor Operations
**File:** `examples/01_basics/tensor_operations.py`

Learn the fundamentals of creating and manipulating tensors in SpiralTorch.

```python
import spiraltorch as st

# Create tensors
x = st.tensor([[1.0, 2.0], [3.0, 4.0]])
y = st.tensor([[5.0, 6.0], [7.0, 8.0]])

# Basic operations
z = x.add(y)          # Element-wise addition
w = x.hadamard(y)     # Element-wise multiplication  
m = x.matmul(y.transpose())  # Matrix multiplication

print(f"Addition: {z.tolist()}")
print(f"Hadamard: {w.tolist()}")
print(f"Matmul: {m.tolist()}")
```

**Concepts:** Tensor creation, basic arithmetic, matrix operations

---

### 02: Your First Neural Network
**File:** `examples/01_basics/first_neural_network.py`

Build a simple feedforward network and train it on synthetic data.

```python
import spiraltorch as st
from spiraltorch.nn import Sequential, Linear, Relu, MeanSquaredError
from spiraltorch import ModuleTrainer

# Build model
model = Sequential()
model.add(Linear("layer1", 4, 8))
model.add(Relu())
model.add(Linear("layer2", 8, 2))

# Prepare data
train_data = [
    (st.Tensor(1, 4, [0.1, -0.2, 0.3, 0.4]), st.Tensor(1, 2, [1.0, 0.0])),
    (st.Tensor(1, 4, [0.5, 0.2, -0.1, 0.3]), st.Tensor(1, 2, [0.0, 1.0])),
]

# Train
trainer = ModuleTrainer(device="cpu", curvature=-1.0, learning_rate=0.05, regularization=0.01)
trainer.prepare(model)

loss_fn = MeanSquaredError()
schedule = trainer.roundtable(rows=1, cols=2)

for epoch in range(10):
    stats = trainer.train_epoch(model, loss_fn, train_data, schedule)
    print(f"Epoch {epoch}: Loss = {stats.average_loss:.6f}")
```

**Concepts:** Sequential models, training loops, loss functions

---

## Computer Vision

### 03: MNIST Classification
**File:** `examples/02_vision/mnist_classifier.py`

Train a convolutional network on MNIST digits using Z-space geometry.

```python
from spiraltorch.nn import Sequential, Conv2d, MaxPool2d, Linear, Relu
import spiraltorch as st

model = Sequential()
model.add(Conv2d("conv1", in_channels=1, out_channels=32, kernel_size=3))
model.add(Relu())
model.add(MaxPool2d(kernel_size=2))
model.add(Conv2d("conv2", in_channels=32, out_channels=64, kernel_size=3))
model.add(Relu())
model.add(MaxPool2d(kernel_size=2))
model.add(Linear("fc1", 64 * 5 * 5, 128))
model.add(Relu())
model.add(Linear("fc2", 128, 10))

# Training code similar to example 02
```

**Concepts:** Convolutional layers, pooling, image classification

---

### 04: Image Segmentation
**File:** `examples/02_vision/unet_segmentation.py`

Implement U-Net for semantic segmentation with SpiralTorchVision.

**Concepts:** Encoder-decoder architecture, skip connections, pixel-wise prediction

---

### 05: Transfer Learning
**File:** `examples/02_vision/transfer_learning.py`

Fine-tune a pre-trained model using DLPack to import PyTorch weights.

```python
import torch
import torchvision
import spiraltorch as st

# Load PyTorch pre-trained model
pt_model = torchvision.models.resnet18(pretrained=True)

# Extract feature extractor
features = torch.nn.Sequential(*list(pt_model.children())[:-1])

# Convert weights to SpiralTorch (zero-copy)
for name, param in features.named_parameters():
    st_param = st.Tensor.from_dlpack(param.data)
    # Store in SpiralTorch model...

# Add custom head in SpiralTorch for fine-tuning
model = st.nn.Sequential()
# ... add feature extractor ...
model.add(st.nn.Linear("head", 512, num_classes))
```

**Concepts:** Transfer learning, weight importing, zero-copy interop

---

## Natural Language Processing

### 06: Text Classification
**File:** `examples/03_nlp/text_classification.py`

Build a sentiment analyzer using Z-space embeddings for hierarchical text structure.

```python
from spiraltorch.nn import ZSpaceProjector, LanguageWaveEncoder, Linear
from spiraltorch.topos import OpenCartesianTopos

encoder = LanguageWaveEncoder(curvature=-0.9, temperature=0.7)
topos = OpenCartesianTopos(curvature=-0.9, tolerance=1e-6, 
                          saturation=1e4, max_volume=512, max_depth=16384)
projector = ZSpaceProjector(topos, encoder)

# Encode text into Z-space
texts = ["This movie is great!", "Terrible experience"]
embeddings = [projector.encode_text(text) for text in texts]

# Classification head
classifier = st.nn.Linear("classifier", embedding_dim, num_classes)
```

**Concepts:** Z-space embeddings, text encoding, hyperbolic geometry for hierarchies

---

### 07: Sequence-to-Sequence
**File:** `examples/03_nlp/seq2seq.py`

Implement an encoder-decoder model for machine translation.

**Concepts:** Recurrent layers, attention mechanisms, sequence generation

---

### 08: ZSpace Coherence Sequencer
**File:** `examples/03_nlp/coherence_sequencer.py`

Use the unique ZSpaceCoherenceSequencer as an alternative to Transformers.

```python
from spiraltorch.nn import ZSpaceCoherenceSequencer, CoherenceBackend

model = ZSpaceCoherenceSequencer(
    dim=768,
    num_heads=12,
    curvature=-1.0,
    topos=topos
)

# Enable pre-discard for efficiency
model.configure_pre_discard(
    dominance_ratio=0.35,
    energy_floor=1e-3,
    min_channels=3
)

# Forward pass with diagnostics
output, coherence, diagnostics = model.forward_with_diagnostics(x)
print(f"Dominant channel: {diagnostics.dominant_channel()}")
print(f"Discarded channels: {diagnostics.discarded_channels()}")
```

**Concepts:** Maxwell pulses, Desire Lagrangian, non-attention sequencing

---

## Reinforcement Learning

### 09: Multi-Armed Bandit
**File:** `examples/04_rl/multi_armed_bandit.py`

Solve the classic MAB problem with SpiralTorchRL agents.

```python
import spiraltorch as st

Agent = st.rl.stAgent
agent = Agent(state_dim=1, action_dim=2, discount=0.0, learning_rate=0.05)

for t in range(1000):
    eps = max(0.01, 0.3 * (1.0 - t/1000))
    agent.set_epsilon(eps)
    
    action = agent.select_action(0)
    reward = get_reward(action)  # Your reward function
    agent.update(0, action, reward, 0)

print(f"Final Q-values: {agent.get_q_values(0)}")
```

**Concepts:** Exploration-exploitation, Q-learning, epsilon-greedy

---

### 10: Policy Gradient with Geometry Feedback
**File:** `examples/04_rl/policy_gradient_cartpole.py`

Train a policy gradient agent with geometric feedback for learning rate adaptation.

```python
from spiraltorch.spiral_rl import PolicyGradient, GeometryFeedback
from spiraltorch import SpiralSession

session = SpiralSession(device="wgpu", curvature=-1.0)
policy = PolicyGradient(state_dim=4, action_dim=2, learning_rate=0.01)
policy.attach_geometry_feedback({"z_space_rank": 24})

for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action, probs = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        policy.record_transition(state, action, reward)
        state = next_state
    
    # Compute differential resonance for geometry feedback
    resonance = session.trace(state).resonate()
    report, signal = policy.finish_episode_with_geometry(resonance)
    
    if signal:
        print(f"Episode {episode}: η̄={signal['averaged_efficiency']:.3f}")
```

**Concepts:** Policy gradients, geometric feedback, observability metrics

---

## Graph Neural Networks

### 11: Node Classification
**File:** `examples/05_graphs/node_classification.py`

Classify nodes in a citation network using Z-space graph convolutions.

```python
from spiraltorch.nn.gnn import ZSpaceGraphConvolution, GraphContext

# Build graph context
context = GraphContext.from_adjacency_matrix(adj_matrix)

# Create GNN layers
gcn1 = ZSpaceGraphConvolution("gcn1", in_features=128, out_features=64)
gcn2 = ZSpaceGraphConvolution("gcn2", in_features=64, out_features=num_classes)

# Forward pass
h1 = gcn1.forward(features, context)
h1 = st.nn.relu(h1)
logits = gcn2.forward(h1, context)
```

**Concepts:** Graph convolutions, message passing, hyperbolic node embeddings

---

## Recommendation Systems

### 12: Collaborative Filtering
**File:** `examples/06_recsys/collaborative_filtering.py`

Build a matrix factorization recommender with topos guards.

```python
from spiraltorch.rec import Recommender

rec = Recommender(
    users=1000, 
    items=5000, 
    factors=64, 
    learning_rate=0.01, 
    regularization=0.001
)

# Training data: (user_id, item_id, rating)
train_data = [(0, 10, 5.0), (0, 25, 4.0), (1, 10, 3.0)]
epoch = rec.train_epoch(train_data)

# Get recommendations
top_k = rec.recommend_top_k(user_id=0, k=10)
print(f"Top 10 items for user 0: {top_k}")
```

**Concepts:** Matrix factorization, implicit feedback, topos regularization

---

## Time Series

### 13: Forecasting with Temporal Resonance
**File:** `examples/07_timeseries/forecasting.py`

Forecast time series using SpiralTorchVision's temporal buffers.

```python
from spiraltorch import SpiralTorchVision

vision = SpiralTorchVision(
    depth=4, 
    height=1, 
    width=seq_length,
    alpha=0.2,
    window="hann",
    temporal=4  # Temporal resonance buffer
)

for t, value in enumerate(time_series):
    vol = [[[value] * seq_length]]
    vision.accumulate(vol)
    
    if t > 10:
        forecast = vision.project()  # Project next value
```

**Concepts:** Temporal buffers, spectral analysis, forecasting

---

## Advanced Z-Space

### 14: Hyperbolic Cross-Entropy
**File:** `examples/08_advanced/hyperbolic_classification.py`

Use hyperbolic geometry for hierarchical classification.

```python
from spiraltorch.nn import HyperbolicCrossEntropy

# For hierarchical labels (e.g., taxonomy)
loss_fn = HyperbolicCrossEntropy(curvature=-0.95)

logits = model.forward(x)  # Shape: (batch, num_classes)
labels = get_hierarchical_labels()  # One-hot with hierarchy

loss = loss_fn.forward(logits, labels)
```

**Concepts:** Hyperbolic embeddings, hierarchical losses, curvature selection

---

### 15: Canvas Transformer Visualization
**File:** `examples/08_advanced/canvas_visualization.py`

Visualize training dynamics using Canvas Transformer.

```python
from spiraltorch import CanvasTransformer, CanvasProjector

canvas = CanvasTransformer(width=256, height=256, smoothing=0.85)
projector = CanvasProjector(canvas_width=256, canvas_height=256)

# During training
for epoch in range(100):
    stats = trainer.train_epoch(...)
    
    # Project training state to canvas
    relation = projector.refresh(gradient_tensor)
    
    # Export as image
    image_data = canvas.to_rgba()
    save_image(f"canvas_epoch_{epoch}.png", image_data)
```

**Concepts:** Canvas projection, gradient visualization, Z-space rendering

---

## Plugin Development

### 16: Custom Plugin
**File:** `examples/09_plugins/custom_plugin.py`

Create a custom SpiralTorch plugin for telemetry or custom operations.

```python
from spiraltorch.plugin import Plugin, PluginMetadata, PluginContext

class TelemetryPlugin:
    def metadata(self):
        return PluginMetadata("telemetry", "1.0.0") \
            .with_description("Custom telemetry collector") \
            .with_capability("monitoring")
    
    def on_load(self, ctx):
        print("TelemetryPlugin loaded")
        ctx.subscribe_event("training_step", self.on_training_step)
        return Ok(())
    
    def on_training_step(self, event):
        print(f"Step {event.step}: loss={event.loss}")

# Register plugin
from spiraltorch.plugin import global_registry, init_plugin_system

init_plugin_system()
global_registry().register(TelemetryPlugin())
```

**Concepts:** Plugin architecture, event bus, lifecycle hooks

---

## Performance Optimization

### 17: Backend Benchmarking
**File:** `examples/10_performance/backend_benchmark.py`

Compare performance across CPU, WGPU, and CUDA backends.

```python
import time
import spiraltorch as st
from spiraltorch import ModuleTrainer

backends = ["cpu", "wgpu", "cuda"]
results = {}

for backend in backends:
    try:
        trainer = ModuleTrainer(device=backend, ...)
        
        start = time.time()
        for epoch in range(10):
            trainer.train_epoch(...)
        elapsed = time.time() - start
        
        results[backend] = elapsed
        print(f"{backend}: {elapsed:.2f}s")
    except Exception as e:
        print(f"{backend}: Not available - {e}")

print(f"\nFastest: {min(results, key=results.get)}")
```

**Concepts:** Backend selection, performance profiling, multi-device support

---

### 18: SpiralK Heuristics Tuning
**File:** `examples/10_performance/spiralk_tuning.py`

Optimize kernel selection with SpiralK DSL.

```python
import os

# Enable soft heuristics
os.environ['SPIRAL_HEUR_SOFT'] = '1'
os.environ['SPIRAL_HEUR_K'] = '''
  # Merge kind: 0=bitonic, 1=shared, 2=warp
  mk: sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  
  # TopK tile size
  tile: sel(log2(c)>15.0, 2048, 
        sel(log2(c)>13.0, 1024,
        sel(log2(c)>12.0, 512, 256)));
  
  # Soft hints for probabilistic selection
  soft(mk, 2, 0.25, sg && (k<=128));
  soft(tile, 2048, 0.20, log2(c)>15.0);
'''

# Your training code will now use optimized kernels
```

**Concepts:** Kernel heuristics, DSL programming, performance tuning

---

## Running the Examples

### Python Examples

```bash
# Install SpiralTorch
pip install spiraltorch

# Run an example
python examples/01_basics/tensor_operations.py
```

### Rust Examples

```bash
# Build and run
cargo run --example tensor_operations

# Or build all examples
cargo build --examples --release
```

## Example Dependencies

Some examples require additional dependencies:

```bash
# For visualization examples
pip install matplotlib pillow

# For PyTorch interop examples
pip install torch torchvision

# For JAX interop examples
pip install jax jaxlib
```

## Contributing Examples

We welcome new examples! To contribute:

1. Create a well-documented example with clear comments
2. Add it to the appropriate category directory
3. Update this index with a description
4. Include any special dependencies in requirements
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

**Explore, learn, and build with SpiralTorch!** 🌀

For more help, see the [Getting Started Guide](getting-started.md) or join our [community discussions](https://github.com/RyoSpiralArchitect/SpiralTorch/discussions).
