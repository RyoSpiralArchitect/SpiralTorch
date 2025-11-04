#!/usr/bin/env python3
"""
SpiralTorch 0.3.0 - Comprehensive Python Bindings Showcase

This example demonstrates the rich, intuitive Python API for SpiralTorch's
Z-space runtime. The bindings provide seamless access to:
- Tensor operations with DLPack interop
- Hypergrad tapes for hyperbolic optimization
- Z-space encoding and training
- Reinforcement learning
- Self-supervised learning
- Vision and canvas transformers
- Recommendation systems
- Safety-aware inference

Run this to verify your SpiralTorch installation and explore the API.
"""

import sys
try:
    import spiraltorch as st
    print(f"✓ SpiralTorch imported successfully (version 0.3.0)")
except ImportError as e:
    print(f"✗ Failed to import spiraltorch: {e}")
    print("  Install with: pip install spiraltorch==0.3.0")
    sys.exit(1)


def demo_tensor_basics():
    """Demonstrate basic tensor creation and operations."""
    print("\n" + "="*70)
    print("1. TENSOR CREATION AND BASIC OPERATIONS")
    print("="*70)
    
    # Create tensors from Python lists
    x = st.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = st.Tensor(2, 3)  # Zero-initialized
    
    print(f"Created tensor x from list: shape={x.shape()}, data={x.tolist()}")
    print(f"Created zero tensor y: shape={y.shape()}, data={y.tolist()}")
    
    # Labeled tensors for semantic dimensions
    labeled = st.tensor(
        [[0.2, 0.8], [0.4, 0.6]],
        axes=[st.Axis("batch", 2), st.Axis("feature", 2)],
    )
    print(f"Labeled tensor axes: {labeled.axis_names()}")
    
    # Basic operations
    z = x.scale(2.0)
    print(f"Scaled tensor: {z.tolist()}")
    

def demo_dlpack_interop():
    """Demonstrate zero-copy interop with PyTorch via DLPack."""
    print("\n" + "="*70)
    print("2. ZERO-COPY DLPACK INTEROPERABILITY")
    print("="*70)
    
    try:
        import torch
        from torch.utils.dlpack import from_dlpack as torch_from_dlpack
        
        # SpiralTorch → PyTorch (zero-copy)
        st_tensor = st.Tensor(2, 3, [1, 2, 3, 4, 5, 6])
        print(f"Original SpiralTorch tensor: {st_tensor.tolist()}")
        
        capsule = st_tensor.to_dlpack()
        torch_tensor = torch_from_dlpack(capsule)
        print(f"Converted to PyTorch: {torch_tensor.tolist()}")
        
        # Mutations are visible in both (shared memory)
        torch_tensor += 10
        print(f"After PyTorch += 10: SpiralTorch sees {st_tensor.tolist()}")
        
        # PyTorch → SpiralTorch
        pt_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        st_from_torch = st.Tensor.from_dlpack(pt_tensor)
        pt_tensor.mul_(2)
        print(f"After PyTorch mul_: SpiralTorch sees {st_from_torch.tolist()}")
        
    except ImportError:
        print("  (Skipped: PyTorch not installed)")


def demo_hypergrad_tapes():
    """Demonstrate hypergrad tapes for Z-space optimization."""
    print("\n" + "="*70)
    print("3. HYPERGRAD TAPES FOR Z-SPACE OPTIMIZATION")
    print("="*70)
    
    # Initialize weights and create hypergrad tape
    weights = st.Tensor(1, 3, [0.1, 0.2, 0.3])
    tape = st.hg[weights](
        curvature=-0.9,  # Hyperbolic curvature
        learning_rate=0.02,
    )
    
    print(f"Created hypergrad tape: shape={tape.shape()}, lr={tape.learning_rate()}")
    
    # Add topological guards for stability
    guarded = st.hg[weights].with_topos(
        tolerance=1e-3,
        saturation=0.8,
        max_depth=8
    )
    print(f"Guarded tape: curvature={guarded.curvature()}")
    
    # Accumulate gradients
    prediction = st.Tensor(1, 3, [0.25, 0.25, 0.25])
    target = st.Tensor(1, 3, [0.0, 1.0, 0.0])
    tape.accumulate_pair(prediction, target)
    
    # Apply updates
    print(f"Weights before update: {weights.tolist()}")
    tape.apply(weights)
    print(f"Weights after hypergrad update: {weights.tolist()}")


def demo_hypergrad_sessions():
    """Demonstrate advanced hypergrad sessions with operator hints."""
    print("\n" + "="*70)
    print("4. ADVANCED HYPERGRAD SESSIONS")
    print("="*70)
    
    try:
        from spiral.hypergrad import (
            hypergrad_session,
            hypergrad_summary_dict,
            suggest_hypergrad_operator
        )
        
        weights = st.Tensor(1, 3, [0.05, -0.15, 0.25])
        targets = st.Tensor(1, 3, [0.0, 1.0, 0.0])
        
        # Context manager for automatic cleanup
        with hypergrad_session(weights, learning_rate=0.03, curvature=-0.85) as tape:
            tape.accumulate_pair(weights, targets)
            
            # Get detailed gradient metrics
            metrics = hypergrad_summary_dict(tape, include_gradient=True)
            
            # Get WGSL operator hints for GPU optimization
            hints = suggest_hypergrad_operator(metrics)
            
            print(f"Gradient stats: {metrics['summary']}")
            print(f"WGSL operator hints: {hints}")
            if 'gradient' in metrics:
                print(f"Gradient sample: {metrics['gradient'][:3]}")
                
    except (ImportError, AttributeError) as e:
        print(f"  (Skipped: {e})")


def demo_zspace_encoding():
    """Demonstrate Z-space encoding and metric normalization."""
    print("\n" + "="*70)
    print("5. Z-SPACE ENCODING AND TRAINING")
    print("="*70)
    
    try:
        # Encode text into Z-space vectors
        z_vec = st.z["Initialize the neural manifold", 0.4]
        print(f"Z-space vector shape: {z_vec.shape()}")
        
        # Create normalized metrics for Z-space
        metrics = st.z.metrics(
            speed=0.55,
            memory=0.12,
            stability=0.78,
            drs=0.05,
            gradient=[0.1, -0.2, 0.05],
        )
        
        # Build partial bundles with telemetry
        roundtable = st.z.partial(
            metrics,
            origin="telemetry",
            telemetry={"roundtable": {"mean": 0.44, "focus": 0.67}},
        )
        
        canvas_hint = st.z.partial(
            speed=0.35,
            memory=0.22,
            coherence_peak=0.61,
            weight=0.5
        )
        
        # Blend and train
        bundle = st.z.bundle(roundtable, canvas_hint)
        trainer = st.ZSpaceTrainer(z_dim=z_vec.shape()[1])
        loss = trainer.step(bundle)
        
        print(f"Training loss: {loss}")
        
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_gpu_softmax():
    """Demonstrate GPU-accelerated row softmax with labeled axes."""
    print("\n" + "="*70)
    print("6. GPU-ACCELERATED ROW SOFTMAX")
    print("="*70)
    
    try:
        from spiraltorch import Axis, tensor
        
        # Define semantic axes
        time = Axis("time")
        feature = Axis("feature", 4)
        
        # Create labeled tensor
        wave = tensor(
            [
                [0.20, 0.80, -0.10, 0.40],
                [0.90, -0.30, 0.10, 0.50],
            ],
            axes=[time.with_size(2), feature],
        )
        
        print(f"Input: {wave.describe()}")
        
        # Softmax dispatches to best backend (WGPU/MPS/CPU)
        softmax = wave.row_softmax()
        print(f"Softmax axis names: {softmax.axis_names()}")
        print(f"Softmax values: {softmax.tolist()}")
        
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_reinforcement_learning():
    """Demonstrate multi-armed bandit with reinforcement learning."""
    print("\n" + "="*70)
    print("7. REINFORCEMENT LEARNING (MULTI-ARMED BANDIT)")
    print("="*70)
    
    try:
        import random
        
        # Check if RL agent is available
        Agent = getattr(st.rl, "stAgent", None)
        if Agent is None:
            print("  (Skipped: RL module not available)")
            return
        
        # Define reward function (arm 0 has 60% win rate, arm 1 has 40%)
        def reward(action):
            p = 0.6 if action == 0 else 0.4
            return 1.0 if random.random() < p else 0.0
        
        # Create agent
        agent = Agent(state_dim=1, action_dim=2, discount=0.0, learning_rate=5e-2)
        
        # Training parameters (abbreviated for demo)
        T = 500
        FORCE_EXPLORE = 50
        eps_hi, eps_lo = 0.3, 0.01
        
        wins = 0
        pulls = [0, 0]
        wins_by_arm = [0, 0]
        
        # Training loop
        for t in range(1, T + 1):
            if t <= FORCE_EXPLORE:
                a = t % 2
            else:
                frac = (t - FORCE_EXPLORE) / (T - FORCE_EXPLORE)
                eps = eps_hi + (eps_lo - eps_hi) * frac
                agent.set_epsilon(eps)
                a = agent.select_action(0)
            
            r = reward(a)
            wins += r
            pulls[a] += 1
            wins_by_arm[a] += r
            agent.update(0, a, r, 0)
        
        print(f"Total win rate: {wins / T:.3f}")
        for k in range(2):
            rate = (wins_by_arm[k] / pulls[k]) if pulls[k] else 0.0
            print(f"  Arm {k}: pulls={pulls[k]}, empirical win rate≈{rate:.3f}")
            
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_selfsupervised():
    """Demonstrate self-supervised learning losses."""
    print("\n" + "="*70)
    print("8. SELF-SUPERVISED LEARNING")
    print("="*70)
    
    try:
        # InfoNCE contrastive loss
        anchors = [[0.1, 0.9], [0.8, 0.2]]
        positives = [[0.12, 0.88], [0.79, 0.21]]
        loss = st.selfsup.info_nce(
            anchors,
            positives,
            temperature=0.1,
            normalize=True
        )
        print(f"InfoNCE loss: {loss}")
        
        # Masked MSE loss
        pred = [[0.2, 0.8], [0.6, 0.4]]
        tgt = [[0.0, 1.0], [1.0, 0.0]]
        mask = [[1], [0]]
        masked_loss = st.selfsup.masked_mse(pred, tgt, mask)
        print(f"Masked MSE: {masked_loss}")
        
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_recommender():
    """Demonstrate recommendation system."""
    print("\n" + "="*70)
    print("9. RECOMMENDATION SYSTEM")
    print("="*70)
    
    try:
        # Create recommender with matrix factorization
        rec = st.Recommender(
            users=8,
            items=12,
            factors=4,
            learning_rate=0.05,
            regularization=0.002
        )
        
        # Train on interaction data (user_id, item_id, rating)
        rec.train_epoch([
            (0, 0, 5.0),
            (0, 1, 3.0),
            (1, 0, 4.0),
            (0, 2, 4.5),
        ])
        
        # Get top-k recommendations
        recommendations = rec.recommend_top_k(0, k=3)
        print(f"Top 3 items for user 0: {recommendations}")
        
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_math_utilities():
    """Demonstrate mathematical helpers and utilities."""
    print("\n" + "="*70)
    print("10. MATHEMATICAL UTILITIES")
    print("="*70)
    
    try:
        # Set global random seed for reproducibility
        st.set_global_seed(42)
        print("Global seed set to 42")
        
        # Mathematical constants
        print(f"Golden ratio: {st.golden_ratio()}")
        print(f"Golden angle: {st.golden_angle()}")
        
        # Fibonacci-based pacing
        fib_pacing = st.fibonacci_pacing(12)
        print(f"Fibonacci pacing(12): {fib_pacing}")
        
        # Tribonacci chunking
        trib_chunks = st.pack_tribonacci_chunks(20)
        print(f"Tribonacci chunks(20): {trib_chunks}")
        
    except (AttributeError, TypeError) as e:
        print(f"  (Skipped: {e})")


def demo_safety_inference():
    """Demonstrate safety-aware generation with chat notation."""
    print("\n" + "="*70)
    print("11. SAFETY-AWARE INFERENCE")
    print("="*70)
    
    try:
        from spiral.inference import (
            ChatMessage,
            ChatPrompt,
            InferenceClient,
        )
        
        # Create inference client with safety thresholds
        client = InferenceClient(refusal_threshold=0.65)
        
        # Build chat prompt
        messages = ChatPrompt.from_messages([
            ChatMessage.system("You are SpiralTorch's safety-tuned narrator."),
            ChatMessage.user("Explain Z-space in two sentences."),
        ])
        
        # Generate with safety checks
        result = client.chat(
            messages,
            candidate="Z-space is SpiralTorch's hyperbolic manifold for ML."
        )
        
        if result.accepted:
            print(f"✓ Response accepted: {result.response}")
        else:
            print(f"✗ Response refused: {result.refusal_message}")
        
        # Review audit events
        events = list(client.audit_events())
        if events:
            print(f"Audit events: {len(events)} recorded")
            for event in events[:3]:  # Show first 3
                print(f"  [{event.timestamp}] {event.channel} → "
                      f"{event.verdict.dominant_risk} "
                      f"(score={event.verdict.score:.2f})")
                      
    except (ImportError, AttributeError) as e:
        print(f"  (Skipped: {e})")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("SPIRALTORCH 0.3.0 - PYTHON BINDINGS SHOWCASE")
    print("="*70)
    print("\nThis showcase demonstrates the rich, intuitive Python API")
    print("for SpiralTorch's Z-space runtime.")
    
    demos = [
        demo_tensor_basics,
        demo_dlpack_interop,
        demo_hypergrad_tapes,
        demo_hypergrad_sessions,
        demo_zspace_encoding,
        demo_gpu_softmax,
        demo_reinforcement_learning,
        demo_selfsupervised,
        demo_recommender,
        demo_math_utilities,
        demo_safety_inference,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SHOWCASE COMPLETE")
    print("="*70)
    print("\nFor more examples, see:")
    print("  - README.md (Python quickstart section)")
    print("  - examples/ directory (Rust examples)")
    print("  - docs/ directory (detailed documentation)")
    print("\nSpiralTorch 0.3.0 - Training where PyTorch can't, inside the Z-space.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
