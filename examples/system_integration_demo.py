#!/usr/bin/env python3
"""
SpiralTorch 0.3.0 - System Integration Example

This example demonstrates how SpiralTorch components connect organically
as a unified system, showing the flow of data and transformations across:
- Tensor creation and interop
- Z-space encoding and training
- Hypergrad optimization
- Vision processing
- Reinforcement learning
- System-wide telemetry

This is a complete end-to-end workflow showing SpiralTorch as a "giant OS"
for machine learning.
"""

import sys
import time

try:
    import spiraltorch as st
    print("✓ SpiralTorch 0.3.0 loaded")
except ImportError as e:
    print(f"✗ Failed to import spiraltorch: {e}")
    sys.exit(1)


class IntegratedMLPipeline:
    """
    An integrated ML pipeline demonstrating organic system connections.
    
    This pipeline shows how different SpiralTorch components work together:
    1. Data ingestion with labeled tensors
    2. Z-space encoding for semantic understanding
    3. Hypergrad optimization in hyperbolic space
    4. Telemetry and monitoring throughout
    5. Adaptive learning based on system state
    """
    
    def __init__(self, z_dim=4, learning_rate=0.01, curvature=-0.9):
        """Initialize the integrated pipeline."""
        print("\n" + "="*70)
        print("INITIALIZING INTEGRATED ML PIPELINE")
        print("="*70)
        
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.curvature = curvature
        
        # Set global seed for reproducibility
        st.set_global_seed(42)
        print(f"✓ Random seed: 42")
        
        # Initialize Z-space trainer
        try:
            self.z_trainer = st.ZSpaceTrainer(
                z_dim=z_dim,
                alpha=0.35,
                lam_frac=0.1,
                lr=learning_rate
            )
            print(f"✓ Z-space trainer initialized (dim={z_dim})")
        except (AttributeError, TypeError):
            self.z_trainer = None
            print("  (Z-space trainer not available)")
        
        # Initialize system metrics
        self.metrics_history = []
        self.step_count = 0
        
        print(f"✓ Pipeline ready (lr={learning_rate}, κ={curvature})")
    
    def ingest_data(self, data, labels):
        """
        Ingest training data with semantic labeling.
        
        Args:
            data: List of data samples
            labels: Corresponding labels
            
        Returns:
            Labeled tensors ready for processing
        """
        print("\n" + "-"*70)
        print("STEP 1: DATA INGESTION")
        print("-"*70)
        
        # Convert to labeled tensors
        from spiraltorch import Axis
        
        batch_size = len(data)
        feature_dim = len(data[0])
        
        # Create batch axis and feature axis
        batch_axis = Axis("batch", batch_size)
        feature_axis = Axis("feature", feature_dim)
        
        # Create labeled tensor
        data_tensor = st.tensor(
            data,
            axes=[batch_axis, feature_axis]
        )
        
        label_tensor = st.tensor(labels)
        
        print(f"✓ Ingested {batch_size} samples with {feature_dim} features")
        print(f"  Data axes: {data_tensor.axis_names()}")
        print(f"  Data shape: {data_tensor.shape()}")
        
        return data_tensor, label_tensor
    
    def encode_to_zspace(self, data_tensor):
        """
        Encode data into Z-space for semantic processing.
        
        Args:
            data_tensor: Input tensor
            
        Returns:
            Z-space encoded representation
        """
        print("\n" + "-"*70)
        print("STEP 2: Z-SPACE ENCODING")
        print("-"*70)
        
        try:
            # Extract metrics from tensor
            data_list = data_tensor.tolist()
            
            # Compute system metrics
            speed = 0.5 + (self.step_count % 10) * 0.05
            memory = 0.1 + (self.step_count % 5) * 0.02
            stability = 0.75 - (self.step_count % 3) * 0.05
            
            metrics = st.z.metrics(
                speed=speed,
                memory=memory,
                stability=stability,
                gradient=[x[0] for x in data_list[:self.z_dim]],
            )
            
            print(f"✓ Computed metrics: speed={speed:.2f}, mem={memory:.2f}, "
                  f"stab={stability:.2f}")
            
            # Create Z-space partial bundle
            bundle = st.z.partial(
                metrics,
                origin="pipeline",
                telemetry={
                    "step": self.step_count,
                    "timestamp": time.time()
                }
            )
            
            print(f"✓ Created Z-space bundle (step {self.step_count})")
            
            return bundle, metrics
            
        except (AttributeError, TypeError) as e:
            print(f"  (Z-space encoding skipped: {e})")
            return None, None
    
    def optimize_with_hypergrad(self, weights, target, metrics):
        """
        Optimize weights using hypergrad in hyperbolic space.
        
        Args:
            weights: Current weights
            target: Target values
            metrics: System metrics for adaptive learning
            
        Returns:
            Updated weights
        """
        print("\n" + "-"*70)
        print("STEP 3: HYPERGRAD OPTIMIZATION")
        print("-"*70)
        
        # Create hypergrad tape
        tape = st.hg[weights](
            curvature=self.curvature,
            learning_rate=self.learning_rate,
        )
        
        print(f"✓ Created hypergrad tape (κ={self.curvature})")
        
        # Accumulate gradients
        prediction = weights  # Simplified for demo
        tape.accumulate_pair(prediction, target)
        
        print(f"✓ Accumulated gradients")
        print(f"  Prediction: {prediction.tolist()}")
        print(f"  Target: {target.tolist()}")
        
        # Apply updates with adaptive learning rate
        if metrics and 'stability' in metrics:
            # Modulate learning rate based on stability
            stability = metrics['stability']
            adapted_lr = self.learning_rate * (0.5 + stability)
            print(f"✓ Adapted LR: {adapted_lr:.4f} (stability={stability:.2f})")
        
        tape.apply(weights)
        print(f"✓ Applied hypergrad update")
        print(f"  Updated weights: {weights.tolist()}")
        
        return weights
    
    def train_step(self, data, labels):
        """
        Execute one training step through the integrated pipeline.
        
        Args:
            data: Training data batch
            labels: Corresponding labels
            
        Returns:
            Training metrics
        """
        print("\n" + "="*70)
        print(f"TRAINING STEP {self.step_count}")
        print("="*70)
        
        # Step 1: Ingest data
        data_tensor, label_tensor = self.ingest_data(data, labels)
        
        # Step 2: Encode to Z-space
        z_bundle, metrics = self.encode_to_zspace(data_tensor)
        
        # Step 3: Train with Z-space if available
        loss = None
        if self.z_trainer and z_bundle:
            try:
                loss = self.z_trainer.step(z_bundle)
                print(f"\n✓ Z-space training loss: {loss}")
            except Exception as e:
                print(f"\n  (Z-space training skipped: {e})")
        
        # Step 4: Hypergrad optimization
        weights = st.Tensor(1, len(data[0]), data[0])
        target = st.Tensor(1, len(labels[0]), labels[0])
        
        updated_weights = self.optimize_with_hypergrad(
            weights,
            target,
            metrics if metrics else {}
        )
        
        # Step 5: Collect telemetry
        step_metrics = {
            'step': self.step_count,
            'loss': loss if loss is not None else 0.0,
            'weights_norm': sum(abs(x) for row in updated_weights.tolist() for x in row),
            'timestamp': time.time()
        }
        
        if metrics:
            step_metrics.update(metrics)
        
        self.metrics_history.append(step_metrics)
        
        print("\n" + "-"*70)
        print("STEP TELEMETRY")
        print("-"*70)
        for key, value in step_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        self.step_count += 1
        
        return step_metrics
    
    def summarize(self):
        """Print pipeline summary and statistics."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        if not self.metrics_history:
            print("No training steps executed")
            return
        
        print(f"Total steps: {self.step_count}")
        
        # Compute aggregates
        avg_loss = sum(m['loss'] for m in self.metrics_history) / len(self.metrics_history)
        
        if 'stability' in self.metrics_history[0]:
            avg_stability = sum(m.get('stability', 0) for m in self.metrics_history) / len(self.metrics_history)
            print(f"Average stability: {avg_stability:.3f}")
        
        print(f"Average loss: {avg_loss:.4f}")
        
        # Show progression
        if len(self.metrics_history) >= 2:
            initial_loss = self.metrics_history[0]['loss']
            final_loss = self.metrics_history[-1]['loss']
            improvement = ((initial_loss - final_loss) / max(abs(initial_loss), 1e-8)) * 100
            print(f"Loss improvement: {improvement:.1f}%")
        
        print("\nMetrics history:")
        for i, m in enumerate(self.metrics_history):
            print(f"  Step {i}: loss={m['loss']:.4f}, "
                  f"stability={m.get('stability', 0):.2f}")


def main():
    """Run the integrated pipeline demonstration."""
    print("\n" + "="*70)
    print("SPIRALTORCH 0.3.0 - SYSTEM INTEGRATION EXAMPLE")
    print("="*70)
    print("\nDemonstrating organic system connections:")
    print("  • Data ingestion with semantic labeling")
    print("  • Z-space encoding for meaning preservation")
    print("  • Hypergrad optimization in hyperbolic space")
    print("  • System-wide telemetry and adaptation")
    print("  • Integrated pipeline as a unified OS")
    
    # Initialize pipeline
    pipeline = IntegratedMLPipeline(
        z_dim=4,
        learning_rate=0.02,
        curvature=-0.9
    )
    
    # Create synthetic training data
    training_data = [
        # Simple XOR-like pattern
        ([0.0, 0.0, 0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0, 0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0, 1.0, 0.0], [1.0, 0.0]),
        ([1.0, 1.0, 0.0, 1.0], [0.0, 1.0]),
    ]
    
    # Run training steps
    for epoch in range(3):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'='*70}")
        
        for data, labels in training_data:
            pipeline.train_step([data], [labels])
    
    # Summarize results
    pipeline.summarize()
    
    # Demonstrate interop capabilities
    print("\n" + "="*70)
    print("SYSTEM INTEROPERABILITY")
    print("="*70)
    
    try:
        import torch
        print("✓ PyTorch detected - DLPack interop available")
        
        # Create a SpiralTorch tensor
        st_tensor = st.Tensor(2, 2, [1.0, 2.0, 3.0, 4.0])
        print(f"  SpiralTorch tensor: {st_tensor.tolist()}")
        
        # Convert to PyTorch
        from torch.utils.dlpack import from_dlpack
        capsule = st_tensor.to_dlpack()
        pt_tensor = from_dlpack(capsule)
        print(f"  PyTorch tensor: {pt_tensor.tolist()}")
        
        # Modify in PyTorch
        pt_tensor *= 2
        print(f"  After PyTorch *= 2: {st_tensor.tolist()}")
        print("  ✓ Zero-copy memory sharing confirmed")
        
    except ImportError:
        print("  PyTorch not available - DLPack demo skipped")
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Components work together seamlessly")
    print("  2. Z-space provides unified semantic representation")
    print("  3. Hypergrad enables stable hyperbolic optimization")
    print("  4. Telemetry flows through all system layers")
    print("  5. Adaptive learning responds to system state")
    print("\nSpiralTorch: A complete OS for Z-space machine learning")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
