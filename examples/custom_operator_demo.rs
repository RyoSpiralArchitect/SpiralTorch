// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Example demonstrating custom operator registration.
//!
//! Shows how to extend SpiralTorch with domain-specific tensor operations
//! using the pluggable operator registry.

use st_core::ops::{global_operator_registry, OperatorBuilder};
use st_core::PureResult;
use st_tensor::Tensor;
use std::sync::Arc;

fn main() -> PureResult<()> {
    println!("🌀 Custom Operator Registration Example 🌀\n");

    let registry = global_operator_registry();

    // Register a custom element-wise square operator
    println!("📝 Registering custom operators...\n");

    let square_op = OperatorBuilder::new("square", 1, 1)
        .with_description("Element-wise square: y = x^2")
        .with_backend("CPU")
        .with_backend("WGPU")
        .with_backend("CUDA")
        .with_differentiable(true)
        .with_attribute("operation_type", "unary")
        .with_forward(Arc::new(|inputs| {
            println!("  🔧 Executing square operator");
            let x = inputs[0];
            let (rows, cols) = x.shape();
            
            let data: Vec<f32> = x
                .data()
                .iter()
                .map(|&v| v * v)
                .collect();
            
            Ok(vec![Tensor::from_vec(rows, cols, data)?])
        }))
        .with_backward(Arc::new(|inputs, _outputs, grad_outputs| {
            println!("  🔄 Computing square gradient");
            let x = inputs[0];
            let grad_y = grad_outputs[0];
            let (rows, cols) = x.shape();
            
            // Gradient: dy/dx = 2x
            let grad_data: Vec<f32> = x
                .data()
                .iter()
                .zip(grad_y.data().iter())
                .map(|(&x_val, &grad_val)| 2.0 * x_val * grad_val)
                .collect();
            
            Ok(vec![Tensor::from_vec(rows, cols, grad_data)?])
        }))
        .build()?;

    registry.register(square_op)?;
    println!("  ✅ Registered 'square' operator\n");

    // Register a custom normalization operator
    let normalize_op = OperatorBuilder::new("normalize", 1, 1)
        .with_description("L2 normalization: y = x / ||x||_2")
        .with_backend("CPU")
        .with_differentiable(true)
        .with_attribute("operation_type", "normalization")
        .with_forward(Arc::new(|inputs| {
            println!("  🔧 Executing normalize operator");
            let x = inputs[0];
            let (rows, cols) = x.shape();
            
            // Compute L2 norm
            let norm: f32 = x.data().iter().map(|&v| v * v).sum::<f32>().sqrt();
            let norm = norm.max(1e-8); // Avoid division by zero
            
            let data: Vec<f32> = x.data().iter().map(|&v| v / norm).collect();
            
            Ok(vec![Tensor::from_vec(rows, cols, data)?])
        }))
        .with_backward(Arc::new(|inputs, outputs, grad_outputs| {
            println!("  🔄 Computing normalize gradient");
            let x = inputs[0];
            let y = outputs[0];
            let grad_y = grad_outputs[0];
            let (rows, cols) = x.shape();
            
            // Gradient computation (simplified)
            let norm: f32 = x.data().iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
            
            let grad_data: Vec<f32> = x
                .data()
                .iter()
                .zip(grad_y.data().iter())
                .zip(y.data().iter())
                .map(|((&x_val, &grad_val), &_y_val)| {
                    (grad_val - x_val * grad_val) / norm
                })
                .collect();
            
            Ok(vec![Tensor::from_vec(rows, cols, grad_data)?])
        }))
        .build()?;

    registry.register(normalize_op)?;
    println!("  ✅ Registered 'normalize' operator\n");

    // Register a custom binary operator: weighted sum
    let weighted_sum_op = OperatorBuilder::new("weighted_sum", 2, 1)
        .with_description("Weighted sum: y = a * x1 + (1-a) * x2")
        .with_backend("CPU")
        .with_differentiable(true)
        .with_attribute("operation_type", "binary")
        .with_forward(Arc::new(|inputs| {
            println!("  🔧 Executing weighted_sum operator");
            let x1 = inputs[0];
            let x2 = inputs[1];
            let (rows, cols) = x1.shape();
            
            // For simplicity, use a = 0.7
            let a = 0.7f32;
            let data: Vec<f32> = x1
                .data()
                .iter()
                .zip(x2.data().iter())
                .map(|(&v1, &v2)| a * v1 + (1.0 - a) * v2)
                .collect();
            
            Ok(vec![Tensor::from_vec(rows, cols, data)?])
        }))
        .with_backward(Arc::new(|_inputs, _outputs, grad_outputs| {
            println!("  🔄 Computing weighted_sum gradient");
            let grad_y = grad_outputs[0];
            let (rows, cols) = grad_y.shape();
            
            let a = 0.7f32;
            
            // Gradient w.r.t. x1: a * grad_y
            let grad_x1: Vec<f32> = grad_y.data().iter().map(|&g| a * g).collect();
            
            // Gradient w.r.t. x2: (1-a) * grad_y
            let grad_x2: Vec<f32> = grad_y.data().iter().map(|&g| (1.0 - a) * g).collect();
            
            Ok(vec![
                Tensor::from_vec(rows, cols, grad_x1)?,
                Tensor::from_vec(rows, cols, grad_x2)?,
            ])
        }))
        .build()?;

    registry.register(weighted_sum_op)?;
    println!("  ✅ Registered 'weighted_sum' operator\n");

    // List registered operators
    println!("📋 Registered operators:");
    for op_name in registry.list_operators() {
        if let Some(op) = registry.get(&op_name) {
            let meta = op.metadata();
            println!("  - {} ({})", op_name, meta.description);
            println!("    Inputs: {}, Outputs: {}", 
                     meta.signature.num_inputs, 
                     meta.signature.num_outputs);
            println!("    Backends: {}", meta.backends.join(", "));
            println!("    Differentiable: {}", meta.signature.differentiable);
        }
    }
    println!();

    // Test the operators
    println!("🧪 Testing custom operators...\n");

    // Create test tensors
    let x = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])?;
    println!("Input tensor x:");
    for (i, row) in x.tolist().iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Test square operator
    let y_square = registry.execute("square", &[&x])?;
    println!("After square:");
    for (i, row) in y_square[0].tolist().iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Test normalize operator
    let y_norm = registry.execute("normalize", &[&x])?;
    println!("After normalize:");
    for (i, row) in y_norm[0].tolist().iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Test weighted sum operator
    let x2 = Tensor::from_vec(2, 2, vec![4.0, 3.0, 2.0, 1.0])?;
    let y_weighted = registry.execute("weighted_sum", &[&x, &x2])?;
    println!("After weighted_sum:");
    for (i, row) in y_weighted[0].tolist().iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    println!();

    // Test gradient computation
    println!("🔬 Testing gradient computation...\n");
    
    let grad_output = Tensor::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0])?;
    
    if let Some(square_op) = registry.get("square") {
        let grad_input = square_op.backward(&[&x], &y_square, &[&grad_output])?;
        println!("Square gradient w.r.t. input:");
        for (i, row) in grad_input[0].tolist().iter().enumerate() {
            println!("  Row {}: {:?}", i, row);
        }
        println!();
    }

    // Find operators by backend
    println!("🔍 Finding operators by backend...\n");
    
    let cuda_ops = registry.find_by_backend("CUDA");
    println!("CUDA-enabled operators:");
    for op in cuda_ops {
        println!("  - {}", op.metadata().signature.name);
    }
    println!();

    let cpu_ops = registry.find_by_backend("CPU");
    println!("CPU-enabled operators:");
    for op in cpu_ops {
        println!("  - {}", op.metadata().signature.name);
    }
    println!();

    println!("✨ Custom operator example completed successfully!\n");
    println!("Key features demonstrated:");
    println!("  ✓ Custom operator registration");
    println!("  ✓ Forward and backward implementations");
    println!("  ✓ Multi-backend support");
    println!("  ✓ Operator discovery and querying");
    println!("  ✓ Gradient computation\n");

    Ok(())
}
