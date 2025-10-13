//! Pure Rust tensor utilities and linear model trainer with zero external dependencies.
//!
//! The goal of this module is to offer a pragmatic starting point for training
//! routines that **do not rely on PyTorch, NumPy, or any other native bindings**.
//! Everything here is written in safe Rust so it can serve as a foundation for a
//! fully independent learning stack.  The implementation intentionally keeps the
//! API small and explicit to make the data-flow obvious when extending the
//! functionality.

/// A simple row-major 2D tensor backed by a `Vec<f32>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a tensor from raw data. The provided vector must match
    /// `rows * cols` elements.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(rows * cols, data.len(), "data length must match shape");
        Self { data, rows, cols }
    }

    /// Returns the `(rows, cols)` pair of the tensor.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns a read-only view of the underlying buffer.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable view of the underlying buffer.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Matrix multiply (`self @ other`).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows, "incompatible shapes for matmul");
        let mut out = Tensor::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[r * self.cols + k];
                let row_offset = k * other.cols;
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.data[row_offset + c];
                }
            }
        }
        out
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "shape mismatch for add");
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a + b);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(), "shape mismatch for sub");
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            data.push(a - b);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Returns a new tensor where every element is scaled by `value`.
    pub fn scale(&self, value: f32) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());
        for a in &self.data {
            data.push(a * value);
        }
        Tensor::from_vec(self.rows, self.cols, data)
    }

    /// Add a scaled tensor to this tensor (`self += scale * other`).
    pub fn add_scaled(&mut self, other: &Tensor, scale: f32) {
        assert_eq!(self.shape(), other.shape(), "shape mismatch for add_scaled");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += scale * b;
        }
    }

    /// Add the provided row vector to every row (`self[row] += bias`).
    pub fn add_row_inplace(&mut self, bias: &[f32]) {
        assert_eq!(bias.len(), self.cols, "bias must match number of columns");
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                self.data[offset + c] += bias[c];
            }
        }
    }

    /// Returns the transpose of the tensor.
    pub fn transpose(&self) -> Tensor {
        let mut out = Tensor::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// Returns the sum over rows for each column.
    pub fn sum_axis0(&self) -> Vec<f32> {
        let mut sums = vec![0.0; self.cols];
        for r in 0..self.rows {
            let offset = r * self.cols;
            for c in 0..self.cols {
                sums[c] += self.data[offset + c];
            }
        }
        sums
    }
}

/// Computes the mean squared error between `predictions` and `targets`.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> f32 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "shape mismatch for mse"
    );
    let mut sum = 0.0f32;
    for (p, t) in predictions.data().iter().zip(targets.data().iter()) {
        let diff = p - t;
        sum += diff * diff;
    }
    sum / (predictions.rows * predictions.cols) as f32
}

/// A minimal fully-connected linear model.
///
/// The model keeps its weights and bias in plain Rust vectors so it can be
/// embedded in `no_std` friendly environments (alloc-only) and easily extended
/// with additional layers.
#[derive(Clone, Debug)]
pub struct LinearModel {
    weights: Tensor,
    bias: Vec<f32>,
}

impl LinearModel {
    /// Creates a new linear model with small deterministic parameters.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert!(
            input_dim > 0 && output_dim > 0,
            "dimensions must be non-zero"
        );
        let mut weights = Tensor::zeros(input_dim, output_dim);
        let mut scale = 0.01f32;
        for w in weights.data_mut().iter_mut() {
            *w = scale;
            scale += 0.01;
        }
        Self {
            weights,
            bias: vec![0.0; output_dim],
        }
    }

    /// Runs a forward pass: `inputs @ weights + bias`.
    pub fn forward(&self, inputs: &Tensor) -> Tensor {
        assert_eq!(
            inputs.shape().1,
            self.weights.shape().0,
            "input dim mismatch"
        );
        let mut out = inputs.matmul(&self.weights);
        out.add_row_inplace(&self.bias);
        out
    }

    /// Performs a single batch of gradient descent and returns the batch loss.
    pub fn train_batch(&mut self, inputs: &Tensor, targets: &Tensor, learning_rate: f32) -> f32 {
        assert_eq!(inputs.shape().0, targets.shape().0, "batch size mismatch");
        assert_eq!(
            targets.shape().1,
            self.weights.shape().1,
            "output dim mismatch"
        );
        let batch_size = inputs.shape().0 as f32;
        let predictions = self.forward(inputs);
        let diff = predictions.sub(targets);
        let inputs_t = inputs.transpose();
        let grad_w = inputs_t.matmul(&diff).scale(1.0 / batch_size);
        let mut grad_b = diff.sum_axis0();
        for val in grad_b.iter_mut() {
            *val /= batch_size;
        }
        self.weights.add_scaled(&grad_w, -learning_rate);
        for (b, g) in self.bias.iter_mut().zip(grad_b.iter()) {
            *b -= learning_rate * g;
        }
        mean_squared_error_from_diff(&diff)
    }

    /// Returns a reference to the model weights.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a reference to the model bias.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

fn mean_squared_error_from_diff(diff: &Tensor) -> f32 {
    let mut sum = 0.0f32;
    for v in diff.data() {
        sum += v * v;
    }
    sum / (diff.rows * diff.cols) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_matmul_and_add_work() {
        let a = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let product = a.matmul(&b);
        assert_eq!(product.shape(), (2, 2));
        let expected = Tensor::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(product, expected);

        let sum = product.add(&Tensor::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]));
        let expected_sum = Tensor::from_vec(2, 2, vec![59.0, 65.0, 140.0, 155.0]);
        assert_eq!(sum, expected_sum);
    }

    #[test]
    fn linear_regression_converges() {
        let inputs = Tensor::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]);
        let targets = Tensor::from_vec(4, 1, vec![1.0, 3.0, 5.0, 7.0]);
        let mut model = LinearModel::new(1, 1);
        let mut loss = 0.0;
        for _ in 0..200 {
            loss = model.train_batch(&inputs, &targets, 0.1);
        }
        assert!(loss < 1e-3, "loss should converge, got {loss}");

        let predictions = model.forward(&inputs);
        let mse = mean_squared_error(&predictions, &targets);
        assert!(mse < 1e-3, "model should fit the line, got {mse}");

        let weight = model.weights().data()[0];
        let bias = model.bias()[0];
        assert!((weight - 2.0).abs() < 1e-2, "weight too far: {weight}");
        assert!((bias - 1.0).abs() < 1e-2, "bias too far: {bias}");
    }
}
