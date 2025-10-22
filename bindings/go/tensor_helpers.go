// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

import "fmt"

// NewTensorFromMatrix constructs a tensor from a rectangular matrix represented
// as a slice of rows. The matrix must be rectangular (all rows have the same
// length). Empty matrices are supported and will yield tensors with either zero
// rows, zero columns, or both.
func NewTensorFromMatrix(matrix [][]float32) (*Tensor, error) {
	if matrix == nil {
		return nil, fmt.Errorf("spiraltorch: matrix cannot be nil")
	}
	rows := len(matrix)
	if rows == 0 {
		return NewZerosTensor(0, 0)
	}

	cols := len(matrix[0])
	for i := 1; i < rows; i++ {
		if len(matrix[i]) != cols {
			return nil, fmt.Errorf("spiraltorch: matrix row %d has length %d, expected %d", i, len(matrix[i]), cols)
		}
	}

	if cols == 0 {
		return NewZerosTensor(rows, 0)
	}

	data := make([]float32, 0, rows*cols)
	for _, row := range matrix {
		data = append(data, row...)
	}
	return NewTensorFromDense(rows, cols, data)
}

// ToMatrix decodes the tensor into a dense row-major matrix. Each row in the
// returned slice owns its backing array, making it safe to mutate without
// affecting other rows.
func (t *Tensor) ToMatrix() ([][]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}

	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	if rows == 0 || cols == 0 {
		return make([][]float32, rows), nil
	}

	flat, err := t.Data()
	if err != nil {
		return nil, err
	}

	matrix := make([][]float32, rows)
	for r := 0; r < rows; r++ {
		start := r * cols
		row := make([]float32, cols)
		copy(row, flat[start:start+cols])
		matrix[r] = row
	}
	return matrix, nil
}
