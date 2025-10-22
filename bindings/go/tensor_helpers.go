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

// NewTensorFromColumns constructs a tensor from a slice of equally sized column
// vectors. The columns are interpreted in column-major order and converted to
// the row-major layout expected by SpiralTorch. Empty column sets are
// supported and will yield zero-sized tensors.
func NewTensorFromColumns(columns [][]float32) (*Tensor, error) {
	if columns == nil {
		return nil, fmt.Errorf("spiraltorch: columns cannot be nil")
	}
	cols := len(columns)
	if cols == 0 {
		return NewZerosTensor(0, 0)
	}

	rows := len(columns[0])
	for i := 1; i < cols; i++ {
		if len(columns[i]) != rows {
			return nil, fmt.Errorf("spiraltorch: column %d has length %d, expected %d", i, len(columns[i]), rows)
		}
	}

	if rows == 0 {
		return NewZerosTensor(0, cols)
	}

	data := make([]float32, rows*cols)
	for c := 0; c < cols; c++ {
		column := columns[c]
		for r := 0; r < rows; r++ {
			data[r*cols+c] = column[r]
		}
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

// Columns returns a slice of column copies from the tensor. Each column is an
// independent slice that can be mutated without affecting the underlying
// tensor or other columns.
func (t *Tensor) Columns() ([][]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}

	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	result := make([][]float32, cols)
	if rows == 0 || cols == 0 {
		for i := range result {
			result[i] = make([]float32, rows)
		}
		return result, nil
	}

	flat, err := t.Data()
	if err != nil {
		return nil, err
	}

	for c := 0; c < cols; c++ {
		column := make([]float32, rows)
		for r := 0; r < rows; r++ {
			column[r] = flat[r*cols+c]
		}
		result[c] = column
	}
	return result, nil
}

// Row returns a copy of the requested row. The returned slice owns its backing
// array and may be mutated freely.
func (t *Tensor) Row(index int) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}

	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= rows {
		return nil, fmt.Errorf("spiraltorch: row index %d out of range [0,%d)", index, rows)
	}
	if cols == 0 {
		return make([]float32, 0), nil
	}

	flat, err := t.Data()
	if err != nil {
		return nil, err
	}

	start := index * cols
	row := make([]float32, cols)
	copy(row, flat[start:start+cols])
	return row, nil
}

// Column returns a copy of the requested column. The returned slice owns its
// backing array and may be mutated freely.
func (t *Tensor) Column(index int) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}

	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	if index < 0 || index >= cols {
		return nil, fmt.Errorf("spiraltorch: column index %d out of range [0,%d)", index, cols)
	}
	if rows == 0 {
		return make([]float32, 0), nil
	}

	flat, err := t.Data()
	if err != nil {
		return nil, err
	}

	column := make([]float32, rows)
	for r := 0; r < rows; r++ {
		column[r] = flat[r*cols+index]
	}
	return column, nil
}
