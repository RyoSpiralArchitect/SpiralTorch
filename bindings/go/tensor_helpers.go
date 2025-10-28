// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

import "fmt"

// NewTensorFromMatrix constructs a tensor from a rectangular matrix represented
// as a slice of rows. The matrix must be rectangular (all rows have the same
// length) and contain at least one row and one column. For zero-sized tensors
// prefer NewZerosTensor.
func NewTensorFromMatrix(matrix [][]float32) (*Tensor, error) {
	if matrix == nil {
		return nil, fmt.Errorf("spiraltorch: matrix cannot be nil")
	}
	rows := len(matrix)
	if rows == 0 {
		return nil, fmt.Errorf("spiraltorch: matrix must contain at least one row")
	}

	cols := len(matrix[0])
	if cols == 0 {
		return nil, fmt.Errorf("spiraltorch: matrix must contain at least one column")
	}
	data := make([]float32, rows*cols)
	offset := 0
	for i, row := range matrix {
		if len(row) != cols {
			return nil, fmt.Errorf("spiraltorch: matrix row %d has length %d, expected %d", i, len(row), cols)
		}
		copy(data[offset:offset+cols], row)
		offset += cols
	}
	return NewTensorFromDense(rows, cols, data)
}

// NewTensorFromColumns constructs a tensor from a slice of equally sized column
// vectors. The columns are interpreted in column-major order and converted to
// the row-major layout expected by SpiralTorch. The input must contain at least
// one column and each column must have at least one element. For zero-sized
// tensors prefer NewZerosTensor.
func NewTensorFromColumns(columns [][]float32) (*Tensor, error) {
	if columns == nil {
		return nil, fmt.Errorf("spiraltorch: columns cannot be nil")
	}
	cols := len(columns)
	if cols == 0 {
		return nil, fmt.Errorf("spiraltorch: columns must contain at least one column")
	}

	rows := len(columns[0])
	if rows == 0 {
		return nil, fmt.Errorf("spiraltorch: columns must contain at least one row")
	}
	for i := 1; i < cols; i++ {
		if len(columns[i]) != rows {
			return nil, fmt.Errorf("spiraltorch: column %d has length %d, expected %d", i, len(columns[i]), rows)
		}
	}

	data := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		base := r * cols
		for c := 0; c < cols; c++ {
			data[base+c] = columns[c][r]
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
		matrix := make([][]float32, rows)
		for r := range matrix {
			matrix[r] = make([]float32, cols)
		}
		return matrix, nil
	}

	matrix := make([][]float32, rows)
	for r := 0; r < rows; r++ {
		row := make([]float32, cols)
		if err := t.copyRowInto(r, cols, row); err != nil {
			return nil, err
		}
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

	for c := 0; c < cols; c++ {
		column := make([]float32, rows)
		if err := t.copyColumnInto(c, rows, column); err != nil {
			return nil, err
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

	row := make([]float32, cols)
	if err := t.copyRowInto(index, cols, row); err != nil {
		return nil, err
	}
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

	column := make([]float32, rows)
	if err := t.copyColumnInto(index, rows, column); err != nil {
		return nil, err
	}
	return column, nil
}
