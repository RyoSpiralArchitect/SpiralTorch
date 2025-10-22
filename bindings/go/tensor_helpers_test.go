package spiraltorch

import "testing"

func TestNewTensorFromMatrixRoundTrip(t *testing.T) {
	matrix := [][]float32{{1, 2, 3}, {4, 5, 6}}

	tensor, err := NewTensorFromMatrix(matrix)
	if err != nil {
		t.Fatalf("NewTensorFromMatrix returned error: %v", err)
	}
	t.Cleanup(func() { tensor.Close() })

	rows, cols, err := tensor.Shape()
	if err != nil {
		t.Fatalf("Shape returned error: %v", err)
	}
	if rows != 2 || cols != 3 {
		t.Fatalf("unexpected shape: got %dx%d", rows, cols)
	}

	data, err := tensor.Data()
	if err != nil {
		t.Fatalf("Data returned error: %v", err)
	}

	expected := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range expected {
		if data[i] != v {
			t.Fatalf("unexpected data at index %d: got %v want %v", i, data[i], v)
		}
	}

	decoded, err := tensor.ToMatrix()
	if err != nil {
		t.Fatalf("ToMatrix returned error: %v", err)
	}
	decoded[0][0] = 99

	dataAfter, err := tensor.Data()
	if err != nil {
		t.Fatalf("Data after decode returned error: %v", err)
	}
	if dataAfter[0] != 1 {
		t.Fatalf("tensor data mutated: got %v want 1", dataAfter[0])
	}
}

func TestNewTensorFromMatrixValidation(t *testing.T) {
	if _, err := NewTensorFromMatrix(nil); err == nil {
		t.Fatalf("expected error for nil matrix")
	}

	ragged := [][]float32{{1}, {2, 3}}
	if _, err := NewTensorFromMatrix(ragged); err == nil {
		t.Fatalf("expected error for ragged matrix")
	}
}

func TestMatrixHelpersWithEmptyDimensions(t *testing.T) {
	zeroRows := [][]float32{}
	tensor, err := NewTensorFromMatrix(zeroRows)
	if err != nil {
		t.Fatalf("unexpected error for zero rows: %v", err)
	}
	t.Cleanup(func() { tensor.Close() })

	rows, cols, err := tensor.Shape()
	if err != nil {
		t.Fatalf("Shape returned error: %v", err)
	}
	if rows != 0 || cols != 0 {
		t.Fatalf("unexpected shape for zero rows: got %dx%d", rows, cols)
	}

	zeroCols := [][]float32{{}, {}}
	tensorCols, err := NewTensorFromMatrix(zeroCols)
	if err != nil {
		t.Fatalf("unexpected error for zero columns: %v", err)
	}
	t.Cleanup(func() { tensorCols.Close() })

	rows, cols, err = tensorCols.Shape()
	if err != nil {
		t.Fatalf("Shape returned error: %v", err)
	}
	if rows != 2 || cols != 0 {
		t.Fatalf("unexpected shape for zero columns: got %dx%d", rows, cols)
	}

	decoded, err := tensorCols.ToMatrix()
	if err != nil {
		t.Fatalf("ToMatrix returned error: %v", err)
	}
	if len(decoded) != 2 {
		t.Fatalf("unexpected decoded row count: %d", len(decoded))
	}
	for i, row := range decoded {
		if len(row) != 0 {
			t.Fatalf("expected zero-length row at index %d", i)
		}
	}
}
