package spiraltorch

import "testing"

func requireRuntime(t *testing.T) *Runtime {
	t.Helper()
	runtime, err := NewRuntime(0, "test-runtime")
	if err != nil {
		t.Fatalf("NewRuntime returned error: %v", err)
	}
	t.Cleanup(func() { runtime.Close() })
	return runtime
}

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

func TestNewTensorFromColumnsRoundTrip(t *testing.T) {
	columns := [][]float32{{1, 4}, {2, 5}, {3, 6}}

	tensor, err := NewTensorFromColumns(columns)
	if err != nil {
		t.Fatalf("NewTensorFromColumns returned error: %v", err)
	}
	t.Cleanup(func() { tensor.Close() })

	rows, cols, err := tensor.Shape()
	if err != nil {
		t.Fatalf("Shape returned error: %v", err)
	}
	if rows != 2 || cols != 3 {
		t.Fatalf("unexpected shape: got %dx%d", rows, cols)
	}

	decoded, err := tensor.ToMatrix()
	if err != nil {
		t.Fatalf("ToMatrix returned error: %v", err)
	}

	expected := [][]float32{{1, 2, 3}, {4, 5, 6}}
	for i, row := range expected {
		for j, v := range row {
			if decoded[i][j] != v {
				t.Fatalf("unexpected decoded[%d][%d]: got %v want %v", i, j, decoded[i][j], v)
			}
		}
	}

	colsCopy, err := tensor.Columns()
	if err != nil {
		t.Fatalf("Columns returned error: %v", err)
	}
	if len(colsCopy) != 3 {
		t.Fatalf("unexpected column count: %d", len(colsCopy))
	}
	colsCopy[0][0] = 99

	column, err := tensor.Column(0)
	if err != nil {
		t.Fatalf("Column returned error: %v", err)
	}
	if column[0] != 1 {
		t.Fatalf("tensor column mutated: got %v want 1", column[0])
	}

	row, err := tensor.Row(1)
	if err != nil {
		t.Fatalf("Row returned error: %v", err)
	}
	if len(row) != 3 || row[2] != 6 {
		t.Fatalf("unexpected row contents: %v", row)
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

func TestNewTensorFromColumnsValidation(t *testing.T) {
	if _, err := NewTensorFromColumns(nil); err == nil {
		t.Fatalf("expected error for nil columns")
	}

	ragged := [][]float32{{1}, {2, 3}}
	if _, err := NewTensorFromColumns(ragged); err == nil {
		t.Fatalf("expected error for ragged columns")
	}
}

func TestMatrixHelpersWithEmptyDimensions(t *testing.T) {
	zeroRows := [][]float32{}
	if _, err := NewTensorFromMatrix(zeroRows); err == nil {
		t.Fatalf("expected error for zero rows")
	}

	zeroCols := [][]float32{{}, {}}
	if _, err := NewTensorFromMatrix(zeroCols); err == nil {
		t.Fatalf("expected error for zero columns")
	}
}

func TestColumnHelpersWithEmptyDimensions(t *testing.T) {
	zeroColumns := [][]float32{}
	if _, err := NewTensorFromColumns(zeroColumns); err == nil {
		t.Fatalf("expected error for zero columns input")
	}

	zeroRows := [][]float32{{}, {}}
	if _, err := NewTensorFromColumns(zeroRows); err == nil {
		t.Fatalf("expected error for zero rows input")
	}
}

func TestNewTensorFromDenseZeroDimensions(t *testing.T) {
	cases := []struct {
		name string
		rows int
		cols int
	}{
		{name: "zero_rows", rows: 0, cols: 5},
		{name: "zero_cols", rows: 3, cols: 0},
		{name: "both_zero", rows: 0, cols: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewTensorFromDense(tc.rows, tc.cols, nil); err == nil {
				t.Fatalf("expected error for zero-dimension dense tensor")
			}
		})
	}
}

func TestNewTensorFromDenseValidation(t *testing.T) {
	if _, err := NewTensorFromDense(-1, 2, []float32{1, 2}); err == nil {
		t.Fatalf("expected error for negative rows")
	}
	if _, err := NewTensorFromDense(2, -1, []float32{1, 2}); err == nil {
		t.Fatalf("expected error for negative cols")
	}
	if _, err := NewTensorFromDense(0, 4, []float32{1}); err == nil {
		t.Fatalf("expected error for mismatched zero-dimension data")
	}
	if _, err := NewTensorFromDense(2, 2, []float32{1, 2, 3}); err == nil {
		t.Fatalf("expected error for mismatched data length")
	}
}

func TestTensorCopyDataInto(t *testing.T) {
	tensor, err := NewTensorFromDense(2, 2, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("NewTensorFromDense returned error: %v", err)
	}
	t.Cleanup(func() { tensor.Close() })

	dest := make([]float32, 6)
	written, err := tensor.CopyDataInto(dest)
	if err != nil {
		t.Fatalf("CopyDataInto returned error: %v", err)
	}
	if written != 4 {
		t.Fatalf("unexpected element count: got %d want 4", written)
	}
	expected := []float32{1, 2, 3, 4}
	for i, value := range expected {
		if dest[i] != value {
			t.Fatalf("unexpected dest[%d]: got %v want %v", i, dest[i], value)
		}
	}

	if _, err := tensor.CopyDataInto(make([]float32, 3)); err == nil {
		t.Fatalf("expected error for undersized destination buffer")
	}

	zero, err := NewZerosTensor(0, 5)
	if err != nil {
		t.Fatalf("NewZerosTensor returned error: %v", err)
	}
	t.Cleanup(func() { zero.Close() })
	if written, err := zero.CopyDataInto(nil); err != nil {
		t.Fatalf("CopyDataInto on zero tensor returned error: %v", err)
	} else if written != 0 {
		t.Fatalf("unexpected elements written for zero tensor: %d", written)
	}
}

func TestRuntimeParallelMatmul(t *testing.T) {
	runtime := requireRuntime(t)

	lhs1, err := NewTensorFromDense(2, 2, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("NewTensorFromDense lhs1 error: %v", err)
	}
	t.Cleanup(func() { lhs1.Close() })

	rhs1, err := NewTensorFromDense(2, 2, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("NewTensorFromDense rhs1 error: %v", err)
	}
	t.Cleanup(func() { rhs1.Close() })

	lhs2, err := NewTensorFromDense(2, 2, []float32{2, 0, 1, 2})
	if err != nil {
		t.Fatalf("NewTensorFromDense lhs2 error: %v", err)
	}
	t.Cleanup(func() { lhs2.Close() })

	rhs2, err := NewTensorFromDense(2, 2, []float32{1, 3, 4, 2})
	if err != nil {
		t.Fatalf("NewTensorFromDense rhs2 error: %v", err)
	}
	t.Cleanup(func() { rhs2.Close() })

	pairs := []TensorPair{{LHS: lhs1, RHS: rhs1}, {LHS: lhs2, RHS: rhs2}}

	results, err := runtime.ParallelMatmul(pairs, 2)
	if err != nil {
		t.Fatalf("ParallelMatmul returned error: %v", err)
	}
	for _, tensor := range results {
		t.Cleanup(func(tensor *Tensor) func() {
			return func() { tensor.Close() }
		}(tensor))
	}

	data0, err := results[0].Data()
	if err != nil {
		t.Fatalf("Data for result[0] error: %v", err)
	}
	expected0 := []float32{19, 22, 43, 50}
	for i, value := range expected0 {
		if data0[i] != value {
			t.Fatalf("unexpected result[0][%d]: got %v want %v", i, data0[i], value)
		}
	}

	data1, err := results[1].Data()
	if err != nil {
		t.Fatalf("Data for result[1] error: %v", err)
	}
	expected1 := []float32{2, 6, 9, 7}
	for i, value := range expected1 {
		if data1[i] != value {
			t.Fatalf("unexpected result[1][%d]: got %v want %v", i, data1[i], value)
		}
	}
}

func TestRuntimeParallelAddErrorPropagation(t *testing.T) {
	runtime := requireRuntime(t)

	lhs, err := NewTensorFromDense(2, 2, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("NewTensorFromDense lhs error: %v", err)
	}
	t.Cleanup(func() { lhs.Close() })

	rhs, err := NewTensorFromDense(2, 2, []float32{5, 6, 7, 8})
	if err != nil {
		t.Fatalf("NewTensorFromDense rhs error: %v", err)
	}
	t.Cleanup(func() { rhs.Close() })

	invalid, err := NewTensorFromDense(3, 1, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("NewTensorFromDense invalid error: %v", err)
	}
	t.Cleanup(func() { invalid.Close() })

	pairs := []TensorPair{
		{LHS: lhs, RHS: rhs},
		{LHS: lhs, RHS: invalid},
	}

	if _, err := runtime.ParallelAdd(pairs, 0); err == nil {
		t.Fatalf("expected error from ParallelAdd with mismatched shapes")
	}
}

func TestRoundtableClassifyBatchesGradient(t *testing.T) {
	gradient := []float32{0.9, 0.1, 0.5, 0.05, 0.2, 0.8, 0.4, 0.02}
	bands, summary, err := RoundtableClassify(gradient, 2, 3, 2, 0.01)
	if err != nil {
		t.Fatalf("RoundtableClassify returned error: %v", err)
	}
	if len(bands) != len(gradient) {
		t.Fatalf("expected %d bands but received %d", len(gradient), len(bands))
	}
	if summary.Above != 2 || summary.Beneath != 2 || summary.Here != len(gradient)-4 {
		t.Fatalf("unexpected counts: %+v", summary)
	}
	if summary.EnergyAbove <= summary.EnergyBeneath {
		t.Fatalf("expected Above energy to dominate Beneath: %+v", summary)
	}
	if bands[0] != RoundtableBandAbove {
		t.Fatalf("expected first lane to land in Above but found %s", bands[0])
	}
}

func TestRoundtableClassifyRequiresGradient(t *testing.T) {
	if _, _, err := RoundtableClassify([]float32{}, 1, 1, 1, 0); err == nil {
		t.Fatalf("expected error for empty gradient")
	}
}
