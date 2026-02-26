//go:build !spiraltorch_sys
// +build !spiraltorch_sys

// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

import "fmt"

var errNativeUnavailable = fmt.Errorf("spiraltorch: native library unavailable (build with -tags spiraltorch_sys)")

// Version reports an empty string when the native library is not available.
func Version() string { return "" }

// Tensor is a stub placeholder for the native tensor handle.
type Tensor struct{}

// Runtime is a stub placeholder for the native runtime handle.
type Runtime struct{}

// NewRuntime fails when the native library is unavailable.
func NewRuntime(workerThreads int, threadName string) (*Runtime, error) {
	return nil, errNativeUnavailable
}

// Close is a no-op for the stub.
func (r *Runtime) Close() {}

// WorkerCount fails when the native library is unavailable.
func (r *Runtime) WorkerCount() (int, error) { return 0, errNativeUnavailable }

func (r *Runtime) Add(lhs, rhs *Tensor) (*Tensor, error)      { return nil, errNativeUnavailable }
func (r *Runtime) Sub(lhs, rhs *Tensor) (*Tensor, error)      { return nil, errNativeUnavailable }
func (r *Runtime) Hadamard(lhs, rhs *Tensor) (*Tensor, error) { return nil, errNativeUnavailable }
func (r *Runtime) Matmul(lhs, rhs *Tensor) (*Tensor, error)   { return nil, errNativeUnavailable }
func (r *Runtime) Scale(t *Tensor, value float32) (*Tensor, error) {
	return nil, errNativeUnavailable
}
func (r *Runtime) Transpose(t *Tensor) (*Tensor, error)               { return nil, errNativeUnavailable }
func (r *Runtime) Reshape(t *Tensor, rows, cols int) (*Tensor, error) { return nil, errNativeUnavailable }
func (r *Runtime) RandomUniformTensor(rows, cols int, min, max float32, seed ...uint64) (*Tensor, error) {
	return nil, errNativeUnavailable
}
func (r *Runtime) RandomNormalTensor(rows, cols int, mean, std float32, seed ...uint64) (*Tensor, error) {
	return nil, errNativeUnavailable
}

func NewZerosTensor(rows, cols int) (*Tensor, error) { return nil, errNativeUnavailable }

func NewTensorFromDense(rows, cols int, data []float32) (*Tensor, error) {
	return nil, errNativeUnavailable
}

func NewRandomUniformTensor(rows, cols int, min, max float32, seed ...uint64) (*Tensor, error) {
	return nil, errNativeUnavailable
}

func NewRandomNormalTensor(rows, cols int, mean, std float32, seed ...uint64) (*Tensor, error) {
	return nil, errNativeUnavailable
}

// Close is a no-op for the stub.
func (t *Tensor) Close() {}

func (t *Tensor) Shape() (int, int, error)  { return 0, 0, errNativeUnavailable }
func (t *Tensor) Elements() (int, error)    { return 0, errNativeUnavailable }
func (t *Tensor) Data() ([]float32, error)  { return nil, errNativeUnavailable }
func (t *Tensor) Add(other *Tensor) (*Tensor, error)      { return nil, errNativeUnavailable }
func (t *Tensor) Sub(other *Tensor) (*Tensor, error)      { return nil, errNativeUnavailable }
func (t *Tensor) Matmul(other *Tensor) (*Tensor, error)   { return nil, errNativeUnavailable }
func (t *Tensor) Hadamard(other *Tensor) (*Tensor, error) { return nil, errNativeUnavailable }
func (t *Tensor) Scale(value float32) (*Tensor, error)    { return nil, errNativeUnavailable }
func (t *Tensor) Transpose() (*Tensor, error)             { return nil, errNativeUnavailable }
func (t *Tensor) Reshape(rows, cols int) (*Tensor, error) { return nil, errNativeUnavailable }

func (t *Tensor) CopyRowInto(index int, dest []float32) error    { return errNativeUnavailable }
func (t *Tensor) CopyColumnInto(index int, dest []float32) error { return errNativeUnavailable }
func (t *Tensor) CopyDataInto(dest []float32) (int, error)       { return 0, errNativeUnavailable }

func (t *Tensor) copyRowInto(index, cols int, dest []float32) error    { return errNativeUnavailable }
func (t *Tensor) copyColumnInto(index, rows int, dest []float32) error { return errNativeUnavailable }

// RoundtableBand identifies which SpiralTorch roundtable band owns a gradient lane.
type RoundtableBand uint8

const (
	RoundtableBandAbove   RoundtableBand = 0
	RoundtableBandHere    RoundtableBand = 1
	RoundtableBandBeneath RoundtableBand = 2
)

func (band RoundtableBand) String() string {
	switch band {
	case RoundtableBandAbove:
		return "above"
	case RoundtableBandHere:
		return "here"
	case RoundtableBandBeneath:
		return "beneath"
	default:
		return fmt.Sprintf("unknown(%d)", uint8(band))
	}
}

// RoundtableSummary reports how many entries landed in each band and their total absolute energy.
type RoundtableSummary struct {
	Above         int
	Here          int
	Beneath       int
	EnergyAbove   float32
	EnergyHere    float32
	EnergyBeneath float32
}

func RoundtableClassify(gradient []float32, aboveK, hereK, beneathK int, tolerance float32) ([]RoundtableBand, RoundtableSummary, error) {
	return nil, RoundtableSummary{}, errNativeUnavailable
}

