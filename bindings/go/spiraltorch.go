// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lspiraltorch_sys
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

size_t spiraltorch_version(char *buffer, size_t capacity);
size_t spiraltorch_last_error_length(void);
size_t spiraltorch_last_error_message(char *buffer, size_t capacity);
void spiraltorch_clear_last_error(void);

void *spiraltorch_tensor_zeros(size_t rows, size_t cols);
void *spiraltorch_tensor_from_dense(size_t rows, size_t cols, const float *data, size_t len);
void spiraltorch_tensor_free(void *tensor);
bool spiraltorch_tensor_shape(const void *tensor, size_t *rows, size_t *cols);
bool spiraltorch_tensor_copy_data(const void *tensor, float *out, size_t len);
bool spiraltorch_tensor_copy_row(const void *tensor, size_t index, float *out, size_t len);
bool spiraltorch_tensor_copy_column(const void *tensor, size_t index, float *out, size_t len);
void *spiraltorch_tensor_add(const void *lhs, const void *rhs);
void *spiraltorch_tensor_sub(const void *lhs, const void *rhs);
void *spiraltorch_tensor_scale(const void *tensor, float value);
void *spiraltorch_tensor_matmul(const void *lhs, const void *rhs);
void *spiraltorch_tensor_transpose(const void *tensor);
void *spiraltorch_tensor_reshape(const void *tensor, size_t rows, size_t cols);
void *spiraltorch_tensor_hadamard(const void *lhs, const void *rhs);
void *spiraltorch_tensor_random_uniform(
    size_t rows,
    size_t cols,
    float min,
    float max,
    uint64_t seed,
    bool has_seed
);
void *spiraltorch_tensor_random_normal(
    size_t rows,
    size_t cols,
    float mean,
    float std,
    uint64_t seed,
    bool has_seed
);

void *spiraltorch_runtime_new(size_t worker_threads, const char *thread_name);
void spiraltorch_runtime_free(void *runtime);
size_t spiraltorch_runtime_worker_count(const void *runtime);
void *spiraltorch_runtime_tensor_add(const void *runtime, const void *lhs, const void *rhs);
void *spiraltorch_runtime_tensor_sub(const void *runtime, const void *lhs, const void *rhs);
void *spiraltorch_runtime_tensor_scale(const void *runtime, const void *tensor, float value);
void *spiraltorch_runtime_tensor_matmul(const void *runtime, const void *lhs, const void *rhs);
void *spiraltorch_runtime_tensor_transpose(const void *runtime, const void *tensor);
void *spiraltorch_runtime_tensor_reshape(const void *runtime, const void *tensor, size_t rows, size_t cols);
void *spiraltorch_runtime_tensor_hadamard(const void *runtime, const void *lhs, const void *rhs);
void *spiraltorch_runtime_tensor_random_uniform(
    const void *runtime,
    size_t rows,
    size_t cols,
    float min,
    float max,
    uint64_t seed,
    bool has_seed
);
void *spiraltorch_runtime_tensor_random_normal(
    const void *runtime,
    size_t rows,
    size_t cols,
    float mean,
    float std,
    uint64_t seed,
    bool has_seed
);

struct spiraltorch_roundtable_summary {
    size_t above;
    size_t here;
    size_t beneath;
    float energy_above;
    float energy_here;
    float energy_beneath;
};

bool spiraltorch_roundtable_classify(
    const float *gradient,
    size_t len,
    size_t above_k,
    size_t here_k,
    size_t beneath_k,
    float tolerance,
    uint8_t *assignments,
    struct spiraltorch_roundtable_summary *summary
);
*/
import "C"

import (
	"fmt"
	goruntime "runtime"
	"unsafe"
)

// Version returns the semantic version of the underlying runtime exposed by the
// shared library.
func Version() string {
	length := C.spiraltorch_version(nil, 0)
	if length == 0 {
		return ""
	}
	buffer := make([]C.char, length+1)
	written := C.spiraltorch_version(&buffer[0], C.size_t(len(buffer)))
	return C.GoStringN(&buffer[0], C.int(written))
}

func lastError() string {
	length := C.spiraltorch_last_error_length()
	if length == 0 {
		return ""
	}
	buffer := make([]C.char, length+1)
	written := C.spiraltorch_last_error_message(&buffer[0], C.size_t(len(buffer)))
	return C.GoStringN(&buffer[0], C.int(written))
}

func clearError() {
	C.spiraltorch_clear_last_error()
}

// Tensor owns a pointer to a tensor allocated by the SpiralTorch runtime.
type Tensor struct {
	handle     unsafe.Pointer
	rows       int
	cols       int
	shapeKnown bool
}

func wrapTensor(ptr unsafe.Pointer, context string) (*Tensor, error) {
	if ptr == nil {
		err := lastError()
		if err == "" {
			err = "unknown error"
		}
		return nil, fmt.Errorf("spiraltorch: %s failed: %s", context, err)
	}
	tensor := &Tensor{handle: ptr}
	goruntime.SetFinalizer(tensor, func(t *Tensor) {
		t.Close()
	})
	clearError()
	return tensor, nil
}

func (t *Tensor) setKnownShape(rows, cols int) {
	t.rows = rows
	t.cols = cols
	t.shapeKnown = true
}

func (t *Tensor) cacheShape() error {
	if t.shapeKnown {
		return nil
	}
	var rows, cols C.size_t
	ok := C.spiraltorch_tensor_shape(t.handle, &rows, &cols)
	if !bool(ok) {
		message := lastError()
		if message == "" {
			message = "tensor_shape failed"
		}
		return fmt.Errorf("spiraltorch: %s", message)
	}
	clearError()
	t.setKnownShape(int(rows), int(cols))
	return nil
}

// Runtime wraps the SpiralTorch golden runtime for scheduling tensor operations.
type Runtime struct {
	handle unsafe.Pointer
}

func wrapRuntime(ptr unsafe.Pointer, context string) (*Runtime, error) {
	if ptr == nil {
		err := lastError()
		if err == "" {
			err = "unknown error"
		}
		return nil, fmt.Errorf("spiraltorch: %s failed: %s", context, err)
	}
	runtime := &Runtime{handle: ptr}
	goruntime.SetFinalizer(runtime, func(r *Runtime) {
		r.Close()
	})
	clearError()
	return runtime, nil
}

// NewRuntime constructs a runtime with the requested worker count. When
// workerThreads <= 0 the core selects a sensible default. An empty threadName
// uses SpiralTorch's standard label.
func NewRuntime(workerThreads int, threadName string) (*Runtime, error) {
	var namePtr *C.char
	if threadName != "" {
		namePtr = C.CString(threadName)
		defer C.free(unsafe.Pointer(namePtr))
	}
	ptr := C.spiraltorch_runtime_new(C.size_t(workerThreads), namePtr)
	return wrapRuntime(ptr, "runtime_new")
}

// Close releases the underlying runtime handle. Subsequent calls are safe.
func (r *Runtime) Close() {
	if r == nil || r.handle == nil {
		return
	}
	C.spiraltorch_runtime_free(r.handle)
	r.handle = nil
	clearError()
}

func (r *Runtime) requireHandle(context string) (unsafe.Pointer, error) {
	if r == nil || r.handle == nil {
		return nil, fmt.Errorf("spiraltorch: %s runtime handle is nil", context)
	}
	return r.handle, nil
}

// WorkerCount returns how many worker threads back this runtime.
func (r *Runtime) WorkerCount() (int, error) {
	handle, err := r.requireHandle("worker_count")
	if err != nil {
		return 0, err
	}
	count := C.spiraltorch_runtime_worker_count(handle)
	if count == 0 {
		if message := lastError(); message != "" {
			return 0, fmt.Errorf("spiraltorch: runtime_worker_count failed: %s", message)
		}
	}
	clearError()
	return int(count), nil
}

func requireTensorHandle(t *Tensor, context string) (unsafe.Pointer, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: %s tensor handle is nil", context)
	}
	return t.handle, nil
}

// Add schedules an element-wise addition on the runtime.
func (r *Runtime) Add(lhs, rhs *Tensor) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_add")
	if err != nil {
		return nil, err
	}
	rows, cols, err := lhs.Shape()
	if err != nil {
		return nil, err
	}
	left, err := requireTensorHandle(lhs, "runtime_tensor_add lhs")
	if err != nil {
		return nil, err
	}
	right, err := requireTensorHandle(rhs, "runtime_tensor_add rhs")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_add(handle, left, right)
	tensor, err := wrapTensor(ptr, "runtime_tensor_add")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Sub schedules an element-wise subtraction on the runtime.
func (r *Runtime) Sub(lhs, rhs *Tensor) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_sub")
	if err != nil {
		return nil, err
	}
	rows, cols, err := lhs.Shape()
	if err != nil {
		return nil, err
	}
	left, err := requireTensorHandle(lhs, "runtime_tensor_sub lhs")
	if err != nil {
		return nil, err
	}
	right, err := requireTensorHandle(rhs, "runtime_tensor_sub rhs")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_sub(handle, left, right)
	tensor, err := wrapTensor(ptr, "runtime_tensor_sub")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Hadamard schedules an element-wise product on the runtime.
func (r *Runtime) Hadamard(lhs, rhs *Tensor) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_hadamard")
	if err != nil {
		return nil, err
	}
	rows, cols, err := lhs.Shape()
	if err != nil {
		return nil, err
	}
	left, err := requireTensorHandle(lhs, "runtime_tensor_hadamard lhs")
	if err != nil {
		return nil, err
	}
	right, err := requireTensorHandle(rhs, "runtime_tensor_hadamard rhs")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_hadamard(handle, left, right)
	tensor, err := wrapTensor(ptr, "runtime_tensor_hadamard")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Matmul schedules a matrix multiplication on the runtime.
func (r *Runtime) Matmul(lhs, rhs *Tensor) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_matmul")
	if err != nil {
		return nil, err
	}
	lhsRows, _, err := lhs.Shape()
	if err != nil {
		return nil, err
	}
	_, rhsCols, err := rhs.Shape()
	if err != nil {
		return nil, err
	}
	left, err := requireTensorHandle(lhs, "runtime_tensor_matmul lhs")
	if err != nil {
		return nil, err
	}
	right, err := requireTensorHandle(rhs, "runtime_tensor_matmul rhs")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_matmul(handle, left, right)
	tensor, err := wrapTensor(ptr, "runtime_tensor_matmul")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(lhsRows, rhsCols)
	return tensor, nil
}

// RandomUniformTensor schedules construction of a tensor sampled from [min, max).
// When seed is supplied the results are deterministic.
func (r *Runtime) RandomUniformTensor(rows, cols int, min, max float32, seed ...uint64) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_random_uniform")
	if err != nil {
		return nil, err
	}
	var (
		seedValue C.uint64_t
		hasSeed   C.bool
	)
	if len(seed) > 0 {
		if len(seed) > 1 {
			return nil, fmt.Errorf("spiraltorch: random_uniform expects at most one seed value")
		}
		seedValue = C.uint64_t(seed[0])
		hasSeed = C.bool(true)
	}
	ptr := C.spiraltorch_runtime_tensor_random_uniform(
		handle,
		C.size_t(rows),
		C.size_t(cols),
		C.float(min),
		C.float(max),
		seedValue,
		hasSeed,
	)
	tensor, err := wrapTensor(ptr, "runtime_tensor_random_uniform")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// RandomNormalTensor schedules construction of a tensor sampled from a normal distribution.
// When seed is provided sampling becomes deterministic.
func (r *Runtime) RandomNormalTensor(rows, cols int, mean, std float32, seed ...uint64) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_random_normal")
	if err != nil {
		return nil, err
	}
	var (
		seedValue C.uint64_t
		hasSeed   C.bool
	)
	if len(seed) > 0 {
		if len(seed) > 1 {
			return nil, fmt.Errorf("spiraltorch: random_normal expects at most one seed value")
		}
		seedValue = C.uint64_t(seed[0])
		hasSeed = C.bool(true)
	}
	ptr := C.spiraltorch_runtime_tensor_random_normal(
		handle,
		C.size_t(rows),
		C.size_t(cols),
		C.float(mean),
		C.float(std),
		seedValue,
		hasSeed,
	)
	tensor, err := wrapTensor(ptr, "runtime_tensor_random_normal")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Scale multiplies all tensor elements by value on the runtime.
func (r *Runtime) Scale(t *Tensor, value float32) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_scale")
	if err != nil {
		return nil, err
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	tensorHandle, err := requireTensorHandle(t, "runtime_tensor_scale tensor")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_scale(handle, tensorHandle, C.float(value))
	tensor, err := wrapTensor(ptr, "runtime_tensor_scale")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Transpose schedules a transpose operation on the runtime.
func (r *Runtime) Transpose(t *Tensor) (*Tensor, error) {
	handle, err := r.requireHandle("runtime_tensor_transpose")
	if err != nil {
		return nil, err
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	tensorHandle, err := requireTensorHandle(t, "runtime_tensor_transpose tensor")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_transpose(handle, tensorHandle)
	tensor, err := wrapTensor(ptr, "runtime_tensor_transpose")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(cols, rows)
	return tensor, nil
}

// Reshape schedules a reshape on the runtime.
func (r *Runtime) Reshape(t *Tensor, rows, cols int) (*Tensor, error) {
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("spiraltorch: reshape dimensions must be non-negative")
	}
	handle, err := r.requireHandle("runtime_tensor_reshape")
	if err != nil {
		return nil, err
	}
	tensorHandle, err := requireTensorHandle(t, "runtime_tensor_reshape tensor")
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_runtime_tensor_reshape(
		handle,
		tensorHandle,
		C.size_t(rows),
		C.size_t(cols),
	)
	tensor, err := wrapTensor(ptr, "runtime_tensor_reshape")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// NewZerosTensor constructs a tensor of the requested shape initialised with zeros.
func NewZerosTensor(rows, cols int) (*Tensor, error) {
	ptr := C.spiraltorch_tensor_zeros(C.size_t(rows), C.size_t(cols))
	tensor, err := wrapTensor(ptr, "tensor_zeros")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// NewTensorFromDense constructs a tensor from the provided row-major data slice.
//
// The data length must match rows*cols. Zero-sized tensors are permitted and are
// created via the same zero allocator used by NewZerosTensor.
func NewTensorFromDense(rows, cols int, data []float32) (*Tensor, error) {
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("spiraltorch: dimensions must be non-negative")
	}
	if rows == 0 || cols == 0 {
		if len(data) != 0 {
			return nil, fmt.Errorf("spiraltorch: data length %d does not match shape %dx%d", len(data), rows, cols)
		}
		return NewZerosTensor(rows, cols)
	}
	if rows*cols != len(data) {
		return nil, fmt.Errorf("spiraltorch: data length %d does not match shape %dx%d", len(data), rows, cols)
	}
	ptr := C.spiraltorch_tensor_from_dense(
		C.size_t(rows),
		C.size_t(cols),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
	)
	tensor, err := wrapTensor(ptr, "tensor_from_dense")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// NewRandomUniformTensor constructs a tensor with values sampled from [min, max).
//
// When seed is provided the distribution becomes deterministic.
func NewRandomUniformTensor(rows, cols int, min, max float32, seed ...uint64) (*Tensor, error) {
	var (
		seedValue C.uint64_t
		hasSeed   C.bool
	)
	if len(seed) > 0 {
		if len(seed) > 1 {
			return nil, fmt.Errorf("spiraltorch: random_uniform expects at most one seed value")
		}
		seedValue = C.uint64_t(seed[0])
		hasSeed = C.bool(true)
	}
	ptr := C.spiraltorch_tensor_random_uniform(
		C.size_t(rows),
		C.size_t(cols),
		C.float(min),
		C.float(max),
		seedValue,
		hasSeed,
	)
	tensor, err := wrapTensor(ptr, "tensor_random_uniform")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// NewRandomNormalTensor constructs a tensor with values sampled from a normal distribution.
//
// When seed is provided the sampling becomes deterministic.
func NewRandomNormalTensor(rows, cols int, mean, std float32, seed ...uint64) (*Tensor, error) {
	var (
		seedValue C.uint64_t
		hasSeed   C.bool
	)
	if len(seed) > 0 {
		if len(seed) > 1 {
			return nil, fmt.Errorf("spiraltorch: random_normal expects at most one seed value")
		}
		seedValue = C.uint64_t(seed[0])
		hasSeed = C.bool(true)
	}
	ptr := C.spiraltorch_tensor_random_normal(
		C.size_t(rows),
		C.size_t(cols),
		C.float(mean),
		C.float(std),
		seedValue,
		hasSeed,
	)
	tensor, err := wrapTensor(ptr, "tensor_random_normal")
	if err != nil {
		return nil, err
	}
	tensor.setKnownShape(rows, cols)
	return tensor, nil
}

// Close releases the underlying tensor handle. Subsequent calls are safe.
func (t *Tensor) Close() {
	if t == nil || t.handle == nil {
		return
	}
	C.spiraltorch_tensor_free(t.handle)
	t.handle = nil
	clearError()
}

// Shape returns the `(rows, cols)` pair describing the tensor dimensions.
func (t *Tensor) Shape() (int, int, error) {
	if t == nil || t.handle == nil {
		return 0, 0, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if err := t.cacheShape(); err != nil {
		return 0, 0, err
	}
	return t.rows, t.cols, nil
}

// Elements returns the number of elements stored in the tensor.
func (t *Tensor) Elements() (int, error) {
	if t == nil || t.handle == nil {
		return 0, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if err := t.cacheShape(); err != nil {
		return 0, err
	}
	return t.rows * t.cols, nil
}

// Data copies the tensor contents into a newly allocated slice.
func (t *Tensor) Data() ([]float32, error) {
	length, err := t.Elements()
	if err != nil {
		return nil, err
	}
	if length == 0 {
		return []float32{}, nil
	}
	buffer := make([]float32, length)
	if err := t.copyDataInto(buffer); err != nil {
		return nil, err
	}
	return buffer, nil
}

func (t *Tensor) copyDataInto(buffer []float32) error {
	if t == nil || t.handle == nil {
		return fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if len(buffer) == 0 {
		return nil
	}
	ok := C.spiraltorch_tensor_copy_data(
		t.handle,
		(*C.float)(unsafe.Pointer(&buffer[0])),
		C.size_t(len(buffer)),
	)
	if !bool(ok) {
		return fmt.Errorf("spiraltorch: %s", lastError())
	}
	clearError()
	return nil
}

func (t *Tensor) copyRowInto(index, cols int, dest []float32) error {
	if t == nil || t.handle == nil {
		return fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if len(dest) < cols {
		return fmt.Errorf("spiraltorch: destination length %d insufficient for %d columns", len(dest), cols)
	}
	if cols == 0 {
		return nil
	}
	ok := C.spiraltorch_tensor_copy_row(
		t.handle,
		C.size_t(index),
		(*C.float)(unsafe.Pointer(&dest[0])),
		C.size_t(cols),
	)
	if !bool(ok) {
		return fmt.Errorf("spiraltorch: %s", lastError())
	}
	clearError()
	return nil
}

// CopyRowInto copies the requested row into `dest`, reusing the caller-provided
// storage when it is large enough.
func (t *Tensor) CopyRowInto(index int, dest []float32) error {
	if index < 0 {
		return fmt.Errorf("spiraltorch: row index %d cannot be negative", index)
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return err
	}
	if index >= rows {
		return fmt.Errorf("spiraltorch: row index %d out of range [0,%d)", index, rows)
	}
	return t.copyRowInto(index, cols, dest)
}

func (t *Tensor) copyColumnInto(index, rows int, dest []float32) error {
	if t == nil || t.handle == nil {
		return fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if len(dest) < rows {
		return fmt.Errorf("spiraltorch: destination length %d insufficient for %d rows", len(dest), rows)
	}
	if rows == 0 {
		return nil
	}
	ok := C.spiraltorch_tensor_copy_column(
		t.handle,
		C.size_t(index),
		(*C.float)(unsafe.Pointer(&dest[0])),
		C.size_t(rows),
	)
	if !bool(ok) {
		return fmt.Errorf("spiraltorch: %s", lastError())
	}
	clearError()
	return nil
}

// CopyColumnInto copies the requested column into `dest`, reusing the
// caller-provided storage when it is large enough.
func (t *Tensor) CopyColumnInto(index int, dest []float32) error {
	if index < 0 {
		return fmt.Errorf("spiraltorch: column index %d cannot be negative", index)
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return err
	}
	if index >= cols {
		return fmt.Errorf("spiraltorch: column index %d out of range [0,%d)", index, cols)
	}
	return t.copyColumnInto(index, rows, dest)
}

// CopyDataInto copies the tensor contents into the provided slice, reusing the
// caller's storage when it is sufficiently large. The number of elements
// written is returned so callers can reslice the buffer as needed.
func (t *Tensor) CopyDataInto(dest []float32) (int, error) {
	length, err := t.Elements()
	if err != nil {
		return 0, err
	}
	if length == 0 {
		return 0, nil
	}
	if len(dest) < length {
		return length, fmt.Errorf("spiraltorch: destination length %d insufficient for %d elements", len(dest), length)
	}
	if err := t.copyDataInto(dest[:length]); err != nil {
		return 0, err
	}
	return length, nil
}

func (t *Tensor) binaryOp(other *Tensor, op func(unsafe.Pointer, unsafe.Pointer) unsafe.Pointer, label string) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if other == nil || other.handle == nil {
		return nil, fmt.Errorf("spiraltorch: other tensor handle is nil")
	}
	ptr := op(t.handle, other.handle)
	return wrapTensor(ptr, label)
}

// Add performs element-wise addition and returns a new tensor.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	result, err := t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_add(lhs, rhs)
	}, "tensor_add")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// Sub performs element-wise subtraction and returns a new tensor.
func (t *Tensor) Sub(other *Tensor) (*Tensor, error) {
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	result, err := t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_sub(lhs, rhs)
	}, "tensor_sub")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// Matmul performs matrix multiplication (`t @ other`).
func (t *Tensor) Matmul(other *Tensor) (*Tensor, error) {
	rows, _, err := t.Shape()
	if err != nil {
		return nil, err
	}
	_, cols, err := other.Shape()
	if err != nil {
		return nil, err
	}
	result, err := t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_matmul(lhs, rhs)
	}, "tensor_matmul")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// Hadamard performs element-wise multiplication and returns a new tensor.
func (t *Tensor) Hadamard(other *Tensor) (*Tensor, error) {
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	result, err := t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_hadamard(lhs, rhs)
	}, "tensor_hadamard")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// Scale multiplies every element by the provided value.
func (t *Tensor) Scale(value float32) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_tensor_scale(t.handle, C.float(value))
	result, err := wrapTensor(ptr, "tensor_scale")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// Transpose returns a new tensor with flipped dimensions.
func (t *Tensor) Transpose() (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	ptr := C.spiraltorch_tensor_transpose(t.handle)
	result, err := wrapTensor(ptr, "tensor_transpose")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(cols, rows)
	return result, nil
}

// Reshape returns a tensor that views the same data with new `(rows, cols)` dimensions.
func (t *Tensor) Reshape(rows, cols int) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("spiraltorch: reshape dimensions must be non-negative")
	}
	ptr := C.spiraltorch_tensor_reshape(t.handle, C.size_t(rows), C.size_t(cols))
	result, err := wrapTensor(ptr, "tensor_reshape")
	if err != nil {
		return nil, err
	}
	result.setKnownShape(rows, cols)
	return result, nil
}

// RoundtableBand identifies which SpiralTorch roundtable band owns a gradient lane.
type RoundtableBand uint8

const (
	// RoundtableBandAbove marks entries assigned to the Above (TopK) band.
	RoundtableBandAbove RoundtableBand = 0
	// RoundtableBandHere marks entries assigned to the Here (MidK) band.
	RoundtableBandHere RoundtableBand = 1
	// RoundtableBandBeneath marks entries assigned to the Beneath (BottomK) band.
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

// RoundtableClassify partitions the provided gradient magnitudes into Above/Here/Beneath bands.
//
// The classification mirrors the roundtable heuristic used by the Rust runtime: the largest
// magnitudes are assigned to Above, the smallest to Beneath, and any remaining lanes are
// preserved as Here unless their magnitude is below `tolerance`. The function returns one band per
// gradient entry along with summary statistics.
func RoundtableClassify(gradient []float32, aboveK, hereK, beneathK int, tolerance float32) ([]RoundtableBand, RoundtableSummary, error) {
	if len(gradient) == 0 {
		return nil, RoundtableSummary{}, fmt.Errorf("spiraltorch: roundtable_classify requires at least one gradient value")
	}
	assignments := make([]C.uchar, len(gradient))
	var summary C.struct_spiraltorch_roundtable_summary
	success := C.spiraltorch_roundtable_classify(
		(*C.float)(unsafe.Pointer(&gradient[0])),
		C.size_t(len(gradient)),
		C.size_t(aboveK),
		C.size_t(hereK),
		C.size_t(beneathK),
		C.float(tolerance),
		(*C.uchar)(unsafe.Pointer(&assignments[0])),
		&summary,
	)
	if !bool(success) {
		message := lastError()
		if message == "" {
			message = "roundtable_classify failed"
		}
		return nil, RoundtableSummary{}, fmt.Errorf("spiraltorch: %s", message)
	}
	bands := make([]RoundtableBand, len(gradient))
	for i := range bands {
		bands[i] = RoundtableBand(assignments[i])
	}
	summaryGo := RoundtableSummary{
		Above:         int(summary.above),
		Here:          int(summary.here),
		Beneath:       int(summary.beneath),
		EnergyAbove:   float32(summary.energy_above),
		EnergyHere:    float32(summary.energy_here),
		EnergyBeneath: float32(summary.energy_beneath),
	}
	return bands, summaryGo, nil
}
