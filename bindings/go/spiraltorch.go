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
size_t spiraltorch_tensor_elements(const void *tensor);
bool spiraltorch_tensor_copy_data(const void *tensor, float *out, size_t len);
void *spiraltorch_tensor_add(const void *lhs, const void *rhs);
void *spiraltorch_tensor_sub(const void *lhs, const void *rhs);
void *spiraltorch_tensor_scale(const void *tensor, float value);
void *spiraltorch_tensor_matmul(const void *lhs, const void *rhs);
void *spiraltorch_tensor_transpose(const void *tensor);
void *spiraltorch_tensor_reshape(const void *tensor, size_t rows, size_t cols);
void *spiraltorch_tensor_hadamard(const void *lhs, const void *rhs);

bool spiraltorch_foreign_register(const char *language, const char *runtime_id, const char *version, const char *capabilities);
bool spiraltorch_foreign_heartbeat(const char *runtime_id);
bool spiraltorch_foreign_record_latency(const char *runtime_id, const char *operation, uint64_t latency_ns, uint8_t ok_flag);
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
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

type foreignRuntimeClient struct {
	id string
}

var foreignClient *foreignRuntimeClient

func init() {
	foreignClient = registerForeignRuntime()
}

func registerForeignRuntime() *foreignRuntimeClient {
	id := fmt.Sprintf("go-%d-%d", time.Now().UnixNano(), os.Getpid())
	caps := []string{
		"tensor.zeros",
		"tensor.from_dense",
		"tensor.add",
		"tensor.sub",
		"tensor.matmul",
		"tensor.hadamard",
		"tensor.scale",
		"tensor.transpose",
		"tensor.reshape",
	}
	language := C.CString("go")
	runtimeID := C.CString(id)
	version := C.CString(runtime.Version())
	capabilities := C.CString(strings.Join(caps, ","))
	defer C.free(unsafe.Pointer(language))
	defer C.free(unsafe.Pointer(runtimeID))
	defer C.free(unsafe.Pointer(version))
	defer C.free(unsafe.Pointer(capabilities))
	registered := C.spiraltorch_foreign_register(language, runtimeID, version, capabilities)
	if !bool(registered) {
		if err := lastError(); err != "" {
			fmt.Fprintf(os.Stderr, "spiraltorch: failed to register Go runtime: %s\n", err)
			clearError()
		}
		return nil
	}
	return &foreignRuntimeClient{id: id}
}

func startForeignCall(operation string) func(success bool) {
	client := foreignClient
	if client == nil {
		return func(bool) {}
	}
	started := time.Now()
	return func(success bool) {
		client.record(operation, time.Since(started), success)
	}
}

func (c *foreignRuntimeClient) record(operation string, elapsed time.Duration, ok bool) {
	if c == nil {
		return
	}
	runtimeID := C.CString(c.id)
	op := C.CString(operation)
	defer C.free(unsafe.Pointer(runtimeID))
	defer C.free(unsafe.Pointer(op))
	var success C.uint8_t
	if ok {
		success = 1
	} else {
		success = 0
	}
	C.spiraltorch_foreign_record_latency(runtimeID, op, C.uint64_t(elapsed.Nanoseconds()), success)
	clearError()
}

// Tensor owns a pointer to a tensor allocated by the SpiralTorch runtime.
type Tensor struct {
	handle unsafe.Pointer
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
	runtime.SetFinalizer(tensor, func(t *Tensor) {
		t.Close()
	})
	clearError()
	return tensor, nil
}

// NewZerosTensor constructs a tensor of the requested shape initialised with zeros.
func NewZerosTensor(rows, cols int) (*Tensor, error) {
	finish := startForeignCall("tensor.zeros")
	ptr := C.spiraltorch_tensor_zeros(C.size_t(rows), C.size_t(cols))
	finish(ptr != nil)
	return wrapTensor(ptr, "tensor_zeros")
}

// NewTensorFromDense constructs a tensor from the provided row-major data slice.
func NewTensorFromDense(rows, cols int, data []float32) (*Tensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("spiraltorch: data slice cannot be empty")
	}
	if rows*cols != len(data) {
		return nil, fmt.Errorf("spiraltorch: data length %d does not match shape %dx%d", len(data), rows, cols)
	}
	finish := startForeignCall("tensor.from_dense")
	ptr := C.spiraltorch_tensor_from_dense(
		C.size_t(rows),
		C.size_t(cols),
		(*C.float)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
	)
	finish(ptr != nil)
	return wrapTensor(ptr, "tensor_from_dense")
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
	var rows, cols C.size_t
	ok := C.spiraltorch_tensor_shape(t.handle, &rows, &cols)
	if !bool(ok) {
		return 0, 0, fmt.Errorf("spiraltorch: %s", lastError())
	}
	clearError()
	return int(rows), int(cols), nil
}

// Elements returns the number of elements stored in the tensor.
func (t *Tensor) Elements() (int, error) {
	if t == nil || t.handle == nil {
		return 0, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	count := C.spiraltorch_tensor_elements(t.handle)
	if count == 0 {
		if err := lastError(); err != "" {
			return 0, fmt.Errorf("spiraltorch: %s", err)
		}
	}
	clearError()
	return int(count), nil
}

// Data copies the tensor contents into a newly allocated slice.
func (t *Tensor) Data() ([]float32, error) {
	rows, cols, err := t.Shape()
	if err != nil {
		return nil, err
	}
	length := rows * cols
	if length == 0 {
		return []float32{}, nil
	}
	buffer := make([]float32, length)
	ok := C.spiraltorch_tensor_copy_data(
		t.handle,
		(*C.float)(unsafe.Pointer(&buffer[0])),
		C.size_t(length),
	)
	if !bool(ok) {
		return nil, fmt.Errorf("spiraltorch: %s", lastError())
	}
	clearError()
	return buffer, nil
}

func (t *Tensor) binaryOp(other *Tensor, op func(unsafe.Pointer, unsafe.Pointer) unsafe.Pointer, label string) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if other == nil || other.handle == nil {
		return nil, fmt.Errorf("spiraltorch: other tensor handle is nil")
	}
	metric := strings.Replace(label, "_", ".", 1)
	finish := startForeignCall(metric)
	ptr := op(t.handle, other.handle)
	finish(ptr != nil)
	return wrapTensor(ptr, label)
}

// Add performs element-wise addition and returns a new tensor.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	return t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_add(lhs, rhs)
	}, "tensor_add")
}

// Sub performs element-wise subtraction and returns a new tensor.
func (t *Tensor) Sub(other *Tensor) (*Tensor, error) {
	return t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_sub(lhs, rhs)
	}, "tensor_sub")
}

// Matmul performs matrix multiplication (`t @ other`).
func (t *Tensor) Matmul(other *Tensor) (*Tensor, error) {
	return t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_matmul(lhs, rhs)
	}, "tensor_matmul")
}

// Hadamard performs element-wise multiplication and returns a new tensor.
func (t *Tensor) Hadamard(other *Tensor) (*Tensor, error) {
	return t.binaryOp(other, func(lhs, rhs unsafe.Pointer) unsafe.Pointer {
		return C.spiraltorch_tensor_hadamard(lhs, rhs)
	}, "tensor_hadamard")
}

// Scale multiplies every element by the provided value.
func (t *Tensor) Scale(value float32) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	finish := startForeignCall("tensor.scale")
	ptr := C.spiraltorch_tensor_scale(t.handle, C.float(value))
	finish(ptr != nil)
	return wrapTensor(ptr, "tensor_scale")
}

// Transpose returns a new tensor with flipped dimensions.
func (t *Tensor) Transpose() (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	finish := startForeignCall("tensor.transpose")
	ptr := C.spiraltorch_tensor_transpose(t.handle)
	finish(ptr != nil)
	return wrapTensor(ptr, "tensor_transpose")
}

// Reshape returns a tensor that views the same data with new `(rows, cols)` dimensions.
func (t *Tensor) Reshape(rows, cols int) (*Tensor, error) {
	if t == nil || t.handle == nil {
		return nil, fmt.Errorf("spiraltorch: tensor handle is nil")
	}
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("spiraltorch: reshape dimensions must be non-negative")
	}
	finish := startForeignCall("tensor.reshape")
	ptr := C.spiraltorch_tensor_reshape(t.handle, C.size_t(rows), C.size_t(cols))
	finish(ptr != nil)
	return wrapTensor(ptr, "tensor_reshape")
}
