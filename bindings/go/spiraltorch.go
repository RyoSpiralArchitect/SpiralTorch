// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -lspiraltorch_sys
#include <stdbool.h>
#include <stddef.h>
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
*/
import "C"

import (
	"fmt"
	"runtime"
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
	handle unsafe.Pointer
}

// NewZerosTensor constructs a tensor of the requested shape initialised with zeros.
func NewZerosTensor(rows, cols int) (*Tensor, error) {
	ptr := C.spiraltorch_tensor_zeros(C.size_t(rows), C.size_t(cols))
	if ptr == nil {
		err := lastError()
		if err == "" {
			err = "unknown error"
		}
		return nil, fmt.Errorf("spiraltorch: %s", err)
	}
	tensor := &Tensor{handle: ptr}
	runtime.SetFinalizer(tensor, func(t *Tensor) {
		t.Close()
	})
	return tensor, nil
}

// NewTensorFromDense constructs a tensor from the provided row-major data slice.
func NewTensorFromDense(rows, cols int, data []float32) (*Tensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("spiraltorch: data slice cannot be empty")
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
	if ptr == nil {
		err := lastError()
		if err == "" {
			err = "unknown error"
		}
		return nil, fmt.Errorf("spiraltorch: %s", err)
	}
	tensor := &Tensor{handle: ptr}
	runtime.SetFinalizer(tensor, func(t *Tensor) {
		t.Close()
	})
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
