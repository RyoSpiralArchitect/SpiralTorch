// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package main

import (
	"fmt"
	"log"

	spiraltorch "github.com/spiraltorch/spiraltorch-go"
)

func mustTensor(rows, cols int, data []float32) *spiraltorch.Tensor {
	tensor, err := spiraltorch.NewTensorFromDense(rows, cols, data)
	if err != nil {
		log.Fatalf("NewTensorFromDense failed: %v", err)
	}
	return tensor
}

func main() {
	runtime, err := spiraltorch.NewRuntime(0, "go-parallel")
	if err != nil {
		log.Fatalf("NewRuntime failed: %v", err)
	}
	defer runtime.Close()

	pairs := []spiraltorch.TensorPair{
		{LHS: mustTensor(2, 2, []float32{1, 2, 3, 4}), RHS: mustTensor(2, 2, []float32{5, 6, 7, 8})},
		{LHS: mustTensor(2, 3, []float32{1, 0, 2, 2, 1, 3}), RHS: mustTensor(3, 2, []float32{2, 1, 0, 1, 4, 3})},
		{LHS: mustTensor(2, 2, []float32{0, 1, 1, 0}), RHS: mustTensor(2, 2, []float32{2, 2, 2, 2})},
	}
	defer func() {
		for _, pair := range pairs {
			pair.LHS.Close()
			pair.RHS.Close()
		}
	}()

	results, err := runtime.ParallelMatmul(pairs, 0)
	if err != nil {
		log.Fatalf("ParallelMatmul failed: %v", err)
	}
	defer func() {
		for _, tensor := range results {
			tensor.Close()
		}
	}()

	for i, tensor := range results {
		matrix, err := tensor.ToMatrix()
		if err != nil {
			log.Fatalf("ToMatrix failed: %v", err)
		}
		fmt.Printf("result %d:\n", i)
		for _, row := range matrix {
			fmt.Printf("  %v\n", row)
		}
	}

	sumPairs := []spiraltorch.TensorPair{
		{LHS: mustTensor(2, 2, []float32{1, 1, 1, 1}), RHS: mustTensor(2, 2, []float32{2, 2, 2, 2})},
		{LHS: mustTensor(2, 2, []float32{0, 3, 6, 9}), RHS: mustTensor(2, 2, []float32{9, 6, 3, 0})},
	}
	defer func() {
		for _, pair := range sumPairs {
			pair.LHS.Close()
			pair.RHS.Close()
		}
	}()

	sums, err := runtime.ParallelAdd(sumPairs, 0)
	if err != nil {
		log.Fatalf("ParallelAdd failed: %v", err)
	}
	defer func() {
		for _, tensor := range sums {
			tensor.Close()
		}
	}()

	for i, tensor := range sums {
		matrix, err := tensor.ToMatrix()
		if err != nil {
			log.Fatalf("ToMatrix for sum failed: %v", err)
		}
		fmt.Printf("sum %d:\n", i)
		for _, row := range matrix {
			fmt.Printf("  %v\n", row)
		}
	}
}
