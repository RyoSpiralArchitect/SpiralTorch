package main

import (
	"fmt"

	spiraltorch "github.com/spiraltorch/spiraltorch-go"
)

func main() {
	version := spiraltorch.Version()
	fmt.Printf("SpiralTorch runtime version: %s\n", version)

	tensor, err := spiraltorch.NewTensorFromDense(2, 2, []float32{1, 2, 3, 4})
	if err != nil {
		panic(err)
	}
	defer tensor.Close()

	other, err := spiraltorch.NewTensorFromDense(2, 2, []float32{5, 6, 7, 8})
	if err != nil {
		panic(err)
	}
	defer other.Close()

	data, err := tensor.Data()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Tensor elements: %v\n", data)

	sum, err := tensor.Add(other)
	if err != nil {
		panic(err)
	}
	defer sum.Close()
	sumData, err := sum.Data()
	if err != nil {
		panic(err)
	}
	fmt.Printf("tensor + other: %v\n", sumData)

	scaled, err := tensor.Scale(0.5)
	if err != nil {
		panic(err)
	}
	defer scaled.Close()
	scaledData, err := scaled.Data()
	if err != nil {
		panic(err)
	}
	fmt.Printf("tensor * 0.5: %v\n", scaledData)

	product, err := tensor.Matmul(other)
	if err != nil {
		panic(err)
	}
	defer product.Close()
	productData, err := product.Data()
	if err != nil {
		panic(err)
	}
	fmt.Printf("tensor @ other: %v\n", productData)
}
