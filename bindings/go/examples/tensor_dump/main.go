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

	data, err := tensor.Data()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Tensor elements: %v\n", data)
}
