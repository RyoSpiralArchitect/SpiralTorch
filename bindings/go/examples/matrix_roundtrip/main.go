package main

import (
	"fmt"
	"log"

	spiraltorch "github.com/spiraltorch/spiraltorch-go"
)

func main() {
	matrix := [][]float32{
		{1, 2, 3},
		{4, 5, 6},
	}

	tensor, err := spiraltorch.NewTensorFromMatrix(matrix)
	if err != nil {
		log.Fatalf("unable to construct tensor: %v", err)
	}
	defer tensor.Close()

	rows, cols, err := tensor.Shape()
	if err != nil {
		log.Fatalf("unable to query shape: %v", err)
	}

	fmt.Printf("constructed tensor with shape %dx%d\n", rows, cols)

	recovered, err := tensor.ToMatrix()
	if err != nil {
		log.Fatalf("unable to decode tensor: %v", err)
	}

	// Mutating the recovered matrix does not affect the tensor, proving the
	// helper returns independent copies.
	recovered[0][0] = 42

	fmt.Printf("original matrix: %v\n", matrix)
	fmt.Printf("recovered matrix (mutated): %v\n", recovered)

	columns, err := tensor.Columns()
	if err != nil {
		log.Fatalf("unable to decode columns: %v", err)
	}
	fmt.Printf("column copies: %v\n", columns)

	firstColumn, err := tensor.Column(0)
	if err != nil {
		log.Fatalf("unable to extract first column: %v", err)
	}
	secondRow, err := tensor.Row(1)
	if err != nil {
		log.Fatalf("unable to extract second row: %v", err)
	}
	fmt.Printf("first column slice: %v\n", firstColumn)
	fmt.Printf("second row slice: %v\n", secondRow)

	rebuilt, err := spiraltorch.NewTensorFromColumns(columns)
	if err != nil {
		log.Fatalf("unable to rebuild from columns: %v", err)
	}
	defer rebuilt.Close()

	echo, err := rebuilt.ToMatrix()
	if err != nil {
		log.Fatalf("unable to decode rebuilt tensor: %v", err)
	}
	fmt.Printf("round-trip via columns: %v\n", echo)
}
