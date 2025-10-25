// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

package spiraltorch

import (
	"fmt"
	goruntime "runtime"
	"sync"
	"sync/atomic"
)

// TensorPair represents a pair of tensors that should be consumed together by a
// binary runtime operation such as matrix multiplication or element-wise
// arithmetic.
type TensorPair struct {
	LHS *Tensor
	RHS *Tensor
}

type runtimeBinary func(*Runtime, *Tensor, *Tensor) (*Tensor, error)

func (r *Runtime) parallelBinary(
	pairs []TensorPair,
	concurrency int,
	label string,
	op runtimeBinary,
) ([]*Tensor, error) {
	if len(pairs) == 0 {
		return []*Tensor{}, nil
	}

	for i, pair := range pairs {
		if pair.LHS == nil {
			return nil, fmt.Errorf("spiraltorch: %s pair %d lhs tensor is nil", label, i)
		}
		if pair.RHS == nil {
			return nil, fmt.Errorf("spiraltorch: %s pair %d rhs tensor is nil", label, i)
		}
	}

	workers := concurrency
	if workers <= 0 {
		if count, err := r.WorkerCount(); err == nil && count > 0 {
			workers = count
		} else {
			workers = goruntime.GOMAXPROCS(0)
		}
	}
	if workers <= 0 {
		workers = 1
	}
	if workers > len(pairs) {
		workers = len(pairs)
	}

	type job struct {
		index int
		pair  TensorPair
	}

	jobs := make(chan job, len(pairs))
	results := make([]*Tensor, len(pairs))

	var firstErr atomic.Value
	var cancelled atomic.Bool
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()
		for job := range jobs {
			if cancelled.Load() {
				continue
			}
			result, err := op(r, job.pair.LHS, job.pair.RHS)
			if err != nil {
				if cancelled.CompareAndSwap(false, true) {
					firstErr.Store(err)
				}
				return
			}
			results[job.index] = result
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()
	}

	for i, pair := range pairs {
		if cancelled.Load() {
			break
		}
		jobs <- job{index: i, pair: pair}
	}
	close(jobs)
	wg.Wait()

	if errVal := firstErr.Load(); errVal != nil {
		err := errVal.(error)
		for _, tensor := range results {
			if tensor != nil {
				tensor.Close()
			}
		}
		return nil, err
	}

	return results, nil
}

// ParallelMatmul schedules multiple matrix multiplications on the runtime and
// executes them concurrently. When concurrency <= 0 the worker count defaults to
// the runtime's cooperative pool size. Each returned tensor remains owned by the
// caller and must be closed once no longer needed.
func (r *Runtime) ParallelMatmul(pairs []TensorPair, concurrency int) ([]*Tensor, error) {
	return r.parallelBinary(pairs, concurrency, "parallel_matmul", func(rt *Runtime, lhs, rhs *Tensor) (*Tensor, error) {
		return rt.Matmul(lhs, rhs)
	})
}

// ParallelAdd schedules element-wise additions for each tensor pair and runs
// them concurrently on the runtime. Results are returned in the same order as
// the inputs.
func (r *Runtime) ParallelAdd(pairs []TensorPair, concurrency int) ([]*Tensor, error) {
	return r.parallelBinary(pairs, concurrency, "parallel_add", func(rt *Runtime, lhs, rhs *Tensor) (*Tensor, error) {
		return rt.Add(lhs, rhs)
	})
}

// ParallelHadamard schedules element-wise products for each tensor pair and
// executes them concurrently.
func (r *Runtime) ParallelHadamard(pairs []TensorPair, concurrency int) ([]*Tensor, error) {
	return r.parallelBinary(pairs, concurrency, "parallel_hadamard", func(rt *Runtime, lhs, rhs *Tensor) (*Tensor, error) {
		return rt.Hadamard(lhs, rhs)
	})
}
