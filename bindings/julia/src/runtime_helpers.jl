# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 SpiralTorch Contributors
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

"""
    with_runtime(f; worker_threads=0, thread_name=nothing)

Construct a [`Runtime`](@ref) using the provided configuration, execute `f` with
that runtime, and guarantee the runtime is shut down afterwards. This helper
simplifies short-lived interactions with the SpiralTorch golden runtime from
Julia code.
"""
function with_runtime(f::Function; worker_threads::Integer=0, thread_name::Union{Nothing,String}=nothing)
    runtime = Runtime(; worker_threads=worker_threads, thread_name=thread_name)
    try
        return f(runtime)
    finally
        close(runtime)
    end
end

const _VecOrMatReal = Union{AbstractMatrix{<:Real}, AbstractVector{<:Real}}

function _as_tensor(data::AbstractMatrix{<:Real})
    return Tensor(data)
end

function _materialize(result::Tensor, materialize::Bool, materialize_into)
    if materialize_into !== nothing
        if isa(materialize_into, AbstractVector)
            expected = length(result)
            if length(materialize_into) != expected
                throw(ArgumentError("materialize_into has length $(length(materialize_into)) but expected $expected"))
            end
        else
            expected = size(result)
            if size(materialize_into) != expected
                throw(ArgumentError("materialize_into has size $(size(materialize_into)) but expected $expected"))
            end
        end
        Base.copyto!(materialize_into, result)
        return materialize_into
    end
    return materialize ? to_array(result) : result
end

"""
    add(runtime, lhs, rhs)

Convert the provided matrices to [`Tensor`](@ref) values and schedule the
addition on `runtime`. The resulting tensor remains managed by the SpiralTorch
runtime and can be converted back to a Julia array with [`to_array`](@ref).
Set `materialize=true` to receive the result immediately as a `Matrix{Float32}`
or pass an existing array (matrix or vector) via `materialize_into` to copy the
data without allocating.
"""
function add(
    runtime::Runtime,
    lhs::AbstractMatrix{<:Real},
    rhs::AbstractMatrix{<:Real};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = add(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize, materialize_into)
end

"""
    sub(runtime, lhs, rhs)

Schedule subtraction on the runtime after promoting the inputs to
[`Tensor`](@ref) values. Set `materialize=true` to obtain a `Matrix{Float32}`
directly or pass a preallocated array (matrix or vector) via `materialize_into`.
"""
function sub(
    runtime::Runtime,
    lhs::AbstractMatrix{<:Real},
    rhs::AbstractMatrix{<:Real};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = sub(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize, materialize_into)
end

"""
    hadamard(runtime, lhs, rhs)

Compute the element-wise product of the two matrices via the runtime. The
result is a [`Tensor`](@ref). Use `materialize=true` to convert it to a
`Matrix{Float32}` automatically or supply `materialize_into` (matrix or vector)
to reuse an existing workspace.
"""
function hadamard(
    runtime::Runtime,
    lhs::AbstractMatrix{<:Real},
    rhs::AbstractMatrix{<:Real};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = hadamard(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize, materialize_into)
end

"""
    matmul(runtime, lhs, rhs)

Multiply the two matrices using the cooperative golden runtime without needing
manual tensor construction. Pass `materialize=true` to collect the product as a
`Matrix{Float32}` or set `materialize_into` to fill a preallocated array (matrix
or vector).
"""
function matmul(
    runtime::Runtime,
    lhs::AbstractMatrix{<:Real},
    rhs::AbstractMatrix{<:Real};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = matmul(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize, materialize_into)
end

"""
    scale(runtime, data, value)

Scale the matrix `data` by `value` using the runtime. Set `materialize=true` to
obtain the scaled values as a `Matrix{Float32}` in one step or supply
`materialize_into` (matrix or vector) to reuse storage.
"""
function scale(
    runtime::Runtime,
    data::AbstractMatrix{<:Real},
    value::Real;
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = scale(runtime, _as_tensor(data), value)
    return _materialize(result, materialize, materialize_into)
end

"""
    transpose_tensor(runtime, data)

Transpose the matrix `data` by promoting it to a [`Tensor`](@ref) and executing
on the runtime. `materialize=true` returns the transposed `Matrix{Float32}`
directly, while `materialize_into` lets you fill an existing destination (matrix
or vector).
"""
function transpose_tensor(
    runtime::Runtime,
    data::AbstractMatrix{<:Real};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = transpose_tensor(runtime, _as_tensor(data))
    return _materialize(result, materialize, materialize_into)
end

"""
    reshape_tensor(runtime, data, rows, cols)

Reshape the provided matrix using the runtime. The number of elements must
remain unchanged. When `materialize=true` the reshaped data is returned as a
`Matrix{Float32}`. Provide `materialize_into` (matrix or vector) to copy the
reshaped values into preallocated storage.
"""
function reshape_tensor(
    runtime::Runtime,
    data::AbstractMatrix{<:Real},
    rows::Integer,
    cols::Integer;
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = reshape_tensor(runtime, _as_tensor(data), rows, cols)
    return _materialize(result, materialize, materialize_into)
end

"""
    reshape_tensor(runtime, data, dims)

Reshape the matrix with dimensions provided as a tuple, delegating to
[`reshape_tensor`](@ref). `materialize=true` returns the reshaped matrix data
immediately, and `materialize_into` copies the result into an existing array
(matrix or vector).
"""
function reshape_tensor(
    runtime::Runtime,
    data::AbstractMatrix{<:Real},
    dims::Tuple{Integer,Integer};
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    return reshape_tensor(
        runtime,
        data,
        dims[1],
        dims[2];
        materialize=materialize,
        materialize_into=materialize_into,
    )
end

"""
    random_uniform(runtime, dims, min, max; seed=nothing)

Convenience method that dispatches to [`random_uniform`](@ref) using a tuple of
dimensions. Set `materialize=true` to obtain a sampled `Matrix{Float32}` or
pass `materialize_into` (matrix or vector) to reuse preallocated storage.
"""
function random_uniform(
    runtime::Runtime,
    dims::Tuple{Integer,Integer},
    min::Real,
    max::Real;
    seed::Union{Nothing,Integer}=nothing,
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = random_uniform(runtime, dims[1], dims[2], min, max; seed=seed)
    return _materialize(result, materialize, materialize_into)
end

"""
    random_normal(runtime, dims, mean, std; seed=nothing)

Tuple-friendly wrapper for [`random_normal`](@ref) when requesting tensors from
the runtime. Use `materialize=true` to receive the samples as a
`Matrix{Float32}` or provide `materialize_into` (matrix or vector) to copy into
existing storage.
"""
function random_normal(
    runtime::Runtime,
    dims::Tuple{Integer,Integer},
    mean::Real,
    std::Real;
    seed::Union{Nothing,Integer}=nothing,
    materialize::Bool=false,
    materialize_into::Union{Nothing,_VecOrMatReal}=nothing,
)
    result = random_normal(runtime, dims[1], dims[2], mean, std; seed=seed)
    return _materialize(result, materialize, materialize_into)
end

function _normalize_thread_count(requested::Integer, tasks::Integer)
    count = requested <= 0 ? Threads.nthreads() : Int(requested)
    if tasks <= 0
        return 0
    end
    return max(1, min(count, tasks))
end

function _parallel_binary(runtime::Runtime, lhs_list, rhs_list, op::Function; threads::Integer)
    count = length(lhs_list)
    if length(rhs_list) != count
        throw(ArgumentError("lhs_list and rhs_list must have the same length"))
    end
    if count == 0
        return Tensor[]
    end

    limit = _normalize_thread_count(threads, count)
    results = Vector{Tensor}(undef, count)
    err_ref = Ref{Any}(nothing)
    bt_ref = Ref{Any}(nothing)
    semaphore = Base.Semaphore(limit)

    Threads.@sync begin
        for idx in eachindex(lhs_list)
            lhs = lhs_list[idx]
            rhs = rhs_list[idx]
            Threads.@spawn begin
                Base.acquire(semaphore)
                try
                    if err_ref[] !== nothing
                        return
                    end
                    results[idx] = op(lhs, rhs)
                catch err
                    if err_ref[] === nothing
                        err_ref[] = err
                        bt_ref[] = Base.catch_backtrace()
                    end
                finally
                    Base.release(semaphore)
                end
            end
        end
    end

    if err_ref[] !== nothing
        Base.throw(err_ref[], bt_ref[])
    end

    return results
end

function _parallel_unary(n::Integer, exec::Function; threads::Integer)
    if n == 0
        return Tensor[]
    end

    limit = _normalize_thread_count(threads, n)
    results = Vector{Tensor}(undef, n)
    err_ref = Ref{Any}(nothing)
    bt_ref = Ref{Any}(nothing)
    semaphore = Base.Semaphore(limit)

    Threads.@sync begin
        for idx in 1:n
            Threads.@spawn begin
                Base.acquire(semaphore)
                try
                    if err_ref[] !== nothing
                        return
                    end
                    results[idx] = exec(idx)
                catch err
                    if err_ref[] === nothing
                        err_ref[] = err
                        bt_ref[] = Base.catch_backtrace()
                    end
                finally
                    Base.release(semaphore)
                end
            end
        end
    end

    if err_ref[] !== nothing
        Base.throw(err_ref[], bt_ref[])
    end

    return results
end

function _materialize_batch(results::Vector{Tensor}, materialize::Bool)
    if !materialize
        return results
    end
    materialized = Vector{Matrix{Float32}}(undef, length(results))
    @inbounds for (idx, tensor) in enumerate(results)
        materialized[idx] = to_array(tensor)
    end
    return materialized
end

"""
    parallel_add(runtime, lhs_list, rhs_list; threads=Threads.nthreads(), materialize=false)

Schedule element-wise additions for each pair in `lhs_list` and `rhs_list` on
the cooperative runtime. Work is distributed across Julia threads (bounded by
`threads`) so multiple additions can complete concurrently. Set
`materialize=true` to receive a vector of `Matrix{Float32}` results.
"""
function parallel_add(
    runtime::Runtime,
    lhs_list::AbstractVector,
    rhs_list::AbstractVector;
    threads::Integer=Threads.nthreads(),
    materialize::Bool=false,
)
    results = _parallel_binary(runtime, lhs_list, rhs_list, (lhs, rhs) -> add(runtime, lhs, rhs); threads=threads)
    return _materialize_batch(results, materialize)
end

"""
    parallel_hadamard(runtime, lhs_list, rhs_list; threads=Threads.nthreads(), materialize=false)

Execute Hadamard products for each pair of tensors or matrices concurrently.
When `materialize=true`, the resulting tensors are copied into Julia
`Matrix{Float32}` values.
"""
function parallel_hadamard(
    runtime::Runtime,
    lhs_list::AbstractVector,
    rhs_list::AbstractVector;
    threads::Integer=Threads.nthreads(),
    materialize::Bool=false,
)
    results = _parallel_binary(runtime, lhs_list, rhs_list, (lhs, rhs) -> hadamard(runtime, lhs, rhs); threads=threads)
    return _materialize_batch(results, materialize)
end

"""
    parallel_matmul(runtime, lhs_list, rhs_list; threads=Threads.nthreads(), materialize=false)

Submit batched matrix multiplications to the runtime using Julia threads to
drive each request. Results preserve input ordering and can be materialised to
`Matrix{Float32}` values by setting `materialize=true`.
"""
function parallel_matmul(
    runtime::Runtime,
    lhs_list::AbstractVector,
    rhs_list::AbstractVector;
    threads::Integer=Threads.nthreads(),
    materialize::Bool=false,
)
    results = _parallel_binary(runtime, lhs_list, rhs_list, (lhs, rhs) -> matmul(runtime, lhs, rhs); threads=threads)
    return _materialize_batch(results, materialize)
end

"""
    parallel_scale(runtime, data_list, values; threads=Threads.nthreads(), materialize=false)

Scale each matrix or tensor in `data_list` by the corresponding scalar in
`values`, executing the work in parallel. Use `materialize=true` to copy the
results back into Julia arrays immediately.
"""
function parallel_scale(
    runtime::Runtime,
    data_list::AbstractVector,
    values::AbstractVector{<:Real};
    threads::Integer=Threads.nthreads(),
    materialize::Bool=false,
)
    if length(data_list) != length(values)
        throw(ArgumentError("data_list and values must be the same length"))
    end
    results = _parallel_unary(length(data_list), idx -> scale(runtime, data_list[idx], values[idx]); threads=threads)
    return _materialize_batch(results, materialize)
end
