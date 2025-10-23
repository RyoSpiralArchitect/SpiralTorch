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

function _as_tensor(data::AbstractMatrix{<:Real})
    return Tensor(data)
end

function _materialize(result::Tensor, materialize::Bool)
    return materialize ? to_array(result) : result
end

"""
    add(runtime, lhs, rhs)

Convert the provided matrices to [`Tensor`](@ref) values and schedule the
addition on `runtime`. The resulting tensor remains managed by the SpiralTorch
runtime and can be converted back to a Julia array with [`to_array`](@ref).
Set `materialize=true` to receive the result immediately as a `Matrix{Float32}`.
"""
function add(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real}; materialize::Bool=false)
    result = add(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize)
end

"""
    sub(runtime, lhs, rhs)

Schedule subtraction on the runtime after promoting the inputs to
[`Tensor`](@ref) values. Set `materialize=true` to obtain a `Matrix{Float32}`
directly.
"""
function sub(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real}; materialize::Bool=false)
    result = sub(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize)
end

"""
    hadamard(runtime, lhs, rhs)

Compute the element-wise product of the two matrices via the runtime. The
result is a [`Tensor`](@ref). Use `materialize=true` to convert it to a
`Matrix{Float32}` automatically.
"""
function hadamard(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real}; materialize::Bool=false)
    result = hadamard(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize)
end

"""
    matmul(runtime, lhs, rhs)

Multiply the two matrices using the cooperative golden runtime without needing
manual tensor construction. Pass `materialize=true` to collect the product as a
`Matrix{Float32}`.
"""
function matmul(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real}; materialize::Bool=false)
    result = matmul(runtime, _as_tensor(lhs), _as_tensor(rhs))
    return _materialize(result, materialize)
end

"""
    scale(runtime, data, value)

Scale the matrix `data` by `value` using the runtime. Set `materialize=true` to
obtain the scaled values as a `Matrix{Float32}` in one step.
"""
function scale(runtime::Runtime, data::AbstractMatrix{<:Real}, value::Real; materialize::Bool=false)
    result = scale(runtime, _as_tensor(data), value)
    return _materialize(result, materialize)
end

"""
    transpose_tensor(runtime, data)

Transpose the matrix `data` by promoting it to a [`Tensor`](@ref) and executing
on the runtime. `materialize=true` returns the transposed `Matrix{Float32}`
directly.
"""
function transpose_tensor(runtime::Runtime, data::AbstractMatrix{<:Real}; materialize::Bool=false)
    result = transpose_tensor(runtime, _as_tensor(data))
    return _materialize(result, materialize)
end

"""
    reshape_tensor(runtime, data, rows, cols)

Reshape the provided matrix using the runtime. The number of elements must
remain unchanged. When `materialize=true` the reshaped data is returned as a
`Matrix{Float32}`.
"""
function reshape_tensor(runtime::Runtime, data::AbstractMatrix{<:Real}, rows::Integer, cols::Integer; materialize::Bool=false)
    result = reshape_tensor(runtime, _as_tensor(data), rows, cols)
    return _materialize(result, materialize)
end

"""
    reshape_tensor(runtime, data, dims)

Reshape the matrix with dimensions provided as a tuple, delegating to
[`reshape_tensor`](@ref). `materialize=true` returns the reshaped matrix data
immediately.
"""
function reshape_tensor(runtime::Runtime, data::AbstractMatrix{<:Real}, dims::Tuple{Integer,Integer}; materialize::Bool=false)
    return reshape_tensor(runtime, data, dims[1], dims[2]; materialize=materialize)
end

"""
    random_uniform(runtime, dims, min, max; seed=nothing)

Convenience method that dispatches to [`random_uniform`](@ref) using a tuple of
dimensions. Set `materialize=true` to obtain a sampled `Matrix{Float32}`.
"""
function random_uniform(runtime::Runtime, dims::Tuple{Integer,Integer}, min::Real, max::Real; seed::Union{Nothing,Integer}=nothing, materialize::Bool=false)
    result = random_uniform(runtime, dims[1], dims[2], min, max; seed=seed)
    return _materialize(result, materialize)
end

"""
    random_normal(runtime, dims, mean, std; seed=nothing)

Tuple-friendly wrapper for [`random_normal`](@ref) when requesting tensors from
the runtime. Use `materialize=true` to receive the samples as a
`Matrix{Float32}`.
"""
function random_normal(runtime::Runtime, dims::Tuple{Integer,Integer}, mean::Real, std::Real; seed::Union{Nothing,Integer}=nothing, materialize::Bool=false)
    result = random_normal(runtime, dims[1], dims[2], mean, std; seed=seed)
    return _materialize(result, materialize)
end
