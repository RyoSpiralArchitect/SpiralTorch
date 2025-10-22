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

"""
    add(runtime, lhs, rhs)

Convert the provided matrices to [`Tensor`](@ref) values and schedule the
addition on `runtime`. The resulting tensor remains managed by the SpiralTorch
runtime and can be converted back to a Julia array with [`to_array`](@ref).
"""
function add(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real})
    return add(runtime, _as_tensor(lhs), _as_tensor(rhs))
end

"""
    sub(runtime, lhs, rhs)

Schedule subtraction on the runtime after promoting the inputs to
[`Tensor`](@ref) values.
"""
function sub(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real})
    return sub(runtime, _as_tensor(lhs), _as_tensor(rhs))
end

"""
    hadamard(runtime, lhs, rhs)

Compute the element-wise product of the two matrices via the runtime. The
result is a [`Tensor`](@ref).
"""
function hadamard(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real})
    return hadamard(runtime, _as_tensor(lhs), _as_tensor(rhs))
end

"""
    matmul(runtime, lhs, rhs)

Multiply the two matrices using the cooperative golden runtime without needing
manual tensor construction.
"""
function matmul(runtime::Runtime, lhs::AbstractMatrix{<:Real}, rhs::AbstractMatrix{<:Real})
    return matmul(runtime, _as_tensor(lhs), _as_tensor(rhs))
end

"""
    scale(runtime, data, value)

Scale the matrix `data` by `value` using the runtime.
"""
function scale(runtime::Runtime, data::AbstractMatrix{<:Real}, value::Real)
    return scale(runtime, _as_tensor(data), value)
end

"""
    transpose_tensor(runtime, data)

Transpose the matrix `data` by promoting it to a [`Tensor`](@ref) and executing
on the runtime.
"""
function transpose_tensor(runtime::Runtime, data::AbstractMatrix{<:Real})
    return transpose_tensor(runtime, _as_tensor(data))
end

"""
    reshape_tensor(runtime, data, rows, cols)

Reshape the provided matrix using the runtime. The number of elements must
remain unchanged.
"""
function reshape_tensor(runtime::Runtime, data::AbstractMatrix{<:Real}, rows::Integer, cols::Integer)
    return reshape_tensor(runtime, _as_tensor(data), rows, cols)
end

"""
    reshape_tensor(runtime, data, dims)

Reshape the matrix with dimensions provided as a tuple, delegating to
[`reshape_tensor`](@ref).
"""
function reshape_tensor(runtime::Runtime, data::AbstractMatrix{<:Real}, dims::Tuple{Integer,Integer})
    return reshape_tensor(runtime, data, dims[1], dims[2])
end

"""
    random_uniform(runtime, dims, min, max; seed=nothing)

Convenience method that dispatches to [`random_uniform`](@ref) using a tuple of
dimensions.
"""
function random_uniform(runtime::Runtime, dims::Tuple{Integer,Integer}, min::Real, max::Real; seed::Union{Nothing,Integer}=nothing)
    return random_uniform(runtime, dims[1], dims[2], min, max; seed=seed)
end

"""
    random_normal(runtime, dims, mean, std; seed=nothing)

Tuple-friendly wrapper for [`random_normal`](@ref) when requesting tensors from
the runtime.
"""
function random_normal(runtime::Runtime, dims::Tuple{Integer,Integer}, mean::Real, std::Real; seed::Union{Nothing,Integer}=nothing)
    return random_normal(runtime, dims[1], dims[2], mean, std; seed=seed)
end
