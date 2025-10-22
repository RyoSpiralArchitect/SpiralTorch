# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 SpiralTorch Contributors
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

"""
    tensor_from_rows(rows)

Construct a [`Tensor`](@ref) from a vector of equally sized row vectors. Input
values are converted to `Float32` before being transferred to the runtime.
"""
function tensor_from_rows(rows::AbstractVector{<:AbstractVector})
    nrows = length(rows)
    if nrows == 0
        return Tensor(0, 0)
    end

    ncols = length(rows[1])
    for (idx, row) in enumerate(rows)
        if length(row) != ncols
            throw(ArgumentError("row $idx has length $(length(row)) but expected $ncols"))
        end
    end

    if ncols == 0
        return Tensor(nrows, 0)
    end

    buffer = Matrix{Float32}(undef, nrows, ncols)
    @inbounds for (i, row) in enumerate(rows)
        buffer[i, :] = Float32.(row)
    end
    return Tensor(buffer)
end

"""
    tensor_from_columns(columns)

Construct a [`Tensor`](@ref) from a vector of column vectors. All columns must
be the same length.
"""
function tensor_from_columns(columns::AbstractVector{<:AbstractVector})
    ncols = length(columns)
    if ncols == 0
        return Tensor(0, 0)
    end

    nrows = length(columns[1])
    for (idx, column) in enumerate(columns)
        if length(column) != nrows
            throw(ArgumentError("column $idx has length $(length(column)) but expected $nrows"))
        end
    end

    if nrows == 0
        return Tensor(0, ncols)
    end

    buffer = Matrix{Float32}(undef, nrows, ncols)
    @inbounds for (j, column) in enumerate(columns)
        buffer[:, j] = Float32.(column)
    end
    return Tensor(buffer)
end

"""
    tensor_rows(tensor)

Return a vector of row copies from the provided [`Tensor`](@ref). Each row is a
`Vector{Float32}` that can be mutated independently.
"""
function tensor_rows(tensor::Tensor)
    data = to_array(tensor)
    nrows, _ = size(data)
    rows = Vector{Vector{Float32}}(undef, nrows)
    @inbounds for i in 1:nrows
        rows[i] = Vector{Float32}(data[i, :])
    end
    return rows
end

"""
    tensor_columns(tensor)

Return a vector of column copies from the provided [`Tensor`](@ref). Each column
is a `Vector{Float32}`.
"""
function tensor_columns(tensor::Tensor)
    data = to_array(tensor)
    _, ncols = size(data)
    columns = Vector{Vector{Float32}}(undef, ncols)
    @inbounds for j in 1:ncols
        columns[j] = Vector{Float32}(data[:, j])
    end
    return columns
end

"""
    tensor_vector(tensor)

Return a flattened `Vector{Float32}` copy of the tensor contents in row-major
order.
"""
function tensor_vector(tensor::Tensor)
    array = to_array(tensor)
    return Vector{Float32}(reshape(array, :))
end

