module SpiralTorch

using Libdl
using Base: getpid, time_ns, catch_backtrace

const _lib_ref = Ref{Ptr{Cvoid}}(C_NULL)
const _foreign_client = Ref{Union{Nothing, String}}(nothing)
const _FOREIGN_CAPABILITIES = String[
    "tensor.zeros",
    "tensor.from_dense",
    "tensor.add",
    "tensor.sub",
    "tensor.matmul",
    "tensor.hadamard",
    "tensor.scale",
    "tensor.transpose",
    "tensor.reshape",
]

function _candidate_paths()
    candidates = String[]
    if haskey(ENV, "SPIRALTORCH_SYS_LIBRARY")
        push!(candidates, ENV["SPIRALTORCH_SYS_LIBRARY"])
    end
    push!(candidates, joinpath(@__DIR__, "..", "deps", "libspiraltorch_sys." * Libdl.dlext))
    push!(candidates, "libspiraltorch_sys." * Libdl.dlext)
    return candidates
end

function load_library!()
    if _lib_ref[] != C_NULL
        return _lib_ref[]
    end
    errors = String[]
    for candidate in _candidate_paths()
        if isempty(candidate)
            continue
        end
        try
            handle = Libdl.dlopen(candidate)
            _lib_ref[] = handle
            return handle
        catch err
            push!(errors, string(candidate, ": ", sprint(showerror, err)))
        end
    end
    error_msg = join(errors, "\n")
    error("Unable to locate libspiraltorch_sys. Set SPIRALTORCH_SYS_LIBRARY to the compiled shared library path.\n" * error_msg)
end

@inline function _lib()
    lib = _lib_ref[]
    return lib == C_NULL ? load_library!() : lib
end

function last_error()
    lib = _lib()
    len = ccall((:spiraltorch_last_error_length, lib), Csize_t, ())
    if len == 0
        return ""
    end
    buffer = Vector{UInt8}(undef, len + 1)
    written = ccall((:spiraltorch_last_error_message, lib), Csize_t, (Ptr{UInt8}, Csize_t), pointer(buffer), length(buffer))
    return String(unsafe_string(pointer(buffer), written))
end

function clear_error!()
    lib = _lib()
    ccall((:spiraltorch_clear_last_error, lib), Cvoid, ())
    return nothing
end

function _register_foreign_runtime()
    lib = _lib()
    runtime_id = "julia-" * string(getpid()) * "-" * string(time_ns())
    capabilities = join(_FOREIGN_CAPABILITIES, ",")
    ok = ccall(
        (:spiraltorch_foreign_register, lib),
        Cuchar,
        (Cstring, Cstring, Cstring, Cstring),
        "julia",
        runtime_id,
        string(VERSION),
        capabilities,
    )
    if ok == 0
        err = last_error()
        clear_error!()
        throw(ErrorException("SpiralTorch.jl failed to register foreign runtime: " * err))
    end
    _foreign_client[] = runtime_id
    return runtime_id
end

@inline function _record_latency(operation::AbstractString, start_ns::UInt64, success::Bool)
    client = _foreign_client[]
    client === nothing && return
    elapsed = UInt64(max(time_ns() - start_ns, 0))
    lib = _lib()
    ok = ccall(
        (:spiraltorch_foreign_record_latency, lib),
        Cuchar,
        (Cstring, Cstring, UInt64, UInt8),
        client,
        operation,
        elapsed,
        success ? UInt8(1) : UInt8(0),
    )
    if ok == 0
        clear_error!()
    end
    return nothing
end

function version()
    lib = _lib()
    len = ccall((:spiraltorch_version, lib), Csize_t, (Ptr{UInt8}, Csize_t), Ptr{UInt8}(C_NULL), 0)
    buffer = Vector{UInt8}(undef, len + 1)
    written = ccall((:spiraltorch_version, lib), Csize_t, (Ptr{UInt8}, Csize_t), pointer(buffer), length(buffer))
    return String(unsafe_string(pointer(buffer), written))
end

mutable struct Tensor
    handle::Ptr{Cvoid}
end

function _register_finalizer!(tensor::Tensor)
    finalizer(tensor) do obj
        if obj.handle != C_NULL
            lib = _lib()
            ccall((:spiraltorch_tensor_free, lib), Cvoid, (Ptr{Cvoid},), obj.handle)
            obj.handle = C_NULL
        end
    end
    return tensor
end

function _wrap_tensor(handle::Ptr{Cvoid}, context::AbstractString)
    if handle == C_NULL
        error("$context failed: " * last_error())
    end
    clear_error!()
    return _register_finalizer!(Tensor(handle))
end

function _require_handle(tensor::Tensor, label)
    if tensor.handle == C_NULL
        error("$(label) handle is null")
    end
    return tensor.handle
end

function Tensor(rows::Integer, cols::Integer)
    lib = _lib()
    start_ns = time_ns()
    handle = ccall((:spiraltorch_tensor_zeros, lib), Ptr{Cvoid}, (Csize_t, Csize_t), rows, cols)
    _record_latency("tensor.zeros", start_ns, handle != C_NULL)
    return _wrap_tensor(handle, "tensor_zeros")
end

function Tensor(data::AbstractMatrix{<:Real})
    mat = Float32.(Array(data))
    rows, cols = size(mat)
    flat = reshape(mat, :)
    lib = _lib()
    start_ns = time_ns()
    handle = ccall((:spiraltorch_tensor_from_dense, lib), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Float32}, Csize_t), rows, cols, pointer(flat), length(flat))
    _record_latency("tensor.from_dense", start_ns, handle != C_NULL)
    return _wrap_tensor(handle, "tensor_from_dense")
end

function shape(tensor::Tensor)
    lib = _lib()
    rows = Ref{Csize_t}(0)
    cols = Ref{Csize_t}(0)
    ok = ccall((:spiraltorch_tensor_shape, lib), Cuchar, (Ptr{Cvoid}, Ptr{Csize_t}, Ptr{Csize_t}), tensor.handle, rows, cols)
    if ok == 0
        error("failed to query tensor shape: " * last_error())
    end
    return (Int(rows[]), Int(cols[]))
end

function elements(tensor::Tensor)
    lib = _lib()
    count = ccall((:spiraltorch_tensor_elements, lib), Csize_t, (Ptr{Cvoid},), tensor.handle)
    if count == 0
        err = last_error()
        if !isempty(err)
            error("failed to query tensor elements: " * err)
        end
    end
    return Int(count)
end

function to_array(tensor::Tensor)
    rows, cols = shape(tensor)
    len = rows * cols
    buffer = Vector{Float32}(undef, len)
    lib = _lib()
    ok = ccall((:spiraltorch_tensor_copy_data, lib), Cuchar, (Ptr{Cvoid}, Ptr{Float32}, Csize_t), tensor.handle, pointer(buffer), len)
    if ok == 0
        error("failed to copy tensor data: " * last_error())
    end
    return reshape(buffer, rows, cols)
end

Base.size(tensor::Tensor) = shape(tensor)
Base.length(tensor::Tensor) = prod(size(tensor))
Base.convert(::Type{Array{Float32,2}}, tensor::Tensor) = to_array(tensor)
Base.convert(::Type{Matrix{Float32}}, tensor::Tensor) = to_array(tensor)

function _binary_tensor_op(fname::Symbol, lhs::Tensor, rhs::Tensor)
    lib = _lib()
    left = _require_handle(lhs, string(fname, " lhs"))
    right = _require_handle(rhs, string(fname, " rhs"))
    op_name = replace(replace(String(fname), "spiraltorch_" => ""), "_" => ".")
    start_ns = time_ns()
    handle = ccall((fname, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), left, right)
    _record_latency(op_name, start_ns, handle != C_NULL)
    return _wrap_tensor(handle, String(fname))
end

function add(lhs::Tensor, rhs::Tensor)
    return _binary_tensor_op(:spiraltorch_tensor_add, lhs, rhs)
end

function sub(lhs::Tensor, rhs::Tensor)
    return _binary_tensor_op(:spiraltorch_tensor_sub, lhs, rhs)
end

function matmul(lhs::Tensor, rhs::Tensor)
    return _binary_tensor_op(:spiraltorch_tensor_matmul, lhs, rhs)
end

function scale(tensor::Tensor, value::Real)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_scale")
    start_ns = time_ns()
    result = ccall((:spiraltorch_tensor_scale, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Cfloat), handle, Float32(value))
    _record_latency("tensor.scale", start_ns, result != C_NULL)
    return _wrap_tensor(result, "tensor_scale")
end

function hadamard(lhs::Tensor, rhs::Tensor)
    return _binary_tensor_op(:spiraltorch_tensor_hadamard, lhs, rhs)
end

function transpose_tensor(tensor::Tensor)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_transpose")
    start_ns = time_ns()
    result = ccall((:spiraltorch_tensor_transpose, lib), Ptr{Cvoid}, (Ptr{Cvoid},), handle)
    _record_latency("tensor.transpose", start_ns, result != C_NULL)
    return _wrap_tensor(result, "tensor_transpose")
end

function reshape_tensor(tensor::Tensor, rows::Integer, cols::Integer)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_reshape")
    start_ns = time_ns()
    result = ccall((:spiraltorch_tensor_reshape, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Csize_t), handle, Csize_t(rows), Csize_t(cols))
    _record_latency("tensor.reshape", start_ns, result != C_NULL)
    return _wrap_tensor(result, "tensor_reshape")
end

Base.:+(lhs::Tensor, rhs::Tensor) = add(lhs, rhs)
Base.:-(lhs::Tensor, rhs::Tensor) = sub(lhs, rhs)
Base.:*(lhs::Tensor, rhs::Tensor) = matmul(lhs, rhs)
Base.:*(tensor::Tensor, value::Real) = scale(tensor, value)
Base.:*(value::Real, tensor::Tensor) = scale(tensor, value)
Base.:.*(lhs::Tensor, rhs::Tensor) = hadamard(lhs, rhs)
Base.transpose(tensor::Tensor) = transpose_tensor(tensor)

function Base.reshape(tensor::Tensor, dims::Integer...)
    if length(dims) != 2
        throw(ArgumentError("SpiralTorch tensors currently support reshape with two dimensions"))
    end
    return reshape_tensor(tensor, dims[1], dims[2])
end

function Base.reshape(tensor::Tensor, dims::Tuple{Vararg{Integer, 2}})
    return reshape_tensor(tensor, dims[1], dims[2])
end

function __init__()
    try
        load_library!()
        _register_foreign_runtime()
    catch err
        @warn "SpiralTorch.jl foreign runtime registration failed" exception=(err, catch_backtrace())
        clear_error!()
    end
end

end # module
