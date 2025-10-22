module SpiralTorch

using Libdl

const _lib_ref = Ref{Ptr{Cvoid}}(C_NULL)

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

mutable struct Runtime
    handle::Ptr{Cvoid}
end

function _register_runtime_finalizer!(runtime::Runtime)
    finalizer(runtime) do obj
        if obj.handle != C_NULL
            lib = _lib()
            ccall((:spiraltorch_runtime_free, lib), Cvoid, (Ptr{Cvoid},), obj.handle)
            obj.handle = C_NULL
        end
    end
    return runtime
end

function _wrap_runtime(handle::Ptr{Cvoid}, context::AbstractString)
    if handle == C_NULL
        error("$context failed: " * last_error())
    end
    clear_error!()
    return _register_runtime_finalizer!(Runtime(handle))
end

function _require_runtime(runtime::Runtime, label)
    if runtime.handle == C_NULL
        error("$(label) runtime handle is null")
    end
    return runtime.handle
end

function _require_handle(tensor::Tensor, label)
    if tensor.handle == C_NULL
        error("$(label) handle is null")
    end
    return tensor.handle
end

function Runtime(; worker_threads::Integer=0, thread_name::Union{Nothing,String}=nothing)
    lib = _lib()
    name_arg = if thread_name === nothing || thread_name == ""
        Ptr{UInt8}(C_NULL)
    else
        Base.cconvert(Cstring, thread_name)
    end
    handle = ccall(
        (:spiraltorch_runtime_new, lib),
        Ptr{Cvoid},
        (Csize_t, Cstring),
        Csize_t(worker_threads),
        name_arg,
    )
    return _wrap_runtime(handle, "runtime_new")
end

function Base.close(runtime::Runtime)
    if runtime.handle == C_NULL
        return nothing
    end
    lib = _lib()
    ccall((:spiraltorch_runtime_free, lib), Cvoid, (Ptr{Cvoid},), runtime.handle)
    runtime.handle = C_NULL
    return nothing
end

function worker_count(runtime::Runtime)
    lib = _lib()
    handle = _require_runtime(runtime, "worker_count")
    count = ccall((:spiraltorch_runtime_worker_count, lib), Csize_t, (Ptr{Cvoid},), handle)
    if count == 0
        err = last_error()
        if !isempty(err)
            error("runtime_worker_count failed: " * err)
        end
    end
    return Int(count)
end

function Tensor(rows::Integer, cols::Integer)
    lib = _lib()
    handle = ccall((:spiraltorch_tensor_zeros, lib), Ptr{Cvoid}, (Csize_t, Csize_t), rows, cols)
    return _wrap_tensor(handle, "tensor_zeros")
end

function Tensor(data::AbstractMatrix{<:Real})
    mat = Float32.(Array(data))
    rows, cols = size(mat)
    flat = reshape(mat, :)
    lib = _lib()
    handle = ccall((:spiraltorch_tensor_from_dense, lib), Ptr{Cvoid}, (Csize_t, Csize_t, Ptr{Float32}, Csize_t), rows, cols, pointer(flat), length(flat))
    return _wrap_tensor(handle, "tensor_from_dense")
end

function random_uniform(rows::Integer, cols::Integer, min::Real, max::Real; seed::Union{Nothing,Integer}=nothing)
    lib = _lib()
    seed_value = seed === nothing ? UInt64(0) : UInt64(seed)
    has_seed = seed === nothing ? UInt8(0) : UInt8(1)
    handle = ccall(
        (:spiraltorch_tensor_random_uniform, lib),
        Ptr{Cvoid},
        (Csize_t, Csize_t, Cfloat, Cfloat, UInt64, Cuchar),
        Csize_t(rows),
        Csize_t(cols),
        Float32(min),
        Float32(max),
        seed_value,
        has_seed,
    )
    return _wrap_tensor(handle, "tensor_random_uniform")
end

function random_normal(rows::Integer, cols::Integer, mean::Real, std::Real; seed::Union{Nothing,Integer}=nothing)
    lib = _lib()
    seed_value = seed === nothing ? UInt64(0) : UInt64(seed)
    has_seed = seed === nothing ? UInt8(0) : UInt8(1)
    handle = ccall(
        (:spiraltorch_tensor_random_normal, lib),
        Ptr{Cvoid},
        (Csize_t, Csize_t, Cfloat, Cfloat, UInt64, Cuchar),
        Csize_t(rows),
        Csize_t(cols),
        Float32(mean),
        Float32(std),
        seed_value,
        has_seed,
    )
    return _wrap_tensor(handle, "tensor_random_normal")
end

function random_uniform(runtime::Runtime, rows::Integer, cols::Integer, min::Real, max::Real; seed::Union{Nothing,Integer}=nothing)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_random_uniform")
    seed_value = seed === nothing ? UInt64(0) : UInt64(seed)
    has_seed = seed === nothing ? UInt8(0) : UInt8(1)
    result = ccall(
        (:spiraltorch_runtime_tensor_random_uniform, lib),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Csize_t, Cfloat, Cfloat, UInt64, Cuchar),
        handle,
        Csize_t(rows),
        Csize_t(cols),
        Float32(min),
        Float32(max),
        seed_value,
        has_seed,
    )
    return _wrap_tensor(result, "runtime_tensor_random_uniform")
end

function random_normal(runtime::Runtime, rows::Integer, cols::Integer, mean::Real, std::Real; seed::Union{Nothing,Integer}=nothing)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_random_normal")
    seed_value = seed === nothing ? UInt64(0) : UInt64(seed)
    has_seed = seed === nothing ? UInt8(0) : UInt8(1)
    result = ccall(
        (:spiraltorch_runtime_tensor_random_normal, lib),
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Csize_t, Cfloat, Cfloat, UInt64, Cuchar),
        handle,
        Csize_t(rows),
        Csize_t(cols),
        Float32(mean),
        Float32(std),
        seed_value,
        has_seed,
    )
    return _wrap_tensor(result, "runtime_tensor_random_normal")
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
    handle = ccall((fname, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), left, right)
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

function add(runtime::Runtime, lhs::Tensor, rhs::Tensor)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_add")
    left = _require_handle(lhs, "runtime_tensor_add lhs")
    right = _require_handle(rhs, "runtime_tensor_add rhs")
    result = ccall((:spiraltorch_runtime_tensor_add, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), handle, left, right)
    return _wrap_tensor(result, "runtime_tensor_add")
end

function sub(runtime::Runtime, lhs::Tensor, rhs::Tensor)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_sub")
    left = _require_handle(lhs, "runtime_tensor_sub lhs")
    right = _require_handle(rhs, "runtime_tensor_sub rhs")
    result = ccall((:spiraltorch_runtime_tensor_sub, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), handle, left, right)
    return _wrap_tensor(result, "runtime_tensor_sub")
end

function matmul(runtime::Runtime, lhs::Tensor, rhs::Tensor)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_matmul")
    left = _require_handle(lhs, "runtime_tensor_matmul lhs")
    right = _require_handle(rhs, "runtime_tensor_matmul rhs")
    result = ccall((:spiraltorch_runtime_tensor_matmul, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), handle, left, right)
    return _wrap_tensor(result, "runtime_tensor_matmul")
end

function scale(tensor::Tensor, value::Real)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_scale")
    result = ccall((:spiraltorch_tensor_scale, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Cfloat), handle, Float32(value))
    return _wrap_tensor(result, "tensor_scale")
end

function hadamard(lhs::Tensor, rhs::Tensor)
    return _binary_tensor_op(:spiraltorch_tensor_hadamard, lhs, rhs)
end

function scale(runtime::Runtime, tensor::Tensor, value::Real)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_scale")
    tensor_handle = _require_handle(tensor, "runtime_tensor_scale tensor")
    result = ccall((:spiraltorch_runtime_tensor_scale, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Cfloat), handle, tensor_handle, Float32(value))
    return _wrap_tensor(result, "runtime_tensor_scale")
end

function hadamard(runtime::Runtime, lhs::Tensor, rhs::Tensor)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_hadamard")
    left = _require_handle(lhs, "runtime_tensor_hadamard lhs")
    right = _require_handle(rhs, "runtime_tensor_hadamard rhs")
    result = ccall((:spiraltorch_runtime_tensor_hadamard, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), handle, left, right)
    return _wrap_tensor(result, "runtime_tensor_hadamard")
end

function transpose_tensor(tensor::Tensor)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_transpose")
    result = ccall((:spiraltorch_tensor_transpose, lib), Ptr{Cvoid}, (Ptr{Cvoid},), handle)
    return _wrap_tensor(result, "tensor_transpose")
end

function transpose_tensor(runtime::Runtime, tensor::Tensor)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_transpose")
    tensor_handle = _require_handle(tensor, "runtime_tensor_transpose tensor")
    result = ccall((:spiraltorch_runtime_tensor_transpose, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), handle, tensor_handle)
    return _wrap_tensor(result, "runtime_tensor_transpose")
end

function reshape_tensor(tensor::Tensor, rows::Integer, cols::Integer)
    lib = _lib()
    handle = _require_handle(tensor, "tensor_reshape")
    result = ccall((:spiraltorch_tensor_reshape, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Csize_t), handle, Csize_t(rows), Csize_t(cols))
    return _wrap_tensor(result, "tensor_reshape")
end

function reshape_tensor(runtime::Runtime, tensor::Tensor, rows::Integer, cols::Integer)
    lib = _lib()
    handle = _require_runtime(runtime, "runtime_tensor_reshape")
    tensor_handle = _require_handle(tensor, "runtime_tensor_reshape tensor")
    result = ccall((:spiraltorch_runtime_tensor_reshape, lib), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t), handle, tensor_handle, Csize_t(rows), Csize_t(cols))
    return _wrap_tensor(result, "runtime_tensor_reshape")
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

end # module
