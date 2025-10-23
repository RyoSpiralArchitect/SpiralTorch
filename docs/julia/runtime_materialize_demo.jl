# Demonstration of the extended Julia helpers that bridge Arrays and Tensors.

using SpiralTorch

# Construct tensors from vectors and tuple-based shapes.
column = Tensor([1.0, 2.0, 3.0])
reshaped = Tensor(collect(1:6), (2, 3))
println("column tensor shape: ", size(column))
println("reshaped tensor contents:\n", to_array(reshaped))

# Mixed arithmetic automatically promotes arrays to tensors.
mat = Float32[1 2 3; 4 5 6]
println("tensor + matrix: ", to_array(reshaped + mat))
println("matrix - tensor: ", to_array(mat - reshaped))
println("tensor .* matrix: ", to_array(reshaped .* mat))

# Runtime helpers can now materialise results directly.
with_runtime(worker_threads=1) do runtime
    reuse = Matrix{Float32}(undef, size(mat, 1), size(mat, 1))
    product = matmul(runtime, mat, transpose(mat); materialize_into=reuse)
    println("runtime matmul reused buffer: ", product === reuse)
    println("runtime matmul result:\n", product)

    uniform = Matrix{Float32}(undef, 2, 2)
    random_uniform(runtime, (2, 2), -1, 1; seed=42, materialize_into=uniform)
    println("deterministic uniform sample (reused storage):\n", uniform)
end

# Direct tensor copies avoid allocations when using preallocated buffers.
tensor = Tensor(mat)
preallocated = Matrix{Float32}(undef, size(mat, 1), size(mat, 2))
copyto!(preallocated, tensor)
println("copyto! into matrix:\n", preallocated)

flat = Vector{Float32}(undef, length(tensor))
copyto!(flat, tensor)
println("copyto! into vector: ", flat)
