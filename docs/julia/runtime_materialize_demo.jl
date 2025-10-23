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
    product = matmul(runtime, mat, transpose(mat); materialize=true)
    println("runtime matmul materialised type: ", typeof(product))
    println("runtime matmul result:\n", product)

    uniform = random_uniform(runtime, (2, 2), -1, 1; seed=42, materialize=true)
    println("deterministic uniform sample:\n", uniform)
end
