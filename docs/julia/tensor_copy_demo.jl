# Demonstrate copying SpiralTorch tensors into existing Julia arrays.

using SpiralTorch

base = Float32[1 2 3; 4 5 6]
tensor = Tensor(base)
rows, cols = size(tensor)

matrix_buffer = Matrix{Float32}(undef, rows, cols)
copyto!(matrix_buffer, tensor)
println("Float32 buffer filled without allocation:\n", matrix_buffer)

float64_buffer = Matrix{Float64}(undef, rows, cols)
copyto!(float64_buffer, tensor)
println("Float64 buffer receives converted copy:\n", float64_buffer)

vector_buffer = Vector{Float32}(undef, length(tensor))
copyto!(vector_buffer, tensor)
println("Flattened tensor copy: ", vector_buffer)
