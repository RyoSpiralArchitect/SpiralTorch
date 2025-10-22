# Usage example for SpiralTorch Julia tensor helpers.

using SpiralTorch

rows = [[1, 2, 3], [4, 5, 6]]
tensor = tensor_from_rows(rows)
println("tensor size: ", size(tensor))

cols = tensor_columns(tensor)
println("first column: ", cols[1])

flattened = tensor_vector(tensor)
println("flattened view: ", flattened)

reconstructed = tensor_rows(tensor)
reconstructed[1][1] = 42
println("mutated reconstruction (tensor unchanged): ", reconstructed)
