# Demonstration of runtime-centric helpers provided by the SpiralTorch Julia bindings.

using SpiralTorch

with_runtime(worker_threads=2, thread_name="julia-demo") do runtime
    println("runtime worker count: ", worker_count(runtime))

    a = Float32[1 2 3; 4 5 6]
    b = reshape(Float32[1, 0, -1], 3, 1)

    product = matmul(runtime, a, b)
    println("matmul result (Tensor): size = ", size(product))
    println("matmul result data: ", to_array(product))

    scaled = scale(runtime, a, 0.5)
    println("scaled matrix: ", to_array(scaled))

    mask = Float32[1 -1 1; -1 1 -1]
    had = hadamard(runtime, a, mask)
    println("hadamard column flips: ", to_array(had))

    sample = random_normal(runtime, (2, 2), 0.0, 1.0; seed=1234)
    println("random normal sample: ", to_array(sample))
end
