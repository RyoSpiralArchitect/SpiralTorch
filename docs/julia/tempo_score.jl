module SpiralTempo
export tempo_score

"""
    tempo_score(tile::UInt32, slack::UInt32)

Reference Julia implementation used by the Rust FFI PoC. Returns a latency
score that mirrors `rust_latency_score`.
"""
function tempo_score(tile::UInt32, slack::UInt32)
    sqrt(float(tile)) + float(slack) / 2
end

end
