var<workgroup> tournament_nodes: array<u32, 512>;

fn initialize_merge_bounds(row: u32, lane: u32) {
    if (lane == 0u) {
        var finite_count = 0u;
        for (var tile = 0u; tile < params.tiles_x; tile = tile + 1u) {
            finite_count = finite_count + tile_counts[row * params.tiles_x + tile];
        }
        let take = min(params.k, finite_count);
        if (params.kind == KIND_MIDK) {
            merge_start = (finite_count - take) / 2u;
        } else {
            merge_start = 0u;
        }
        merge_end = merge_start + take;
    }
    workgroupBarrier();
}

fn find_midk_cutoff(row: u32, lane: u32, rank_start: u32) -> vec2<u32> {
    var key = 0u;
    var index = 0u;
    for (var bit = 0x80000000u; bit > 0u; bit = bit >> 1u) {
        let probe = key | bit;
        if (row_count_before(row, probe, 0u, lane) <= rank_start) { key = probe; }
    }
    for (var bit = 0x80000000u; bit > 0u; bit = bit >> 1u) {
        if (bit < params.cols) {
            let probe = index | bit;
            if (row_count_before(row, key, probe, lane) <= rank_start) { index = probe; }
        }
    }
    return vec2<u32>(key, index);
}

fn tournament_winner(left: u32, right: u32) -> u32 {
    return select(left, right, candidate_before(
        merge_values[right], merge_indices[right],
        merge_values[left], merge_indices[left],
    ));
}

// One cached head per tile; after emitting a winner only its ancestors change.
// Parallel construction is followed by a dependency-ordered O(k log tiles)
// update loop, with no per-output workgroup or storage barrier.
fn merge_midk_tournament(
    row: u32, lane: u32, rank_start: u32, rank_end: u32,
    seek_prefix: bool, cutoff_key: u32, cutoff_index: u32,
) {
    var leaf_count = 1u;
    loop {
        if (leaf_count >= params.tiles_x) { break; }
        leaf_count = leaf_count << 1u;
    }
    var cursor = 0u;
    var value = 0.0;
    var index = INVALID_INDEX;
    if (lane < params.tiles_x) {
        let state = row * params.tiles_x + lane;
        if (seek_prefix) {
            cursor = tile_lower_bound(state, cutoff_key, cutoff_index);
        }
        if (cursor < tile_counts[state]) {
            let address = state * params.tile_stride + cursor;
            value = scratch_values[address];
            index = scratch_indices[address];
        }
    }
    merge_values[lane] = value;
    merge_indices[lane] = index;
    merge_tiles[lane] = cursor;
    if (lane < leaf_count) { tournament_nodes[leaf_count + lane] = lane; }
    for (var output_slot = lane; output_slot < params.k; output_slot = output_slot + 256u) {
        output_values[row * params.k + output_slot] = 0x7fc00000u;
        output_indices[row * params.k + output_slot] = INVALID_INDEX;
    }
    storageBarrier();
    workgroupBarrier();
    for (var stride = leaf_count >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            let node = stride + lane;
            tournament_nodes[node] = tournament_winner(
                tournament_nodes[node * 2u], tournament_nodes[node * 2u + 1u],
            );
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        for (var rank = select(0u, rank_start, seek_prefix); rank < rank_end; rank = rank + 1u) {
            let tile = tournament_nodes[1];
            if (merge_indices[tile] == INVALID_INDEX) { break; }
            if (rank >= rank_start) {
                let destination = row * params.k + (rank - rank_start);
                output_values[destination] = bitcast<u32>(merge_values[tile]);
                output_indices[destination] = merge_indices[tile];
            }
            // The final retained rank needs no successor load or tree repair.
            if (rank + 1u == rank_end) { break; }
            let state = row * params.tiles_x + tile;
            let next = merge_tiles[tile] + 1u;
            merge_tiles[tile] = next;
            merge_values[tile] = 0.0;
            merge_indices[tile] = INVALID_INDEX;
            if (next < tile_counts[state]) {
                let address = state * params.tile_stride + next;
                merge_values[tile] = scratch_values[address];
                merge_indices[tile] = scratch_indices[address];
            }
            for (var node = (leaf_count + tile) >> 1u; node > 0u; node = node >> 1u) {
                tournament_nodes[node] = tournament_winner(
                    tournament_nodes[node * 2u], tournament_nodes[node * 2u + 1u],
                );
            }
        }
    }
}

// This module is separate from the legacy tile-sort and row-merge module.
@compute @workgroup_size(256)
fn rankk_exact_2ce_midk_tournament(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = workgroup_id.x;
    if (row >= params.rows) { return; }
    initialize_merge_bounds(row, local_id.x);
    let rank_end = workgroupUniformLoad(&merge_end);
    let rank_start = workgroupUniformLoad(&merge_start);
    let seek_prefix = rank_start >= 64u;
    var cutoff = vec2<u32>(0u);
    if (seek_prefix) { cutoff = find_midk_cutoff(row, local_id.x, rank_start); }
    merge_midk_tournament(row, local_id.x, rank_start, rank_end,
        seek_prefix, cutoff.x, cutoff.y);
}
