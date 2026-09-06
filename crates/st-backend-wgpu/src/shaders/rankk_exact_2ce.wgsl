// Exact two-command TopK/MidK/BottomK for finite f32 row candidates.

const KIND_TOPK: u32 = 0u;
const KIND_MIDK: u32 = 1u;
const INVALID_INDEX: u32 = 0xffffffffu;
const LOCAL_SORT_CAPACITY: u32 = 1024u;
const MERGE_PARALLEL_MIDK: u32 = 1u;

struct Params {
    rows: u32,
    cols: u32,
    k: u32,
    tile_cols: u32,
    tile_stride: u32,
    tiles_x: u32,
    kind: u32,
    merge_mode: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_cursors: array<u32>;
// Preserve sentinel bits without constructing a non-finite WGSL constant.
@group(0) @binding(5) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> merge_values: array<f32, 256>;
var<workgroup> merge_indices: array<u32, 256>;
var<workgroup> merge_tiles: array<u32, 256>;
var<workgroup> merge_start: u32;
var<workgroup> merge_end: u32;
var<workgroup> sort_values: array<f32, 1024>;
var<workgroup> sort_indices: array<u32, 1024>;
var<workgroup> tournament_nodes: array<u32, 512>;

fn float_total_key(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits ^ 0x80000000u;
}

fn finite_f32(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn candidate_before(
    left_value: f32,
    left_index: u32,
    right_value: f32,
    right_index: u32,
) -> bool {
    if (left_index == INVALID_INDEX) {
        return false;
    }
    if (right_index == INVALID_INDEX) {
        return true;
    }

    let left_key = float_total_key(left_value);
    let right_key = float_total_key(right_value);
    if (left_key == right_key) {
        return left_index < right_index;
    }
    if (params.kind == KIND_TOPK) {
        return left_key > right_key;
    }
    return left_key < right_key;
}

// Ascending lower bound in one finite sorted run, used only by MidK.
fn tile_lower_bound(state: u32, key: u32, index: u32) -> u32 {
    let base = state * params.tile_stride;
    var low = 0u;
    var high = tile_counts[state];
    loop {
        if (low >= high) { break; }
        let middle = low + (high - low) / 2u;
        let other_key = float_total_key(scratch_values[base + middle]);
        if (other_key < key || (other_key == key && scratch_indices[base + middle] < index)) {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}

// Each lane owns tiles lane + n * 256. Higher lanes contain only the reduction
// identity, so skip empty upper levels without changing the surviving tree.
fn merge_reduction_start() -> u32 {
    var stride = 128u;
    loop {
        if (stride < params.tiles_x || stride == 0u) { break; }
        stride = stride >> 1u;
    }
    return stride;
}

fn row_count_before(row: u32, key: u32, index: u32, lane: u32) -> u32 {
    var count = 0u;
    for (var tile = lane; tile < params.tiles_x; tile = tile + 256u) {
        count = count + tile_lower_bound(row * params.tiles_x + tile, key, index);
    }
    merge_indices[lane] = count;
    workgroupBarrier();
    for (var stride = merge_reduction_start(); stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            merge_indices[lane] = merge_indices[lane] + merge_indices[lane + stride];
        }
        workgroupBarrier();
    }
    let total = workgroupUniformLoad(&merge_indices[0]);
    // All lanes must finish reading before a later search reuses this scratch.
    workgroupBarrier();
    return total;
}

fn initialize_merge_bounds(row: u32, lane: u32) {
    if (lane == 0u) {
        var finite_count = 0u;
        for (var tile = 0u; tile < params.tiles_x; tile = tile + 1u) {
            finite_count = finite_count + tile_counts[row * params.tiles_x + tile];
        }
        let take = min(params.k, finite_count);
        merge_start = select(0u, (finite_count - take) / 2u, params.kind == KIND_MIDK);
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

fn swap_scratch(left: u32, right: u32) {
    let value = scratch_values[left];
    let index = scratch_indices[left];
    scratch_values[left] = scratch_values[right];
    scratch_indices[left] = scratch_indices[right];
    scratch_values[right] = value;
    scratch_indices[right] = index;
}

fn sort_tile_local(row: u32, tile: u32, lane: u32) {
    let tile_state = row * params.tiles_x + tile;
    let scratch_base = tile_state * params.tile_stride;
    let column_base = tile * params.tile_cols;
    let tile_length = min(params.tile_cols, params.cols - column_base);
    for (var slot = lane; slot < params.tile_stride; slot = slot + 256u) {
        var value = 0.0;
        var index = INVALID_INDEX;
        if (slot < tile_length) {
            let column = column_base + slot;
            let candidate = input_values[row * params.cols + column];
            if (finite_f32(candidate)) {
                value = candidate;
                index = column;
            }
        }
        sort_values[slot] = value;
        sort_indices[slot] = index;
    }
    workgroupBarrier();

    // Only this workgroup owns the tile. Keep compare/exchange traffic local,
    // then publish the sorted run once for the separate row-merge pass.
    for (var span = 2u; span <= params.tile_stride; span = span << 1u) {
        for (var distance = span >> 1u; distance > 0u; distance = distance >> 1u) {
            for (var slot = lane; slot < params.tile_stride; slot = slot + 256u) {
                let partner = slot ^ distance;
                if (partner > slot && partner < params.tile_stride) {
                    let left_value = sort_values[slot];
                    let left_index = sort_indices[slot];
                    let right_value = sort_values[partner];
                    let right_index = sort_indices[partner];
                    let ascending_half = (slot & span) == 0u;
                    let swap = select(
                        candidate_before(left_value, left_index, right_value, right_index),
                        candidate_before(right_value, right_index, left_value, left_index),
                        ascending_half,
                    );
                    if (swap) {
                        sort_values[slot] = right_value;
                        sort_indices[slot] = right_index;
                        sort_values[partner] = left_value;
                        sort_indices[partner] = left_index;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    for (var slot = lane; slot < params.tile_stride; slot = slot + 256u) {
        scratch_values[scratch_base + slot] = sort_values[slot];
        scratch_indices[scratch_base + slot] = sort_indices[slot];
    }
    if (lane == 0u) {
        var count = 0u;
        for (; count < params.tile_stride; count = count + 1u) {
            if (sort_indices[count] == INVALID_INDEX) { break; }
        }
        tile_counts[tile_state] = count;
    }
}

@compute @workgroup_size(256)
fn rankk_exact_2ce_tile_sort(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tile = workgroup_id.x;
    let row = workgroup_id.y;
    if (row >= params.rows || tile >= params.tiles_x) {
        return;
    }
    if (params.tile_stride <= LOCAL_SORT_CAPACITY) {
        sort_tile_local(row, tile, local_id.x);
        return;
    }

    let tile_state = row * params.tiles_x + tile;
    let scratch_base = tile_state * params.tile_stride;
    let column_base = tile * params.tile_cols;
    let tile_length = min(params.tile_cols, params.cols - column_base);

    var slot = local_id.x;
    loop {
        if (slot >= params.tile_stride) {
            break;
        }
        let scratch_index = scratch_base + slot;
        if (slot < tile_length) {
            let column = column_base + slot;
            let value = input_values[row * params.cols + column];
            if (finite_f32(value)) {
                scratch_values[scratch_index] = value;
                scratch_indices[scratch_index] = column;
            } else {
                scratch_values[scratch_index] = 0.0;
                scratch_indices[scratch_index] = INVALID_INDEX;
            }
        } else {
            scratch_values[scratch_index] = 0.0;
            scratch_indices[scratch_index] = INVALID_INDEX;
        }
        slot = slot + 256u;
    }
    storageBarrier();
    workgroupBarrier();

    var span = 2u;
    loop {
        if (span > params.tile_stride) {
            break;
        }
        var distance = span >> 1u;
        loop {
            if (distance == 0u) {
                break;
            }
            slot = local_id.x;
            loop {
                if (slot >= params.tile_stride) {
                    break;
                }
                let partner = slot ^ distance;
                if (partner > slot && partner < params.tile_stride) {
                    let left = scratch_base + slot;
                    let right = scratch_base + partner;
                    let left_before = candidate_before(
                        scratch_values[left],
                        scratch_indices[left],
                        scratch_values[right],
                        scratch_indices[right],
                    );
                    let right_before = candidate_before(
                        scratch_values[right],
                        scratch_indices[right],
                        scratch_values[left],
                        scratch_indices[left],
                    );
                    let ascending_half = (slot & span) == 0u;
                    if ((ascending_half && right_before) || (!ascending_half && left_before)) {
                        swap_scratch(left, right);
                    }
                }
                slot = slot + 256u;
            }
            storageBarrier();
            workgroupBarrier();
            distance = distance >> 1u;
        }
        span = span << 1u;
    }

    if (local_id.x == 0u) {
        var count = 0u;
        loop {
            if (count >= params.tile_stride) {
                break;
            }
            if (scratch_indices[scratch_base + count] == INVALID_INDEX) {
                break;
            }
            count = count + 1u;
        }
        tile_counts[tile_state] = count;
    }
}

@compute @workgroup_size(256)
fn rankk_exact_2ce_row_merge(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let parallel_midk = params.merge_mode == MERGE_PARALLEL_MIDK;
    let row = select(workgroup_id.x, workgroup_id.y, parallel_midk);
    if (row >= params.rows) {
        return;
    }

    initialize_merge_bounds(row, local_id.x);
    // Make the data-dependent loop bound provably uniform to browser validators.
    let rank_end = workgroupUniformLoad(&merge_end);
    let rank_start = workgroupUniformLoad(&merge_start);

    if (parallel_midk) {
        // Every tile is sorted by the same total (value, source-index) order.
        // Binary-search other tiles to compute each candidate's global rank,
        // avoiding a serial merge through half the row just to discard it.
        // Keep the original merge for extremely fragmented tile geometries.
        // Each workgroup owns one tile's candidates. Valid global ranks have
        // unique destinations; initialize ONLY the disjoint missing-value tail.
        let take = rank_end - rank_start;
        for (var tail = workgroup_id.x * 256u + local_id.x;
             tail < params.k - take; tail = tail + params.tiles_x * 256u) {
            let destination = row * params.k + take + tail;
            output_values[destination] = 0x7fc00000u;
            output_indices[destination] = INVALID_INDEX;
        }
        let own_tile = workgroup_id.x;
        let own_state = row * params.tiles_x + own_tile;
        var own_offset = local_id.x;
        loop {
            if (own_offset >= tile_counts[own_state]) { break; }
            let address = own_state * params.tile_stride + own_offset;
            let value = scratch_values[address];
            let index = scratch_indices[address];
            var global_rank = own_offset;
            for (var other = 0u; other < params.tiles_x; other = other + 1u) {
                if (other != own_tile) {
                    let state = row * params.tiles_x + other;
                    let base = state * params.tile_stride;
                    var low = 0u;
                    var high = tile_counts[state];
                    loop {
                        if (low >= high) { break; }
                        let middle = low + (high - low) / 2u;
                        if (candidate_before(scratch_values[base + middle],
                            scratch_indices[base + middle], value, index)) {
                            low = middle + 1u;
                        } else {
                            high = middle;
                        }
                    }
                    global_rank = global_rank + low;
                }
            }
            if (global_rank >= rank_start && global_rank < rank_end) {
                let destination = row * params.k + global_rank - rank_start;
                output_values[destination] = bitcast<u32>(value);
                output_indices[destination] = index;
            }
            own_offset = own_offset + 256u;
        }
        return;
    }

    // Find the exact first retained (total-float-key, source-index) in at most
    // 32 + ceil(log2(cols)) cooperative probes. This skips a large discarded
    // prefix without changing the k-way merge or adding another dispatch.
    // Short prefixes retain the cheaper direct merge.
    let seek_prefix = params.kind == KIND_MIDK && rank_start >= 64u;
    var cutoff = vec2<u32>(0u);
    if (seek_prefix) {
        cutoff = find_midk_cutoff(row, local_id.x, rank_start);
    }
    let cutoff_key = cutoff.x;
    let cutoff_index = cutoff.y;

    var tile = local_id.x;
    for (; tile < params.tiles_x; tile = tile + 256u) {
        let state = row * params.tiles_x + tile;
        var cursor = 0u;
        if (seek_prefix) {
            cursor = tile_lower_bound(state, cutoff_key, cutoff_index);
        }
        tile_cursors[state] = cursor;
    }
    for (var output_slot = local_id.x; output_slot < params.k; output_slot = output_slot + 256u) {
        output_values[row * params.k + output_slot] = 0x7fc00000u;
        output_indices[row * params.k + output_slot] = INVALID_INDEX;
    }
    storageBarrier();
    workgroupBarrier();

    var rank = select(0u, rank_start, seek_prefix);
    loop {
        if (rank >= rank_end) {
            break;
        }

        var best_value = 0.0;
        var best_index = INVALID_INDEX;
        var best_tile = INVALID_INDEX;
        tile = local_id.x;
        loop {
            if (tile >= params.tiles_x) {
                break;
            }
            let tile_state = row * params.tiles_x + tile;
            let cursor = tile_cursors[tile_state];
            if (cursor < tile_counts[tile_state]) {
                let scratch_index = tile_state * params.tile_stride + cursor;
                let value = scratch_values[scratch_index];
                let index = scratch_indices[scratch_index];
                if (candidate_before(value, index, best_value, best_index)) {
                    best_value = value;
                    best_index = index;
                    best_tile = tile;
                }
            }
            tile = tile + 256u;
        }
        merge_values[local_id.x] = best_value;
        merge_indices[local_id.x] = best_index;
        merge_tiles[local_id.x] = best_tile;
        workgroupBarrier();

        var stride = merge_reduction_start();
        loop {
            if (stride == 0u) {
                break;
            }
            if (local_id.x < stride) {
                let right = local_id.x + stride;
                if (candidate_before(
                    merge_values[right],
                    merge_indices[right],
                    merge_values[local_id.x],
                    merge_indices[local_id.x],
                )) {
                    merge_values[local_id.x] = merge_values[right];
                    merge_indices[local_id.x] = merge_indices[right];
                    merge_tiles[local_id.x] = merge_tiles[right];
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        if (local_id.x == 0u) {
            let selected_tile = merge_tiles[0];
            if (selected_tile != INVALID_INDEX) {
                if (rank >= merge_start) {
                    let destination = row * params.k + (rank - merge_start);
                    output_values[destination] = bitcast<u32>(merge_values[0]);
                    output_indices[destination] = merge_indices[0];
                }
                let tile_state = row * params.tiles_x + selected_tile;
                tile_cursors[tile_state] = tile_cursors[tile_state] + 1u;
            }
        }
        storageBarrier();
        workgroupBarrier();
        rank = rank + 1u;
    }
}

// A separate entry point keeps the cached tree's storage and register pressure
// out of the existing parallel MidK and streaming rank pipelines.
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
