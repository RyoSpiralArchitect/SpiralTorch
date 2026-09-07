// Exact two-command TopK/MidK/BottomK for finite f32 row candidates.

const KIND_TOPK: u32 = 0u;
const KIND_MIDK: u32 = 1u;
const INVALID_INDEX: u32 = 0xffffffffu;
const LOCAL_SORT_CAPACITY: u32 = 1024u;
const MERGE_PARALLEL_MIDK: u32 = 1u;
const MERGE_PARALLEL_PREFIX: u32 = 3u;

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
