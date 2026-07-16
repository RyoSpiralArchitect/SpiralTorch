// Exact two-command TopK/MidK/BottomK for finite f32 row candidates.

const KIND_TOPK: u32 = 0u;
const KIND_MIDK: u32 = 1u;
const INVALID_INDEX: u32 = 0xffffffffu;

struct Params {
    rows: u32,
    cols: u32,
    k: u32,
    tile_cols: u32,
    tile_stride: u32,
    tiles_x: u32,
    kind: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_cursors: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> merge_values: array<f32, 256>;
var<workgroup> merge_indices: array<u32, 256>;
var<workgroup> merge_tiles: array<u32, 256>;
var<workgroup> merge_start: u32;
var<workgroup> merge_end: u32;

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

fn swap_scratch(left: u32, right: u32) {
    let value = scratch_values[left];
    let index = scratch_indices[left];
    scratch_values[left] = scratch_values[right];
    scratch_indices[left] = scratch_indices[right];
    scratch_values[right] = value;
    scratch_indices[right] = index;
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
    let row = workgroup_id.x;
    if (row >= params.rows) {
        return;
    }

    var tile = local_id.x;
    loop {
        if (tile >= params.tiles_x) {
            break;
        }
        tile_cursors[row * params.tiles_x + tile] = 0u;
        tile = tile + 256u;
    }

    var output_slot = local_id.x;
    loop {
        if (output_slot >= params.k) {
            break;
        }
        output_values[row * params.k + output_slot] = bitcast<f32>(0x7fc00000u);
        output_indices[row * params.k + output_slot] = INVALID_INDEX;
        output_slot = output_slot + 256u;
    }
    storageBarrier();
    workgroupBarrier();

    if (local_id.x == 0u) {
        var finite_count = 0u;
        for (var current_tile = 0u; current_tile < params.tiles_x; current_tile = current_tile + 1u) {
            finite_count = finite_count + tile_counts[row * params.tiles_x + current_tile];
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

    var rank = 0u;
    loop {
        if (rank >= merge_end) {
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

        var stride = 128u;
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
                    output_values[destination] = merge_values[0];
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
