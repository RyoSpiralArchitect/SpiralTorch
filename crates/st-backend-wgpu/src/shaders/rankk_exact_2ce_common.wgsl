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
