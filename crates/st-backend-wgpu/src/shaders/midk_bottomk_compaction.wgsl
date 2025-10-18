// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct CParams {
    rows: u32,
    cols: u32,
    row_stride: u32,
    kind: u32,
    tiles_x: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> CX: array<f32>;
@group(0) @binding(1) var<storage, read> CMASK: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUTPOS: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> OUTVAL: array<f32>;
@group(0) @binding(4) var<uniform> CP: CParams;
@group(0) @binding(5) var<storage, read_write> PREFIX: array<u32>;

var<workgroup> temp: array<u32, 256u>;

@compute @workgroup_size(256)
fn midk_compact_scan_tiles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = wid.y;
    let tile = gid.x;
    if (r >= CP.rows || tile >= CP.tiles_x) {
        return;
    }

    let start = tile * 256u;
    var v: u32 = 0u;
    var c = lid.x;
    loop {
        let col = start + c;
        if (col >= CP.cols) {
            break;
        }
        if (CMASK[r * CP.row_stride + col] != 0u) {
            v = v + 1u;
        }
        c = c + 256u;
    }
    temp[lid.x] = v;
    workgroupBarrier();

    var offset = 1u;
    var d = 256u;
    loop {
        if (d <= 1u) {
            break;
        }
        let half = d >> 1u;
        if (lid.x < half) {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }
        offset = offset << 1u;
        d = half;
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        PREFIX[r * CP.tiles_x + tile] = temp[255u];
    }
}

@compute @workgroup_size(256)
fn midk_compact_row_prefix(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = wid.y;
    if (r >= CP.rows) {
        return;
    }

    var acc: u32 = 0u;
    var base: u32 = 0u;
    loop {
        if (base >= CP.tiles_x) {
            break;
        }
        let idx = r * CP.tiles_x + base + lid.x;
        var x: u32 = 0u;
        if (base + lid.x < CP.tiles_x) {
            x = PREFIX[idx];
        }
        temp[lid.x] = x;
        workgroupBarrier();

        var off = 1u;
        var d = 256u;
        loop {
            if (d <= 1u) {
                break;
            }
            let h = d >> 1u;
            if (lid.x < h) {
                let ai = off * (2u * lid.x + 1u) - 1u;
                let bi = off * (2u * lid.x + 2u) - 1u;
                temp[bi] = temp[bi] + temp[ai];
            }
            off = off << 1u;
            d = h;
            workgroupBarrier();
        }

        var last = temp[255u];
        if (lid.x == 255u) {
            temp[255u] = 0u;
        }
        workgroupBarrier();

        var h2 = 128u;
        var step = 128u;
        loop {
            if (h2 == 0u) {
                break;
            }
            if (lid.x < h2) {
                let ai = step * (2u * lid.x + 1u) - 1u;
                let bi = step * (2u * lid.x + 2u) - 1u;
                let t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] = temp[bi] + t;
            }
            step = step >> 1u;
            h2 = h2 >> 1u;
            workgroupBarrier();
        }

        if (base + lid.x < CP.tiles_x) {
            PREFIX[idx] = temp[lid.x] + acc;
        }
        acc = acc + last;
        base = base + 256u;
    }

    if (lid.x == 0u) {
        atomicStore(&OUTPOS[r], acc);
    }
}

var<workgroup> wg_sg_base: atomic<u32>;
var<workgroup> sg_bases: array<u32, 8u>;

@compute @workgroup_size(256)
fn midk_compact_apply_sg(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let r = wid.y;
    let tile = gid.x;
    if (r >= CP.rows || tile >= CP.tiles_x) {
        return;
    }

    let start = tile * 256u;
    let base = PREFIX[r * CP.tiles_x + tile];

    if (lid.x == 0u) {
        atomicStore(&wg_sg_base, 0u);
    }
    workgroupBarrier();

    var flag: u32 = 0u;
    let col = start + lid.x;
    if (col < CP.cols) {
        if (CMASK[r * CP.row_stride + col] != 0u) {
            flag = 1u;
        }
    }

    let sgc = 256u / max(sg_size, 1u);
    let sg_id = lid.x / max(sg_size, 1u);
    var local_excl: u32 = 0u;
    var sg_total: u32 = 0u;
    for (var j: u32 = 0u; j < max(sg_size, 1u); j = j + 1u) {
        let st = start + j + sg_id * max(sg_size, 1u);
        if (st < (start + 256u) && st < CP.cols) {
            let f = u32(CMASK[r * CP.row_stride + st] != 0u);
            if (j < sg_lane) {
                local_excl = local_excl + f;
            }
            sg_total = sg_total + f;
        }
    }

    if (sg_lane == 0u) {
        let b = atomicAdd(&wg_sg_base, sg_total);
        sg_bases[sg_id] = b;
    }
    workgroupBarrier();
    let my_base = sg_bases[sg_id];

    if (flag == 1u && col < CP.cols) {
        let pos = base + my_base + local_excl;
        OUTVAL[r * CP.cols + pos] = CX[r * CP.row_stride + col];
    }
}

var<workgroup> sg_totals: array<u32, 8u>;
var<workgroup> sg_temp: array<u32, 8u>;

@compute @workgroup_size(256)
fn midk_compact_apply_sg2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let r = wid.y;
    let tile = gid.x;
    if (r >= CP.rows || tile >= CP.tiles_x) {
        return;
    }

    let start = tile * 256u;
    let base = PREFIX[r * CP.tiles_x + tile];

    var flag: u32 = 0u;
    let col = start + lid.x;
    if (col < CP.cols) {
        if (CMASK[r * CP.row_stride + col] != 0u) {
            flag = 1u;
        }
    }

    let sgsz = max(sg_size, 1u);
    let sgc = 256u / sgsz;
    let sg_id = lid.x / sgsz;

    var local_excl: u32 = 0u;
    var sg_total: u32 = 0u;
    let sg_start = start + sg_id * sgsz;
    for (var j: u32 = 0u; j < sgsz; j = j + 1u) {
        let idx = sg_start + j;
        if (idx < CP.cols) {
            let f = u32(CMASK[r * CP.row_stride + idx] != 0u);
            if (j < sg_lane) {
                local_excl = local_excl + f;
            }
            sg_total = sg_total + f;
        }
    }

    if (sg_lane == 0u) {
        sg_totals[sg_id] = sg_total;
    }
    workgroupBarrier();

    if (lid.x < sgc) {
        sg_temp[lid.x] = sg_totals[lid.x];
    }
    workgroupBarrier();

    var d = 1u;
    var n = sgc;
    loop {
        if (d >= n) {
            break;
        }
        let i = ((lid.x + 1u) * (d << 1u)) - 1u;
        if (lid.x < n && i < n) {
            sg_temp[i] = sg_temp[i] + sg_temp[i - d];
        }
        d = d << 1u;
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        sg_temp[n - 1u] = 0u;
    }
    workgroupBarrier();

    var step = n >> 1u;
    var stride = step;
    loop {
        if (step == 0u) {
            break;
        }
        let i = ((lid.x + 1u) * (stride << 1u)) - 1u;
        if (lid.x < n && i < n) {
            let t = sg_temp[i - stride];
            sg_temp[i - stride] = sg_temp[i];
            sg_temp[i] = sg_temp[i] + t;
        }
        step = step >> 1u;
        stride = max(stride >> 1u, 1u);
        workgroupBarrier();
    }

    if (lid.x < sgc) {
        sg_bases[lid.x] = sg_temp[lid.x];
    }
    workgroupBarrier();
    let sg_base = sg_bases[sg_id];

    if (flag == 1u && col < CP.cols) {
        let pos = base + sg_base + local_excl;
        OUTVAL[r * CP.cols + pos] = CX[r * CP.row_stride + col];
    }
}

var<workgroup> temp2: array<u32, 256u>;

@compute @workgroup_size(256)
fn midk_compact_apply(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = wid.y;
    let tile = gid.x;
    if (r >= CP.rows || tile >= CP.tiles_x) {
        return;
    }

    let start = tile * 256u;
    let base = PREFIX[r * CP.tiles_x + tile];

    var flag: u32 = 0u;
    let col0 = start + lid.x;
    if (col0 < CP.cols) {
        if (CMASK[r * CP.row_stride + col0] != 0u) {
            flag = 1u;
        }
    }
    temp2[lid.x] = flag;
    workgroupBarrier();

    var off = 1u;
    var d = 256u;
    loop {
        if (d <= 1u) {
            break;
        }
        let h = d >> 1u;
        if (lid.x < h) {
            let ai = off * (2u * lid.x + 1u) - 1u;
            let bi = off * (2u * lid.x + 2u) - 1u;
            temp2[bi] = temp2[bi] + temp2[ai];
        }
        off = off << 1u;
        d = h;
        workgroupBarrier();
    }

    if (lid.x == 255u) {
        temp2[255u] = 0u;
    }
    workgroupBarrier();

    var h2 = 128u;
    var step = 128u;
    loop {
        if (h2 == 0u) {
            break;
        }
        if (lid.x < h2) {
            let ai = step * (2u * lid.x + 1u) - 1u;
            let bi = step * (2u * lid.x + 2u) - 1u;
            let t = temp2[ai];
            temp2[ai] = temp2[bi];
            temp2[bi] = temp2[bi] + t;
        }
        step = step >> 1u;
        h2 = h2 >> 1u;
        workgroupBarrier();
    }

    if (col0 < CP.cols && flag == 1u) {
        let pos = base + temp2[lid.x];
        OUTVAL[r * CP.cols + pos] = CX[r * CP.row_stride + col0];
    }
}
