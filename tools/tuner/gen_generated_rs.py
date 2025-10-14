#!/usr/bin/env python3
import sys, json, math

def main():
    if len(sys.argv)<2:
        print("usage: gen_generated_rs.py tuner_results.json", file=sys.stderr); sys.exit(1)
    data = json.load(open(sys.argv[1], 'r'))
    # naive piecewise stitching: bucket by sg, then by log2 cols ranges and k thresholds
    # we generate if-else chains; collisions resolved by last-wins order.
    print("// Auto-generated from tuner_results.json")
    print("use super::Choice;")
    print("pub fn choose(rows: usize, cols: usize, k: usize, subgroup: bool) -> Option<Choice> {")
    print("    let lc = (cols as f32).log2();")
    print("    let mut mk:u32 = if subgroup && k<=128 {2} else if k<=2048 {1} else {0};")
    print("    let mut tile:u32 = if lc>15.0 {2048} else if lc>13.0 {1024} else if lc>12.0 {512} else {256};")
    print("    let mut tile_cols:u32 = if cols>131_072 {4096} else if cols>65_536 {2048} else if cols>16_384 {1024} else {512};")
    print("    let mut radix:u32 = if (k & (k-1))==0 {4} else {2};")
    print("    let mut segments:u32 = if cols>131_072 {4} else if cols>32_768 {2} else {1};")
    print("    let mut use_2ce = (cols>32_768) || (k>128);")
    for e in data:
        sg = "true" if e.get("sg", False) else "false"
        cmin = e.get("cols_min", 0); cmax = e.get("cols_max", 1<<30); kmax=e.get("k_max", 1<<30)
        mk = e.get("mk")
        tl = e.get("tile")
        tc = e.get("tile_cols")
        rd = e.get("radix")
        sgmt = e.get("segments")
        u2 = e.get("use_2ce")
        print(f"    if subgroup=={sg} && cols>={cmin} && cols<={cmax} && k<={kmax} {{")
        if mk is not None:
            print(f"        mk={int(mk)};")
        if tl is not None:
            print(f"        tile={int(tl)};")
        if tc is not None:
            print(f"        tile_cols={int(tc)};")
        if rd is not None:
            print(f"        radix={int(rd)};")
        if sgmt is not None:
            print(f"        segments={int(sgmt)};")
        if u2 is not None:
            print(f"        use_2ce={str(bool(u2)).lower()};")
        print("    }")
    print("    let wg = if subgroup {256} else {128};")
    print("    let kl = if k>=64 {32} else if k>=16 {16} else {8};")
    print("    let ch = if cols>16_384 {8192} else {0};")
    print("    Some(Choice{ use_2ce, wg, kl, ch, algo_topk: 0, ctile: tile, mode_midk: 0, mode_bottomk: 0, tile_cols, radix, segments })")
    print("}")
if __name__=='__main__': main()
