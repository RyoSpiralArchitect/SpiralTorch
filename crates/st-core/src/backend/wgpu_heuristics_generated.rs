// Auto-generated (sample). Replace via tools/tuner/gen_generated_rs.py
use super::Choice;
#[allow(unused)]
pub fn choose(rows: usize, cols: usize, k: usize, subgroup: bool) -> Option<Choice> {
    let lc = (cols as f32).log2();
    let mk = if subgroup && k<=128 { 2 }
             else if k<=2048 { 1 } else { 0 };
    let tile = if lc>15.0 { 2048 } else if lc>13.0 { 1024 } else if lc>12.0 { 512 } else { 256 };
    let wg = if subgroup {256} else {128};
    let kl = if k>=64 {32} else if k>=16 {16} else {8};
    let ch = if cols>16_384 {8192} else {0};
    let use_2ce = (cols>32_768) || (k>128);
    Some(Choice{ use_2ce, wg, kl, ch, mk, tile })
}
