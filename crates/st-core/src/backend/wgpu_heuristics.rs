// WGPU TopK/MidK/BottomK heuristics Choice (includes mkd/ctile and two_ce_hint)
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,   // 0=bitonic,1=shared,2=warp
    pub tile:u32,
    pub mkd: u32,   // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub ctile:u32,  // compaction tile
    pub two_ce_hint: bool,
}

pub fn choose(rows:u32, cols:u32, k:u32, subgroup: bool) -> Option<Choice> {
    if let Some(c) = choose_generated(rows as usize, cols as usize, k as usize, subgroup) {
        return Some(Choice{
            use_2ce: c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch,
            mk: c.mk, tile:c.tile,
            mkd: c.mkd.unwrap_or(0),
            ctile: c.ctile.unwrap_or(0),
            two_ce_hint: c.two_ce_hint,
        });
    }
    None
}

include!("wgpu_heuristics_generated.rs");
