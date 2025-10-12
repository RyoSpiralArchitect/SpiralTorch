// WGPU TopK heuristics: SpiralK → Generated → Fallback
// (Choice extended with mkd/ctile to align with unified chooser)
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,   // 0=bitonic,1=shared,2=warp
    pub tile:u32,
    pub mkd: u32,   // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub ctile:u32,  // compaction tile hint (for MidK/BottomK)
}

pub fn choose(rows:u32, cols:u32, k:u32, subgroup: bool) -> Option<Choice> {
    if let Some(c) = choose_generated(rows as usize, cols as usize, k as usize, subgroup) {
        return Some(Choice{
            use_2ce: c.use_2ce, wg:c.wg, kl:c.kl, ch:c.ch,
            mk: c.mk, tile: c.tile,
            mkd: c.mkd.unwrap_or(0),
            ctile: c.ctile.unwrap_or(0),
        });
    }
    None
}

// Generated (or stub) is included here (same crate):
include!("wgpu_heuristics_generated.rs");
