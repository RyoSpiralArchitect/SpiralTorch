use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub rows:u32, pub cols:u32, pub k:u32, pub sg:bool }

#[derive(Clone, Copy, Debug, Default)]
pub struct Choice {
    pub use_2ce: bool,
    pub wg:  u32,
    pub kl:  u32,
    pub ch:  u32,
    pub mk:  u32,    // 0=bitonic,1=shared,2=warp
    pub mkd: u32,    // 0=auto,1=heap,2=kway,3=bitonic,4=warp_heap,5=warp_bitonic
    pub tile:u32,    // 256..8192
    pub ctile:u32,   // compaction/scan tile (MidK/BottomK) 256..8192
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field { Use2ce, Wg, Kl, Ch, Mk, Mkd, Tile, Ctile }

#[derive(Clone, Copy, Debug)]
pub enum Value { B(bool), U(u32) }

#[derive(Clone, Copy, Debug)]
pub struct SoftRule { pub field: Field, pub value: Value, pub weight: f32 }

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveCfg { pub noise:f32, pub seed:u64 }

fn hard_ok(choice:&Choice, ctx:&Ctx)->bool{
    if ![128,256].contains(&choice.wg) { return false; }
    if ![8,16,32].contains(&choice.kl) { return false; }
    if !(choice.ch==0 || choice.ch==8192) { return false; }
    if ![0,1,2].contains(&choice.mk) { return false; }
    if ![0,1,2,3,4,5].contains(&choice.mkd) { return false; }
    if ![256,512,1024,2048,4096,8192].contains(&choice.tile) { return false; }
    if ![256,512,1024,2048,4096,8192].contains(&choice.ctile) { return false; }
    if choice.tile as u32  > ctx.cols { return false; }
    if choice.ctile as u32 > ctx.cols { return false; }
    if !ctx.sg && choice.kl>16 { return false; }
    true
}

#[inline] fn l2(x:f32)->f32{ x.log2() }

fn soft_score(choice:&Choice, ctx:&Ctx, extra:&[SoftRule])->f32{
    let mut s=0.0f32;
    // baseline (shared with TopK)
    if ctx.cols>32_768 || ctx.k>128 { s += if choice.use_2ce {0.6} else {-0.2}; }
    if ctx.sg { s += if choice.wg==256 {0.3} else {-0.05}; }
    if ctx.k>=64 { s += match choice.kl {32=>0.3,16=>0.05,_=>-0.05} }
    else if ctx.k>=16 { s += match choice.kl {16=>0.2,8=>0.05,_=>-0.1} }
    else { s += if choice.kl==8 {0.15} else {-0.05} }
    if ctx.cols>16_384 { s += if choice.ch==8192 {0.25} else {-0.05}; }
    // mk preference
    if ctx.sg && ctx.k<=128 { s += if choice.mk==2 {0.30} else {-0.05}; }
    else if ctx.k<=2048 { s += if choice.mk==1 {0.20} else {-0.05}; }
    else { s += if choice.mk==0 {0.10} else {-0.10}; }
    // mkd (sub-strategy)
    s += match (choice.mk, choice.mkd) {
        (2,4) => 0.20, // warp_heap
        (2,5) => 0.12, // warp_bitonic
        (1,1) => 0.15, // shared_heap
        (1,2) => 0.14, // shared_kway
        (0,3) => 0.10, // pure bitonic
        _     => 0.0
    };
    // tile scale
    let lc = l2(ctx.cols as f32);
    s += match choice.tile {
        8192 => if lc>16.0 {0.22} else {-0.10},
        4096 => if lc>15.0 {0.18} else {-0.05},
        2048 => if lc>15.0 {0.15} else {-0.02},
        1024 => if lc>13.0 {0.12} else {-0.02},
        512  => if lc>12.0 {0.08} else {-0.02},
        256  => 0.05,
        _    => 0.0
    };
    // ctile preference (MidK/BottomK compaction/scan)
    s += match choice.ctile {
        8192 => if lc>16.0 {0.12} else {-0.08},
        4096 => if lc>15.0 {0.10} else {-0.05},
        2048 => if lc>15.0 {0.08} else {-0.02},
        1024 => if lc>13.0 {0.06} else {-0.02},
        512  => if lc>12.0 {0.04} else {-0.01},
        256  => 0.02,
        _    => 0.0
    };
    // extra soft
    for r in extra {
        match (r.field, r.value) {
            (Field::Use2ce, Value::B(v)) if choice.use_2ce==v => s+=r.weight,
            (Field::Wg,     Value::U(v)) if choice.wg     ==v => s+=r.weight,
            (Field::Kl,     Value::U(v)) if choice.kl     ==v => s+=r.weight,
            (Field::Ch,     Value::U(v)) if choice.ch     ==v => s+=r.weight,
            (Field::Mk,     Value::U(v)) if choice.mk     ==v => s+=r.weight,
            (Field::Mkd,    Value::U(v)) if choice.mkd    ==v => s+=r.weight,
            (Field::Tile,   Value::U(v)) if choice.tile   ==v => s+=r.weight,
            (Field::Ctile,  Value::U(v)) if choice.ctile  ==v => s+=r.weight,
            _=>{}
        }
    }
    s
}

pub fn solve_soft(ctx:Ctx, cfg:SolveCfg, extra:&[SoftRule])->(Choice,f32){
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let u2s = [false,true]; let wgs=[128,256]; let kls=[8,16,32]; let chs=[0,8192];
    let mks = [0u32,1u32,2u32]; let mkds=[0u32,1u32,2u32,3u32,4u32,5u32];
    let tiles=[256u32,512u32,1024u32,2048u32,4096u32,8192u32];
    let mut best=Choice::default(); let mut best_s=f32::NEG_INFINITY;
    for &u2 in &u2s { for &wg in &wgs { for &kl in &kls { for &ch in &chs {
        for &mk in &mks { for &mkd in &mkds { for &tile in &tiles { for &ct in &tiles {
            let cand=Choice{use_2ce:u2,wg,kl,ch,mk,mkd,tile,ctile:ct};
            if !hard_ok(&cand, &ctx){ continue; }
            let mut sc = soft_score(&cand, &ctx, extra);
            if cfg.noise>0.0 { sc += rng.gen::<f32>()*cfg.noise; }
            if sc>best_s { best_s=sc; best=cand; }
    }}}}}}}}
    (best,best_s)
}
