use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub rows:u32, pub cols:u32, pub k:u32, pub sg: bool }

#[derive(Clone, Copy, Debug, Default)]
pub struct Choice { pub use_2ce: bool, pub wg:u32, pub kl:u32, pub ch:u32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field { Use2ce, Wg, Kl, Ch }

#[derive(Clone, Copy, Debug)]
pub enum Value { B(bool), U(u32) }

#[derive(Clone, Copy, Debug)]
pub struct SoftRule { pub field: Field, pub value: Value, pub weight: f32 }

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveCfg { pub noise:f32, pub seed:u64 }

fn hard_ok(c:&Choice, ctx:&Ctx)->bool{
    if ![128,256].contains(&c.wg) { return false; }
    if ![8,16,32].contains(&c.kl) { return false; }
    if !(c.ch==0 || c.ch==8192) { return false; }
    if !ctx.sg && c.kl>16 { return false; }
    true
}

fn base_score(c:&Choice, ctx:&Ctx)->f32{
    let mut s=0.0;
    if ctx.cols>32_768 || ctx.k>128 { s += if c.use_2ce {0.6} else {-0.2}; }
    if ctx.sg { s += if c.wg==256 {0.3} else {-0.05}; }
    if ctx.k>=64 { s+=match c.kl{32=>0.3,16=>0.05,_=>-0.05} }
    else if ctx.k>=16 { s+=match c.kl{16=>0.2,8=>0.05,_=>-0.1} } else { s += if c.kl==8 {0.15} else {-0.05} }
    if ctx.cols>16_384 { s += if c.ch==8192 {0.25} else {-0.05}; }
    s
}

pub fn solve_soft(ctx:Ctx, cfg:SolveCfg, extra:&[SoftRule])->(Choice,f32){
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let u2s=[false,true]; let wgs=[128,256]; let kls=[8,16,32]; let chs=[0,8192];
    let mut best=Choice::default(); let mut best_s=f32::NEG_INFINITY;
    for &u2 in &u2s { for &wg in &wgs { for &kl in &kls { for &ch in &chs {
        let cand=Choice{use_2ce:u2,wg,kl,ch};
        if !hard_ok(&cand,&ctx){continue}
        let mut s=base_score(&cand,&ctx);
        for r in extra {
            match (r.field, r.value) {
                (Field::Use2ce, Value::B(v)) if cand.use_2ce==v => s+=r.weight,
                (Field::Wg,     Value::U(v)) if cand.wg     ==v => s+=r.weight,
                (Field::Kl,     Value::U(v)) if cand.kl     ==v => s+=r.weight,
                (Field::Ch,     Value::U(v)) if cand.ch     ==v => s+=r.weight,
                _=>{}
            }
        }
        if cfg.noise>0.0 { s += rng.gen::<f32>()*cfg.noise; }
        if s>best_s { best_s=s; best=cand; }
    }}}}
    (best,best_s)
}
