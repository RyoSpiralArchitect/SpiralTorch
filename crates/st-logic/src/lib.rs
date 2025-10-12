use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx { pub rows: u32, pub cols: u32, pub k: u32, pub sg: bool }
#[derive(Clone, Copy, Debug, Default)]
pub struct Choice { pub use_2ce: bool, pub wg: u32, pub kl: u32, pub ch: u32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Field { Use2ce, Wg, Kl, Ch }
#[derive(Clone, Copy, Debug)] pub enum Value { B(bool), U(u32) }
#[derive(Clone, Copy, Debug)] pub struct SoftRule { pub field: Field, pub value: Value, pub weight: f32 }

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveCfg { pub noise: f32, pub seed: u64 }

fn hard_ok(choice: &Choice, ctx: &Ctx) -> bool {
    if ![128, 256].contains(&choice.wg) { return false; }
    if ![8, 16, 32].contains(&choice.kl) { return false; }
    if !(choice.ch == 0 || choice.ch == 8192) { return false; }
    if !ctx.sg && choice.kl > 16 { return false; }
    true
}

fn soft_score(choice: &Choice, ctx: &Ctx, extra: &[SoftRule]) -> f32 {
    let mut s = 0.0f32;
    if ctx.cols > 32_768 || ctx.k > 128 { s += if choice.use_2ce { 0.6 } else { -0.2 }; }
    if ctx.sg { s += if choice.wg == 256 { 0.3 } else { -0.05 }; }
    if ctx.k >= 64 { s += match choice.kl { 32 => 0.3, 16 => 0.05, _ => -0.05 } }
    else if ctx.k >= 16 { s += match choice.kl { 16 => 0.2, 8 => 0.05, _ => -0.1 } }
    else { s += if choice.kl == 8 { 0.15 } else { -0.05 } }
    if ctx.cols > 16_384 { s += if choice.ch == 8192 { 0.25 } else { -0.05 }; }
    for r in extra {
        match (r.field, r.value) {
            (Field::Use2ce, Value::B(v)) if choice.use_2ce == v => { s += r.weight; }
            (Field::Wg,     Value::U(v)) if choice.wg      == v => { s += r.weight; }
            (Field::Kl,     Value::U(v)) if choice.kl      == v => { s += r.weight; }
            (Field::Ch,     Value::U(v)) if choice.ch      == v => { s += r.weight; }
            _ => {}
        }
    }
    s
}

pub fn solve_soft(ctx: Ctx, cfg: SolveCfg, extra_rules: &[SoftRule]) -> (Choice, f32) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let u2s = [false, true]; let wgs = [128u32, 256u32]; let kls=[8u32,16u32,32u32]; let chs=[0u32,8192u32];
    let mut best = Choice::default(); let mut best_s = f32::NEG_INFINITY;
    for &u2 in &u2s { for &wg in &wgs { for &kl in &kls { for &ch in &chs {
        let cand = Choice { use_2ce: u2, wg, kl, ch };
        if !hard_ok(&cand, &ctx) { continue; }
        let mut sc = soft_score(&cand, &ctx, extra_rules);
        if cfg.noise > 0.0 { sc += rng.gen::<f32>() * cfg.noise; }
        if sc > best_s { best_s = sc; best = cand; }
    }}}}
    (best, best_s)
}

/// Binomial test (two-sided) p-value for wins out of trials under p0=0.5 (normal approx).
pub fn pvalue_wins(wins:u32, trials:u32)->f64{
    let n=trials.max(1) as f64; let w=wins as f64; let phat=w/n; let se=(0.25/n).sqrt(); let z=((phat-0.5)/se).abs();
    let t = 1.0/(1.0+0.5*(z/2f64.sqrt()).abs());
    let tau=t*( (-(z*z)/2.0).exp() - 1.26551223 + 1.00002368*t + 0.37409196*t*t + 0.09678418*t*t*t
        -0.18628806*t.powi(4) + 0.27886807*t.powi(5) - 1.13520398*t.powi(6) + 1.48851587*t.powi(7)
        -0.82215223*t.powi(8) + 0.17087277*t.powi(9) );
    let erf = 1.0 - tau.copysign(z/2f64.sqrt());
    2.0*(1.0 - erf)
}
