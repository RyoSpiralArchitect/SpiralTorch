use rand::{Rng, SeedableRng}; use rand::rngs::StdRng;
#[derive(Clone, Copy, Debug)] pub struct LaneParams { pub lane:i32, pub kl:i32, pub ch:i32, pub prob:f32 }
pub fn sample_lane_params(seed:u64, lane_set:&[i32], kl_set:&[i32], ch_set:&[i32], weights:(f32,f32,f32)) -> LaneParams {
    let mut rng = StdRng::seed_from_u64(seed);
    let pick = |set:&[i32], w:f32| -> i32 { if w>=0.5 { set[set.len()-1] } else if w>=0.25 { set[set.len()/2] } else { set[0] } };
    LaneParams{ lane: pick(lane_set,weights.0), kl: pick(kl_set,weights.1), ch: pick(ch_set,weights.2), prob: 0.5+0.5*rng.gen::<f32>() }
}
