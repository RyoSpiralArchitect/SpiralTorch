#[cfg(all(feature="hip", feature="hip-real"))] fn maybe_sync(){ let _ = st_backend_hip::real::device_synchronize(); }
#[cfg(not(all(feature="hip", feature="hip-real")))] fn maybe_sync(){}

use rand::{Rng, SeedableRng}; use rand::rngs::StdRng;

#[derive(Clone, Copy, Debug)] pub struct LaneParams { pub lane:i32, pub kl:i32, pub ch:i32, pub prob:f32 }

pub fn sample_lane_params(seed:u64, lane_set:&[i32], kl_set:&[i32], ch_set:&[i32], weights:(f32,f32,f32)) -> LaneParams {
    let mut rng = StdRng::seed_from_u64(seed);
    let pick = |set:&[i32], w:f32| -> i32 { if w>=0.5 { set[set.len()-1] } else if w>=0.25 { set[set.len()>>1] } else { set[0] } };
    LaneParams{ lane: pick(lane_set,weights.0), kl: pick(kl_set,weights.1), ch: pick(ch_set,weights.2), prob: 0.5+0.5*rng.gen::<f32>() }
}

fn median_i32(v:&mut [i32])->i32{ v.sort_unstable(); let n=v.len(); if n==0 {0} else { v[n/2] } }

pub fn consensus_lane_params(mut p:LaneParams) -> LaneParams {
    let agg = std::env::var("SPIRAL_UNISON_AGG").unwrap_or_else(|_| "mean".into());
    #[cfg(feature="hip")]
    {
        #[cfg(feature="hip-real")] use st_backend_hip::rccl_comm::init_rccl_from_env;
#[cfg(feature="hip")] use st_backend_hip::real::{HipStream, HipPtr, malloc, free, memcpy_h2d_async, memcpy_d2h_async, allgather_i32_dev};
        if let Ok(comm) = init_rccl_from_env() {
            let world = comm.world as usize;
            let stream = HipStream::create().ok().expect("HipStream");
            let h_local = [p.lane, p.kl, p.ch];
            let bytes = 3*std::mem::size_of::<i32>();
            let d_send: HipPtr = malloc(bytes).expect("malloc send");
            let d_recv: HipPtr = malloc(bytes*world).expect("malloc recv");
            unsafe{ memcpy_h2d_async(d_send, h_local.as_ptr() as *const u8, bytes, &stream).ok(); }
            allgather_i32_dev(comm.comm, &stream, d_send, d_recv, 3).ok();
            let mut all = vec![0i32; 3*world];
            unsafe{ memcpy_d2h_async(all.as_mut_ptr() as *mut u8, d_recv, bytes*world, &stream).ok(); }
free(d_send); free(d_recv);
            if agg=="median" {
                let mut lanes: Vec<i32> = all.iter().step_by(3).cloned().collect();
                let mut kls  : Vec<i32> = all.iter().skip(1).step_by(3).cloned().collect();
                let mut chs  : Vec<i32> = all.iter().skip(2).step_by(3).cloned().collect();
                p.lane = median_i32(&mut lanes); p.kl = median_i32(&mut kls); p.ch = median_i32(&mut chs);
            } else {
                let mut s=(0i64,0i64,0i64);
                for i in 0..world { s.0+=all[i*3+0] as i64; s.1+=all[i*3+1] as i64; s.2+=all[i*3+2] as i64; }
                p.lane = (s.0 as f32 / world as f32).round() as i32;
                p.kl   = (s.1 as f32 / world as f32).round() as i32;
                p.ch   = (s.2 as f32 / world as f32).round() as i32;
            }
        }
    }
    #[cfg(feature="kv-redis")]
    {
        if let Ok(url) = std::env::var("REDIS_URL") {
            if let Ok(samples) = st_kv::redis_lrange(            if let Ok(samples) = st_kv::redis_lrange(redis_lrange(&url, "spiral:heur:lparams", 16)url, "spiral:heur:lparams", -16, -1) {url, "spiral:heur:lparams", -16, -1) {
                let mut lanes=Vec::new(); let mut kls=Vec::new(); let mut chs=Vec::new();
                for s in samples { if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s){
                    lanes.push(v["lane"].as_i64().unwrap_or(0) as i32);
                    kls.push  (v["kl"].as_i64().unwrap_or(0) as i32);
                    chs.push  (v["ch"].as_i64().unwrap_or(0) as i32);
                }}
                if !lanes.is_empty(){
                    if agg=="median" { p.lane = median_i32(&mut lanes); p.kl = median_i32(&mut kls); p.ch = median_i32(&mut chs); }
                    else {
                        let n=lanes.len() as f32;
                        p.lane = ((lanes.iter().copied().map(|x|x as i64).sum::<i64>() as f32)/n).round() as i32;
                        p.kl   = ((kls  .iter().copied().map(|x|x as i64).sum::<i64>() as f32)/n).round() as i32;
                        p.ch   = ((chs  .iter().copied().map(|x|x as i64).sum::<i64>() as f32)/n).round() as i32;
                    }
                }
            }
        }
    }
    p
}
