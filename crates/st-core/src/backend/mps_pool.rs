use once_cell::sync::OnceCell;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

fn round_pow2(x: u64) -> u64 { let mut n=256u64; while n<x { n<<=1; } n }

#[derive(Default, Clone)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evicted: u64,
    pub classes: Vec<(u64, usize)>,
    pub bytes_total: u64,
}

struct Entry { buf: metal::Buffer, age: u64 }
struct Pool {
    max_bytes: u64, bytes_total: u64, max_per_class: usize,
    age: u64, classes: BTreeMap<u64, VecDeque<Entry>>,
    hits: u64, misses: u64, evicted: u64, trace: bool, last: Instant,
}
static POOL: OnceCell<std::sync::Mutex<Pool>> = OnceCell::new();
fn pool()->&'static std::sync::Mutex<Pool>{
    POOL.get_or_init(|| {
        let trace = std::env::var("ST_MPS_POOL_TRACE").ok().map(|v| v=="1").unwrap_or(false);
        std::sync::Mutex::new(Pool{
            max_bytes: std::env::var("ST_MPS_POOL_MAX_BYTES").ok().and_then(|s| s.parse().ok()).unwrap_or(512*1024*1024),
            bytes_total:0, max_per_class: std::env::var("ST_MPS_POOL_MAX_PER_CLASS").ok().and_then(|s| s.parse().ok()).unwrap_or(16),
            age:1, classes:BTreeMap::new(), hits:0, misses:0, evicted:0, trace, last: Instant::now()
        })
    })
}
fn maybe_log(p:&mut Pool){
    if !p.trace {return;}
    if p.last.elapsed() > Duration::from_secs(5){
        let mut cs:Vec<(u64,usize)> = p.classes.iter().map(|(k,v)| (*k, v.len())).collect();
        cs.sort_by_key(|x|x.0);
        eprintln!("[MPS Pool] hits={} misses={} evicted={} bytes={} classes={:?}", p.hits,p.misses,p.evicted,p.bytes_total,cs);
        p.last = Instant::now();
    }
}
pub fn temp(bytes:u64, dev:&metal::Device) -> metal::Buffer {
    let class = round_pow2(bytes.max(256));
    let mut p = pool().lock().unwrap();
    if let Some(q) = p.classes.get_mut(&class) {
        if let Some(e) = q.pop_front() { p.bytes_total -= class; p.hits+=1; maybe_log(&mut p); return e.buf; }
    }
    p.misses+=1; maybe_log(&mut p);
    dev.new_buffer(class, metal::MTLResourceOptions::CPUCacheModeDefaultCache)
}
pub fn recycle(buf: metal::Buffer){
    let class = buf.length();
    let mut p = pool().lock().unwrap();
    let q = p.classes.entry(class).or_insert_with(VecDeque::new);
    if q.len() < p.max_per_class { q.push_back(Entry{ buf, age: p.age }); p.age+=1; p.bytes_total += class; }
    else { p.evicted += 1; }
    while p.bytes_total > p.max_bytes {
        let mut oldest_key=None; let mut oldest_age=u64::MAX;
        for (k,v) in p.classes.iter() {
            if let Some(front)=v.front(){ if front.age < oldest_age { oldest_age=front.age; oldest_key=Some(*k); } }
        }
        if let Some(k)=oldest_key { if let Some(v) = p.classes.get_mut(&k) { if v.pop_front().is_some(){ p.bytes_total -= k; } } }
        else { break; }
    }
    maybe_log(&mut p);
}
pub fn stats() -> PoolStats {
    let p = pool().lock().unwrap();
    let mut cs:Vec<(u64,usize)> = p.classes.iter().map(|(k,v)| (*k, v.len())).collect();
    cs.sort_by_key(|x|x.0);
    PoolStats{ hits:p.hits, misses:p.misses, evicted:p.evicted, classes: cs, bytes_total: p.bytes_total }
}
