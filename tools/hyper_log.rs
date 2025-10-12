
use std::{fs::File, io::Write};
use chrono::Utc;

pub fn log_hyper(name:&str, lr:f32, reg:f32, h_unrolled:f32, h_impl:f32){
    let ts = Utc::now().to_rfc3339();
    let mut f = File::options().create(true).append(true).open("hyper_log.csv").unwrap();
    writeln!(f, "{ts},{name},{lr:.6},{reg:.6},{h_unrolled:.6},{h_impl:.6}").unwrap();
}

fn main(){
    log_hyper("demo", 1e-2, 1e-4, -0.123, -0.115);
    eprintln!("wrote hyper_log.csv");
}
