use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn log_path() -> std::path::PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".spiraltorch").join("ablog.ndjson")
    } else {
        std::path::PathBuf::from("ablog.ndjson")
    }
}

pub fn ab_log(kind:&str, cols:u32, k:u32, choice:&str, win:bool) {
    let _ = std::fs::create_dir_all(log_path().parent().unwrap());
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log_path()) {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let _ = writeln!(f, r#"{{"ts":{},"kind":"{}","cols":{},"k":{},"choice":"{}","win":{}}}"#, ts, kind, cols, k, choice, if win {"true"} else {"false"});
    }
}

// helpers
pub fn ab_log_topk(cols:u32,k:u32,algo:&str,win:bool){ ab_log("topk",cols,k,algo,win) }
pub fn ab_log_midk(cols:u32,k:u32,mode:&str,win:bool){ ab_log("midk",cols,k,mode,win) }
pub fn ab_log_bottomk(cols:u32,k:u32,mode:&str,win:bool){ ab_log("bottomk",cols,k,mode,win) }
