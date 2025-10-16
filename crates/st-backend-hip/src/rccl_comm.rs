// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::real::{RcclComm, RcclUniqueId};
use crate::HipErr;
use base64::{engine::general_purpose, Engine as _};
use std::{
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};

extern "C" {
    fn rcclGetUniqueId(id: *mut RcclUniqueId) -> i32;
    fn rcclCommInitRank(comm: *mut RcclComm, nranks: i32, id: RcclUniqueId, rank: i32) -> i32;
    fn rcclCommDestroy(comm: RcclComm) -> i32;
}

#[repr(C)]
pub struct RcclUniqueId {
    pub internal: [u8; 128],
}
#[repr(C)]
pub struct RcclComm {
    pub internal: *mut std::ffi::c_void,
}

pub struct RcclCommGuard {
    pub comm: RcclComm,
    pub world: i32,
    pub rank: i32,
}
impl Drop for RcclCommGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = rcclCommDestroy(self.comm);
        }
    }
}

fn read_env_rank() -> Option<(i32, i32)> {
    let world = std::env::var("WORLD_SIZE")
        .or_else(|_| std::env::var("OMPI_COMM_WORLD_SIZE"))
        .or_else(|_| std::env::var("PMI_SIZE"))
        .ok()?;
    let rank = std::env::var("RANK")
        .or_else(|_| std::env::var("OMPI_COMM_WORLD_RANK"))
        .or_else(|_| std::env::var("PMI_RANK"))
        .ok()?;
    Some((world.parse().ok()?, rank.parse().ok()?))
}

fn write_file(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = Path::new(path).parent() {
        let _ = fs::create_dir_all(dir);
    }
    fs::write(path, bytes)
}
fn read_file_with_ttl(path: &str, ttl_ms: u64, retry_ms: u64) -> Option<Vec<u8>> {
    let deadline = Instant::now() + Duration::from_millis(ttl_ms);
    while Instant::now() < deadline {
        if let Ok(b) = fs::read(path) {
            return Some(b);
        }
        thread::sleep(Duration::from_millis(retry_ms));
    }
    None
}

pub fn init_rccl_from_env() -> Result<RcclCommGuard, HipErr> {
    let (world_sz, rank) =
        read_env_rank().ok_or_else(|| HipErr::Other("RCCL: WORLD_SIZE/RANK missing".into()))?;
    let ttl_ms: u64 = std::env::var("RCCL_UID_TTL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30000);
    let retry_ms: u64 = std::env::var("RCCL_UID_RETRY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let mut id = RcclUniqueId {
        internal: [0u8; 128],
    };
    if let Ok(b64) = std::env::var("RCCL_UNIQUE_ID_B64") {
        let bytes = general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| HipErr::Other(format!("b64: {e}")))?;
        if bytes.len() != 128 {
            return Err(HipErr::Other("uniqueId size != 128".into()));
        }
        id.internal.copy_from_slice(&bytes[..128]);
    } else if let Ok(path) = std::env::var("RCCL_UNIQUE_ID_FILE") {
        if rank == 0 {
            unsafe {
                if rcclGetUniqueId(&mut id as *mut _) != 0 {
                    return Err(HipErr::Other("rcclGetUniqueId".into()));
                }
            }
            write_file(&path, &id.internal)
                .map_err(|e| HipErr::Other(format!("write uniqueId: {e}")))?;
        } else {
            let bytes = read_file_with_ttl(&path, ttl_ms, retry_ms)
                .ok_or_else(|| HipErr::Other("wait uniqueId file timeout".into()))?;
            if bytes.len() != 128 {
                return Err(HipErr::Other("uniqueId size != 128".into()));
            }
            id.internal.copy_from_slice(&bytes[..128]);
        }
    } else {
        if rank == 0 {
            unsafe {
                if rcclGetUniqueId(&mut id as *mut _) != 0 {
                    return Err(HipErr::Other("rcclGetUniqueId".into()));
                }
            }
            let b64 = general_purpose::STANDARD.encode(&id.internal);
            eprintln!("[rccl] export RCCL_UNIQUE_ID_B64='{}'", b64);
        } else {
            return Err(HipErr::Other(
                "set RCCL_UNIQUE_ID_B64 or RCCL_UNIQUE_ID_FILE".into(),
            ));
        }
    }
    let mut comm = RcclComm {
        internal: std::ptr::null_mut(),
    };
    unsafe {
        let rc = rcclCommInitRank(&mut comm as *mut _, world_sz, id, rank);
        if rc != 0 {
            return Err(HipErr::Other(format!("rcclCommInitRank rc={}", rc)));
        }
    }
    Ok(RcclCommGuard {
        comm,
        world: world_sz,
        rank,
    })
}
