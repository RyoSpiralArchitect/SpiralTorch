// crates/st-core/src/backend/hip_loader.rs  (v1.8.7 skeleton)
#[cfg(feature="hip")]
pub struct HipModule { /* hip-sys / rccl, etc. */ }

#[cfg(feature="hip")]
pub fn load_hsaco_module(_hsaco:&[u8]) -> Result<HipModule,String> {
    // TODO: wire ROCm loader; placeholder for now
    Ok(HipModule{})
}
