// crates/st-core/src/backend/cuda_loader.rs (skeleton)
#[cfg(feature="cuda")]
pub struct CudaModule {}
#[cfg(feature="cuda")]
pub fn load_ptx_module(_ptx:&[u8]) -> Result<CudaModule,String> { Ok(CudaModule{}) }
