# RCCL Bootstrap: uniqueId (env/file), TTL/Retry, MPI-like env

**Rank/World**:
```bash
export WORLD_SIZE=4
export RANK=$LOCAL_RANK  # or OMPI_COMM_WORLD_RANK / PMI_RANK
```

**uniqueId**:
- **File**: `export RCCL_UNIQUE_ID_FILE=/tmp/rccl_uid.bin`
  - Rank0: rcclGetUniqueId() → write file
  - Others: wait (TTL=RCCL_UID_TTL_MS, default 30000ms; retry=RCCL_UID_RETRY_MS, default 100ms)
- **Env**: `export RCCL_UNIQUE_ID_B64=...` (rank0 can print helper line if unset)

**Build HIP real**:
```bash
export ROCM_PATH=/opt/rocm
export HIPCC=$ROCM_PATH/bin/hipcc
cargo build -p st-core --features hip,st-backend-hip/hip-real --release
```
