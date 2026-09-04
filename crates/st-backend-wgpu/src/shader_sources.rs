//! Canonical WGSL sources owned by the WGPU backend.
//!
//! Higher layers may specialise these templates, but should not reach into the
//! backend crate's source tree with relative `include_str!` paths.

pub const SOFTMAX_WORKGROUP_WGSL: &str = include_str!("shaders/softmax_workgroup.wgsl");
pub const SOFTMAX_SUBGROUP_WGSL: &str = include_str!("shaders/softmax_subgroup.wgsl");
pub const SOFTMAX_ZSPACE_PROJECTION_WGSL: &str =
    include_str!("shaders/softmax_zspace_projection.wgsl");
pub const SOFTMAX_SPIRAL_CONSENSUS_WGSL: &str =
    include_str!("shaders/softmax_spiral_consensus.wgsl");
pub const FUSED_ATTENTION_ONLINE_WGSL: &str = include_str!("shaders/fused_attention_online.wgsl");

pub const DENSE_MATMUL_WGSL: &str = include_str!("shaders/dense_matmul.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatmulKernel {
    Scalar,
    Register2x2,
}

impl std::str::FromStr for MatmulKernel {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "scalar" => Ok(Self::Scalar),
            "register_2x2" => Ok(Self::Register2x2),
            _ => Err("kernel must be scalar or register_2x2"),
        }
    }
}

impl MatmulKernel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Register2x2 => "register_2x2",
        }
    }

    pub fn outputs_per_thread(self) -> [u32; 2] {
        match self {
            Self::Scalar => [1, 1],
            Self::Register2x2 => [2, 2],
        }
    }

    pub fn workgroup_size(self, [m, n, k]: [u32; 3]) -> Option<[u32; 3]> {
        let [thread_m, thread_n] = self.outputs_per_thread();
        if m == 0 || n == 0 || k == 0 || !m.is_multiple_of(thread_m) || !n.is_multiple_of(thread_n)
        {
            return None;
        }
        Some([n / thread_n, m / thread_m, 1])
    }
}

/// Low-level specializer; callers validate tile and dispatch limits before use.
/// Float32 is shared across host-tensor and resident workloads. Int8 remains an
/// explicit option for the existing approximate host-tensor path only.
pub fn dense_matmul_source(tile: [u32; 3], int8: bool) -> String {
    dense_matmul_source_with_kernel(tile, int8, MatmulKernel::Scalar)
        .expect("caller must validate matmul tile dimensions")
}

pub fn dense_matmul_source_with_kernel(
    tile: [u32; 3],
    int8: bool,
    kernel: MatmulKernel,
) -> Result<String, &'static str> {
    let [m, n, k] = tile;
    let [thread_m, thread_n] = kernel.outputs_per_thread();
    let [wg_x, wg_y, _] = kernel
        .workgroup_size(tile)
        .ok_or("kernel requires positive dimensions and an evenly divisible output tile")?;
    let mut declarations = String::new();
    let mut accumulate = String::new();
    let mut write_output = String::new();
    for row in 0..thread_m {
        accumulate.push_str(&format!(
            "let a{row} = tile_a[(local_row + {row}u) * TILE_K + k];\n"
        ));
    }
    for col in 0..thread_n {
        accumulate.push_str(&format!(
            "let b{col} = tile_b[k * TILE_N + local_col + {col}u];\n"
        ));
    }
    for row in 0..thread_m {
        for col in 0..thread_n {
            declarations.push_str(&format!("var acc_{row}_{col}: f32 = 0.0;\n"));
            accumulate.push_str(&format!(
                "acc_{row}_{col} = acc_{row}_{col} + a{row} * b{col};\n"
            ));
            let guard = if row == 0 && col == 0 {
                "in_bounds".to_owned()
            } else {
                format!("global_row + {row}u < params.rows && global_col + {col}u < params.cols")
            };
            write_output.push_str(&format!(
                "if ({guard}) {{\n\
                 let index = (global_row + {row}u) * params.cols + global_col + {col}u;\n\
                 out[index] = apply_fusions(acc_{row}_{col}, index, global_col + {col}u);\n\
                 }}\n"
            ));
        }
    }
    let rhs_load = if int8 {
        "let stride = (params.inner + 3u) / 4u;
        let base = col * stride + (k >> 2u);
        let word = rhs_packed[base];
        let lane = (k & 3u) * 8u;
        let byte_val = (word >> lane) & 0xFFu;
        let signed_val = bitcast<i32>(byte_val << 24u) >> 24;
        let scale = scales[col];
        return f32(signed_val) * scale;"
    } else {
        "return rhs_packed[k * params.cols + col];"
    };
    Ok(DENSE_MATMUL_WGSL
        .replace("{f16_enable}", "")
        .replace(
            "{rhs_storage_type}",
            if int8 { "array<u32>" } else { "array<f32>" },
        )
        .replace("{rhs_load_body}", rhs_load)
        .replace("{tile_m}", &m.to_string())
        .replace("{tile_n}", &n.to_string())
        .replace("{tile_k}", &k.to_string())
        .replace("{tile_mk}", &format!("{}u", u64::from(m) * u64::from(k)))
        .replace("{tile_nk}", &format!("{}u", u64::from(n) * u64::from(k)))
        .replace("{workgroup_size_x}", &wg_x.to_string())
        .replace("{workgroup_size_y}", &wg_y.to_string())
        .replace("{thread_m}", &thread_m.to_string())
        .replace("{thread_n}", &thread_n.to_string())
        .replace("{accumulator_declarations}", &declarations)
        .replace("{accumulate}", &accumulate)
        .replace("{write_output}", &write_output))
}
