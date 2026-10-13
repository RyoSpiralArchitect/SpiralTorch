use super::*;

// Independent integer oracle: numerator * 2^power, rounded once to f32.
fn integer_bits(numerator: i128, power: i32) -> u32 {
    let sign = if numerator < 0 { 0x8000_0000 } else { 0 };
    let magnitude = numerator.unsigned_abs();
    if magnitude == 0 {
        return 0;
    }
    let high = 127 - magnitude.leading_zeros() as i32;
    let exponent = high + power;
    if exponent >= 128 {
        return sign | 0x7f80_0000;
    }
    let quantum = (exponent - 23).max(-149);
    let shift = quantum - power;
    let mut rounded = if shift <= 0 {
        magnitude << -shift
    } else if shift >= 128 {
        0
    } else {
        let quotient = magnitude >> shift;
        let remainder = magnitude & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        quotient + u128::from(remainder > half || (remainder == half && quotient & 1 != 0))
    };
    if rounded == 0 {
        return sign;
    }
    let high = 127 - rounded.leading_zeros() as i32;
    let exponent = high + quantum;
    if exponent >= 128 {
        return sign | 0x7f80_0000;
    }
    if exponent < -126 {
        return sign | ((rounded << (quantum + 149)) as u32);
    }
    if high > 23 {
        rounded >>= high - 23;
    } else {
        rounded <<= 23 - high;
    }
    sign | (((exponent + 127) as u32) << 23) | (rounded as u32 & 0x7f_ffff)
}

#[test]
fn scalar_expansions_match_integer_oracles_and_rounding_boundaries() {
    let Some(d) = device() else { return };
    let context = d.runtime().context();
    let gpu = context.device();
    let mut cases: Vec<Vec<(f32, i32)>> = Vec::new();
    let mut expected = Vec::new();
    let mut add = |terms: Vec<(f32, i32)>, bits| {
        assert!(terms.len() <= 24);
        cases.push(terms);
        expected.push(bits);
    };
    for sign in [1f32, -1.] {
        let sign_bit = sign.to_bits() & 0x8000_0000;
        for tail in [-1f32, 0., 1.] {
            for (base, half, bits) in [
                ((1., 0), (1., -24), 1f32.to_bits()),
                ((0., 0), (1., -150), 0),
                ((1., -148), (1., -150), 2),
            ] {
                add(
                    vec![
                        (sign * base.0, base.1),
                        (sign * half.0, half.1),
                        (sign * tail, -500),
                    ],
                    (bits + u32::from(tail > 0.)) | sign_bit,
                );
            }
            add(
                vec![(sign * f32::MAX, 0), (sign, 103), (sign * tail, -500)],
                (if tail < 0. {
                    f32::MAX.to_bits()
                } else {
                    f32::INFINITY.to_bits()
                }) | sign_bit,
            );
        }
        add(
            vec![
                (sign, 300),
                (sign, 150),
                (sign, -149),
                (-sign, 150),
                (-sign, 300),
            ],
            1 | sign_bit,
        );
        add(
            vec![(sign, 2_000_000_000), (sign, 0), (-sign, 2_000_000_000)],
            1f32.to_bits() | sign_bit,
        );
        add(
            vec![(sign, -2_000_000_000), (sign, 0), (-sign, 0)],
            sign_bit,
        );
        add(
            vec![(sign, 46), (sign * 0.2475, 0), (-sign, 46)],
            0.2475f32.to_bits() | sign_bit,
        );
        add(
            vec![(sign, 0), (-sign, -25), (-sign, -500)],
            (1f32.to_bits() - 1) | sign_bit,
        );
    }
    let mut rng = 0x3de2_93ca_8030_a927u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng
    };
    for case in 0..1024 {
        let base = (case % 601) - 300;
        let mut numerator = 0i128;
        let mut terms = Vec::new();
        for _ in 0..16 {
            let value = ((next() >> 32) & 0x3f_ffff) as i32 - 0x20_0000;
            let exponent = (next() % 81) as i32;
            numerator += i128::from(value) << exponent;
            terms.push((value as f32, base - 40 + exponent));
        }
        // Cancel selected large components only after smaller terms appeared.
        for i in 0..8 {
            let (value, exponent) = terms[i];
            numerator -= i128::from(value as i32) << (exponent - (base - 40));
            terms.push((-value, exponent));
        }
        add(terms, integer_bits(numerator, base - 40));
    }
    let pairs = cases.len();
    let values = 16 * pairs;
    let mut storage = vec![0u32; 4 * values + 2 * 26 * pairs];
    for (p, terms) in cases.iter().enumerate() {
        for (i, &(value, exponent)) in terms.iter().enumerate() {
            let offset = 4 * values + 2 * (26 * p + i);
            storage[offset] = value.to_bits();
            storage[offset + 1] = exponent as u32;
        }
    }
    let scratch = runtime::upload_slice(
        gpu,
        "wave.reduction.cases",
        &storage,
        wgpu::BufferUsages::STORAGE,
    )
    .unwrap();
    let output = runtime::empty_buffer::<u32>(
        gpu,
        "wave.reduction.results",
        pairs,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
    .unwrap();
    let uniform = runtime::upload_slice(
        gpu,
        "wave.reduction.params",
        &[1u32, 8, values as u32, pairs as u32],
        wgpu::BufferUsages::UNIFORM,
    )
    .unwrap();
    let shader = source(
        r#"
ROUNDED_ADD
WIDE_ARITHMETIC
PARAMETER_REDUCTION
struct ProbeParams { batch:u32, steps:u32, values:u32, pairs:u32 };
@group(0) @binding(0) var<storage,read_write> state_gradient:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@group(0) @binding(2) var<uniform> params:ProbeParams;
// Test the raw rounding result, including infinity. Public guards are tested
// through the actual wave pipeline in the sibling integration tests.
fn checked(v:f32)->f32 {return v;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if(id.x>=params.pairs) {return;}
    output[id.x]=bitcast<u32>(wave_reduce(id.x));
}
"#,
    );
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wave.reduction.shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    // Exercise the same explicit binding contract as the production kernels.
    let bind_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wave.reduction.layout"),
        entries: &(0..3)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 2 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wave.reduction.pipeline.layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wave.reduction.pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    let group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wave.reduction.bindings"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scratch.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((pairs as u32).div_ceil(64), 1, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    let actual =
        runtime::read_buffer::<u32>(gpu, context.queue(), &output, pairs, "wave.reduction.read")
            .unwrap();
    for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "case={i}, terms={:?}", cases[i]);
    }
}
