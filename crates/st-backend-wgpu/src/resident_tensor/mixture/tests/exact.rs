use super::*;

#[test]
fn primitive_router_products_round_only_after_cancellation() {
    let Some(d) = device() else { return };
    let context = d.runtime().context();
    let gpu = context.device();
    let mut cases: Vec<[[u32; 4]; 8]> = Vec::new();
    let mut expected = Vec::new();
    let mut case = |terms: &[[f32; 4]], bits| {
        let mut input = [[0; 4]; 8];
        for (row, term) in input.iter_mut().zip(terms) {
            *row = term.map(f32::to_bits);
        }
        cases.push(input);
        expected.push(bits);
    };
    let tiny = f32::from_bits(1);
    for sign in [1.0f32, -1.] {
        let sign_bit = sign.to_bits() & 0x80000000;
        for direction in [-1., 0., 1.] {
            for (base, half_ulp, bits) in [
                (1., [2f32.powi(-24), 1., 1., 1.], 1.0f32.to_bits()),
                (0., [tiny, 0.5, 1., 1.], 0),
                (f32::from_bits(2), [tiny, 0.5, 1., 1.], 2),
            ] {
                let mut half_ulp = half_ulp;
                half_ulp[0] *= sign;
                case(
                    &[
                        [sign * base, 1., 1., 1.],
                        half_ulp,
                        [sign * direction * tiny, tiny, tiny, tiny],
                    ],
                    (bits + u32::from(direction > 0.)) | sign_bit,
                );
            }
            case(
                &[
                    [sign * f32::MAX, 1., 1., 1.],
                    [sign * 2f32.powi(103), 1., 1., 1.],
                    [sign * direction * tiny, tiny, tiny, tiny],
                ],
                (if direction < 0. {
                    f32::MAX.to_bits()
                } else {
                    f32::INFINITY.to_bits()
                }) | sign_bit,
            );
        }
        case(
            &[
                [sign * f32::from_bits(0x7fffff), 1., 1., 1.],
                [sign * tiny, 0.5, 1., 1.],
            ],
            0x800000 | sign_bit,
        );
        case(&[[sign * f32::MAX, 1., 1., 2.]], 0x7f800000 | sign_bit);
        case(
            &[
                [sign * f32::MAX, 1., 1., f32::MAX],
                [sign * tiny, 0.5, 0.5, 4.],
                [-sign * f32::MAX, 1., 1., f32::MAX],
            ],
            1 | sign_bit,
        );
        case(&[[sign * tiny, tiny, tiny, tiny]], sign_bit);
    }
    let a = f32::from_bits(1.0f32.to_bits() + 1);
    case(
        &[
            [4. * a, 0.5 * a, 0.5 * a, a],
            [-1., 1., 1., 1.],
            [-2f32.powi(-21), 1., 1., 1.],
            [-6. * 2f32.powi(-46), 1., 1., 1.],
            [-2f32.powi(-67), 1., 1., 1.],
        ],
        2f32.powi(-92).to_bits(),
    );

    let input = runtime::upload_slice(
        gpu,
        "router.exact.cases",
        &cases,
        wgpu::BufferUsages::STORAGE,
    )
    .unwrap();
    let output = runtime::empty_buffer::<u32>(
        gpu,
        "router.exact.results",
        cases.len(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
    .unwrap();
    let shader = source(
        r#"
EXACT_ROUTER
struct Case { terms:array<vec4<u32>,8> };
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if(id.x>=arrayLength(&cases)) {return;}
    var sum=RouterSum();
    for(var i=0u;i<8u;i++) {
        router_product(&sum,bitcast<vec4<f32>>(cases[id.x].terms[i]),false);
    }
    result[id.x]=bitcast<u32>(router_float(&sum));
}
"#,
    );
    let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("router.exact.shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("router.exact.pipeline"),
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    let group = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("router.exact.bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    let actual = runtime::read_buffer::<u32>(
        gpu,
        context.queue(),
        &output,
        cases.len(),
        "router.exact.read",
    )
    .unwrap();
    for (i, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "case={i}, factors={:?}", cases[i]);
    }
}
