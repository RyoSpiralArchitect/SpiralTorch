use super::*;

#[test]
fn shader_validates_without_a_device() {
    let source = shader_source();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn preflight_rejects_missing_uniform_bindings_before_pipeline_creation() {
    for shape in [&[1, 1][..], &[0, 3][..]] {
        let layout = NdLayout::contiguous(shape).unwrap();
        let limits = wgpu::Limits {
            max_uniform_buffers_per_shader_stage: 0,
            ..Default::default()
        };
        assert!(matches!(
            preflight(&layout, &limits),
            Err(TensorError::Limit("embedding uniform"))
        ));
        let limits = wgpu::Limits {
            max_uniform_buffer_binding_size: std::mem::size_of::<Params>() as u32 - 1,
            ..Default::default()
        };
        assert!(matches!(
            preflight(&layout, &limits),
            Err(TensorError::Limit("embedding uniform"))
        ));
        assert!(preflight(&layout, &wgpu::Limits::default()).is_ok());
    }
}

#[test]
fn preflight_uses_two_dimensional_dispatch_and_rejects_oversize_grids() {
    let limits = wgpu::Limits {
        max_compute_workgroups_per_dimension: 3,
        ..Default::default()
    };
    assert_eq!(
        preflight(&NdLayout::contiguous(&[1025]).unwrap(), &limits).unwrap(),
        [3, 2, 5]
    );
    assert!(preflight(&NdLayout::contiguous(&[2305]).unwrap(), &limits).is_err());
    assert_eq!(
        preflight(&NdLayout::contiguous(&[0]).unwrap(), &limits).unwrap(),
        [1, 1, 1]
    );
}
