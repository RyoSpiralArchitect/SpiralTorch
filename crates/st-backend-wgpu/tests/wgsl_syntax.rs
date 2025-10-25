use naga::front::wgsl::parse_str;

const SHADERS: &[(&str, &str)] = &[
    (
        "wgpu_compaction_1ce",
        include_str!("../src/shaders/wgpu_compaction_1ce.wgsl"),
    ),
    (
        "wgpu_compaction_apply",
        include_str!("../src/shaders/wgpu_compaction_apply.wgsl"),
    ),
    (
        "wgpu_compaction_apply_pass",
        include_str!("../src/shaders/wgpu_compaction_apply_pass.wgsl"),
    ),
    (
        "wgpu_compaction_scan",
        include_str!("../src/shaders/wgpu_compaction_scan.wgsl"),
    ),
    (
        "wgpu_compaction_scan_pass",
        include_str!("../src/shaders/wgpu_compaction_scan_pass.wgsl"),
    ),
    (
        "midk_bottomk_compaction",
        include_str!("../src/shaders/midk_bottomk_compaction.wgsl"),
    ),
    (
        "topk_keepk_workgroup",
        include_str!("../src/shaders/topk_keepk_workgroup.wgsl"),
    ),
    (
        "topk_keepk_subgroup",
        include_str!("../src/shaders/topk_keepk_subgroup.wgsl"),
    ),
    (
        "topk_keepk_subgroup_1ce",
        include_str!("../src/shaders/topk_keepk_subgroup_1ce.wgsl"),
    ),
    (
        "topk_keepk_subgroup_1ce_large",
        include_str!("../src/shaders/topk_keepk_subgroup_1ce_large.wgsl"),
    ),
    (
        "softmax_workgroup",
        include_str!("../src/shaders/softmax_workgroup.wgsl"),
    ),
    (
        "softmax_subgroup",
        include_str!("../src/shaders/softmax_subgroup.wgsl"),
    ),
    (
        "row_softmax_subgroup",
        include_str!("../src/shaders/row_softmax_subgroup.wgsl"),
    ),
    (
        "fused_attention",
        include_str!("../src/shaders/fused_attention_online.wgsl"),
    ),
    ("reduce_db", include_str!("../src/shaders/reduce_db.wgsl")),
    (
        "fused_gelu_back",
        include_str!("../src/shaders/fused_gelu_back.wgsl"),
    ),
    (
        "nerf_raymarch",
        include_str!("../src/shaders/nerf_raymarch.wgsl"),
    ),
    (
        "nerf_volume_utils",
        include_str!("../src/shaders/nerf_volume_utils.wgsl"),
    ),
    ("nd_indexer", include_str!("../src/shaders/nd_indexer.wgsl")),
    (
        "transforms_horizontal_flip",
        include_str!("../shaders/transforms/horizontal_flip.wgsl"),
    ),
    (
        "transforms_resize",
        include_str!("../shaders/transforms/resize.wgsl"),
    ),
    (
        "transforms_center_crop",
        include_str!("../shaders/transforms/center_crop.wgsl"),
    ),
    (
        "transforms_color_jitter",
        include_str!("../shaders/transforms/color_jitter.wgsl"),
    ),
];

#[test]
fn all_backend_shaders_parse() {
    for (name, source) in SHADERS {
        parse_str(source).unwrap_or_else(|err| panic!("{name} failed: {err}"));
    }
}
