use super::*;
use ndarray::array;

#[test]
fn detects_boundary_presence() {
    let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let sig = gauge.analyze(&mask);
    assert!(sig.has_interface());
    assert_eq!(sig.kappa_d, 2.0 * std::f32::consts::PI);
    assert!((sig.perimeter_density[IxDyn(&[1, 1])] - sig.kappa_d).abs() < 1e-5);
    assert_eq!(sig.perimeter_density[IxDyn(&[0, 0])], 0.0);
    assert!((sig.physical_radius - 1.0).abs() < 1e-6);
}

#[test]
fn oriented_normals_require_label() {
    let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let sig = gauge.analyze_with_label(&mask, Some(&c_prime));
    let orient = sig.orientation.expect("orientation missing");
    let normal_y = orient[IxDyn(&[0, 1, 1])];
    let normal_x = orient[IxDyn(&[1, 1, 1])];
    assert!(normal_y.abs() > 0.5);
    assert!(normal_x.abs() < 1e-3);
}

#[test]
fn z_lift_produces_oriented_bias() {
    let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let sig = gauge.analyze_with_label(&mask, Some(&c_prime));
    let projector = LeechProjector::new(24, 0.5);
    let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
    let pulse = lift.project(&sig);
    let (above, here, beneath) = pulse.band_energy;
    assert!(above > beneath);
    assert!(here >= 0.0);
    assert!(pulse.z_bias > 0.0);
    let feedback = pulse.clone().into_softlogic_feedback();
    assert_eq!(feedback.band_energy, pulse.band_energy);
    assert_eq!(feedback.z_signal, pulse.z_bias);
}

#[test]
fn z_lift_remains_neutral_without_orientation() {
    let mask = array![[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]].into_dyn();
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let sig = gauge.analyze(&mask);
    let projector = LeechProjector::new(24, 0.5);
    let lift = InterfaceZLift::new(&[0.0, 1.0], projector);
    let pulse = lift.project(&sig);
    let (above, here, beneath) = pulse.band_energy;
    assert!(above <= f32::EPSILON);
    assert!(beneath <= f32::EPSILON);
    assert!(here > 0.0);
    assert_eq!(pulse.z_bias, 0.0);
}

#[test]
fn curvature_detects_flat_and_curved_interfaces() {
    let flat = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn();
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let flat_sig = gauge.analyze(&flat);
    let flat_curvature = flat_sig.mean_curvature[IxDyn(&[1, 1])];
    assert!(
        flat_curvature.abs() < 1e-3,
        "flat interface should be near-zero curvature"
    );

    let curved = array![
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ]
    .mapv(|v| v as f32)
    .into_dyn();
    let curved_sig = gauge.analyze(&curved);
    let max_curvature = curved_sig
        .mean_curvature
        .iter()
        .fold(0.0f32, |acc, v| acc.max(*v));
    assert!(
        max_curvature > 0.05,
        "curved interface should register positive curvature"
    );
}

#[test]
fn multiradius_analysis_returns_distinct_signatures() {
    let mask = array![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]
    .into_dyn();
    let gauge = InterfaceGauge::new(1.0, 1.5);
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let signatures = gauge.analyze_multiradius(&mask, Some(&c_prime), &[0.75, 1.5, 2.5]);
    assert_eq!(signatures.len(), 3);
    assert!(signatures[0].radius <= signatures[1].radius);
    assert!(signatures[1].radius <= signatures[2].radius);
    assert!((signatures[1].physical_radius - 1.5).abs() < 1e-6);
    assert!(signatures.iter().all(|sig| sig.orientation.is_some()));
}

#[test]
fn conductor_fuses_multiscale_pulses_with_smoothing() {
    let mask = array![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]
    .into_dyn();
    let mut flipped = mask.clone();
    flipped[[1, 2]] = 0.0;
    flipped[[2, 2]] = 0.0;
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let c_prime_neg = c_prime.mapv(|v| -v);

    let gauge_fine = InterfaceGauge::new(1.0, 1.0);
    let gauge_coarse = InterfaceGauge::new(1.0, 2.5);
    let projector = LeechProjector::new(24, 0.5);
    let lift = InterfaceZLift::new(&[1.0, 0.0], projector).with_bias_gain(0.5);
    let mut conductor = InterfaceZConductor::new(vec![gauge_fine, gauge_coarse], lift).with_smoothing(0.5);

    let first = conductor.step(&mask, Some(&c_prime), None, None);
    assert!(first.has_interface());
    assert!(first.fused_pulse.z_bias > 0.0);
    assert_eq!(first.qualities.len(), first.pulses.len());
    assert!(first.budget_scale > 0.0);

    let second = conductor.step(&flipped, Some(&c_prime_neg), None, None);
    let raw_second = InterfaceZPulse::aggregate(&second.pulses);
    assert!(raw_second.z_bias < 0.0);
    let (above, here, beneath) = second.fused_pulse.band_energy;
    assert!(second.fused_z.pulse.band_energy == second.fused_pulse.band_energy);
    assert!((second.fused_z.pulse.support.leading - above).abs() < 1e-6);
    assert!((second.fused_z.pulse.support.central - here).abs() < 1e-6);
    assert!((second.fused_z.pulse.support.trailing - beneath).abs() < 1e-6);
    assert!(second.fused_z.z > raw_second.z_bias);
    assert_eq!(second.feedback.band_energy, second.fused_pulse.band_energy);
    assert_eq!(second.qualities.len(), second.pulses.len());
    assert!(second.budget_scale > 0.0);
}

#[derive(Debug)]
struct HalfPolicy;

impl ZSourcePolicy for HalfPolicy {
    fn quality(&self, _: &InterfaceZPulse) -> f32 {
        0.5
    }
}

#[test]
fn custom_policy_scales_fused_pulse() {
    let mask = array![[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],].into_dyn();
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let projector = LeechProjector::new(16, 0.5);
    let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
    let mut conductor = InterfaceZConductor::new(vec![gauge], lift).with_policy(HalfPolicy);

    let report = conductor.step(&mask, Some(&c_prime), None, None);
    let raw = InterfaceZPulse::aggregate(&report.pulses);
    assert_eq!(report.qualities.len(), 1);
    let quality = report.qualities[0];
    assert!((quality - 0.5).abs() < 1e-6);
    assert!((report.fused_pulse.support - raw.support * 0.5).abs() < 1e-6);
    assert!((report.fused_pulse.z_bias - raw.z_bias * 0.5).abs() < 1e-6);
}

#[test]
fn budget_policy_clamps_bias() {
    let mask = array![[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],].into_dyn();
    let c_prime = mask.mapv(|v| if v > 0.5 { 1.0 } else { -1.0 });
    let gauge = InterfaceGauge::new(1.0, 1.0);
    let projector = LeechProjector::new(16, 0.5);
    let lift = InterfaceZLift::new(&[1.0, 0.0], projector);
    let mut conductor = InterfaceZConductor::new(vec![gauge], lift).with_budget_policy(BudgetPolicy::new(0.02));

    let report = conductor.step(&mask, Some(&c_prime), None, None);
    assert!(report.budget_scale <= 1.0);
    assert!(report.fused_pulse.z_bias.abs() <= 0.02 + 1e-6);
}

#[test]
fn band_policy_demotes_unbalanced_energy() {
    let pulse = InterfaceZPulse {
        support: 1.0,
        interface_cells: 1.0,
        band_energy: (0.9, 0.05, 0.05),
        drift: 0.4,
        z_bias: 0.3,
        ..InterfaceZPulse::default()
    };
    let policy = BandPolicy::new([0.2, 0.2, 0.2]);
    let quality = policy.project_quality(&pulse);
    assert!(quality < 1.0);
}

#[test]
fn maxwell_policy_prefers_confident_z_scores() {
    let mut pulse = InterfaceZPulse {
        support: 1.0,
        interface_cells: 1.0,
        band_energy: (0.6, 0.2, 0.2),
        drift: 0.4,
        z_bias: 0.2,
        source: ZSource::Maxwell,
        z_score: Some(2.5),
        standard_error: Some(0.05),
        ..InterfaceZPulse::default()
    };
    let policy = MaxwellPolicy::default();
    let strong = policy.quality(&pulse);
    pulse.z_score = Some(0.5);
    let weak = policy.quality(&pulse);
    assert!(strong > weak);
    pulse.z_score = None;
    assert!(
        (policy.quality(&pulse) - DefaultZSourcePolicy::new().quality(&pulse)).abs() < 1e-6
    );
}

#[test]
fn realgrad_policy_scales_with_residual_and_band() {
    let mut pulse = InterfaceZPulse {
        support: 1.0,
        interface_cells: 1.0,
        band_energy: (0.2, 0.4, 0.4),
        drift: 0.1,
        z_bias: 0.05,
        source: ZSource::RealGrad,
        residual_p90: Some(0.05),
        quality_hint: Some(0.8),
        has_low_band: true,
        ..InterfaceZPulse::default()
    };
    let policy = RealGradPolicy::default();
    let baseline = policy.quality(&pulse);
    pulse.residual_p90 = Some(0.5);
    let noisy = policy.quality(&pulse);
    assert!(baseline > noisy);
    pulse.has_low_band = false;
    let no_bonus = policy.quality(&pulse);
    assert!(baseline > no_bonus);
}

#[derive(Clone, Debug)]
struct FixedPolicy(f32);

impl ZSourcePolicy for FixedPolicy {
    fn quality(&self, _: &InterfaceZPulse) -> f32 {
        self.0
    }
}

#[test]
fn composite_policy_routes_per_source() {
    let composite =
        CompositePolicy::new(FixedPolicy(0.5)).with(ZSource::RealGrad, FixedPolicy(0.9));
    let mut pulse = InterfaceZPulse {
        support: 1.0,
        interface_cells: 1.0,
        band_energy: (0.3, 0.3, 0.4),
        drift: 0.2,
        z_bias: 0.1,
        ..InterfaceZPulse::default()
    };
    assert!((composite.quality(&pulse) - 0.5).abs() < 1e-6);
    pulse.source = ZSource::RealGrad;
    assert!((composite.quality(&pulse) - 0.9).abs() < 1e-6);
}
