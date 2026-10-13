use super::*;
use st_kernel_contracts::causal_wave::{CausalWaveError, MAX_DECAY, MAX_STEPS};

pub(super) async fn run(d: &TensorDevice) -> Result<Value> {
    let mut checks = Vec::new();
    let large = 2f32.powi(124);
    let small = f32::from_bits(0x0020_0000); // 2^-128, without host powi underflow.
    let raw = d.upload(&[1], &[0.])?;
    for case in 0..3 {
        let (name, shape, drive, initial, terminal, expected, channel) = if case < 2 {
            let mut terminal = vec![0.; 10];
            for (b, value) in [large, 0.25, small, -0.25, -large].into_iter().enumerate() {
                terminal[2 * b + case] = value;
            }
            let coefficient = if case == 0 {
                f64::from(MAX_DECAY * 0.25)
            } else {
                f64::from(MAX_DECAY * 0.5) * f64::from(std::f32::consts::PI)
            };
            (
                if case == 0 {
                    "nested_decay"
                } else {
                    "nested_phase"
                },
                [5, 1, 2],
                vec![0.; 10],
                vec![1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                terminal,
                (coefficient * f64::from(small)) as f32,
                case,
            )
        } else {
            let rho = f64::from(MAX_DECAY * 0.5);
            (
                "time_then_batch",
                [2, 2, 2],
                vec![large, 0., 0., 0., large, 0., 0., 0.],
                vec![1., 0., 0., 0.],
                vec![0., 1., 0., -1.],
                (rho * rho * f64::from(std::f32::consts::PI)) as f32,
                1,
            )
        };
        let spec = CausalWaveSpec::new(shape, -1.)?;
        let seed = vec![0.; spec.len()];
        let cpu = CausalWaveForward::new(spec, &drive, &[0.], &[0.], &initial)?
            .backward(&seed, &terminal)?;
        let forward = d.upload(&shape, &drive)?.causal_zspace_wave(
            &raw,
            &raw,
            &d.upload(&[shape[0], 2], &initial)?,
            -1.,
        )?;
        let gpu = forward.backward(
            &d.upload(&shape, &seed)?,
            &d.upload(&[shape[0], 2], &terminal)?,
        )?;
        let host = if channel == 0 {
            cpu.raw_decay[0]
        } else {
            cpu.raw_phase[0]
        };
        let resident = read(if channel == 0 {
            gpu.raw_decay()
        } else {
            gpu.raw_phase()
        })
        .await?[0];
        if host.to_bits() != expected.to_bits() || resident.to_bits() != expected.to_bits() {
            return Err(format!(
                "{name}: lost reduction tail: cpu={host}, gpu={resident}, expected={expected}"
            )
            .into());
        }
        for t in [
            gpu.drive(),
            gpu.raw_decay(),
            gpu.raw_phase(),
            gpu.initial_state(),
        ] {
            if read(t).await?.iter().any(|v| !v.is_finite()) {
                return Err(format!("{name}: nonfinite gradient family").into());
            }
        }
        checks.push(json!({"name":name,"expected":expected,"cpu":host,"gpu":resident,
            "expected_bits":expected.to_bits(),"cpu_bits":host.to_bits(),"gpu_bits":resident.to_bits(),
            "finite_family":true}));
    }
    let sequence_bound = CausalWaveSpec::new([1, MAX_STEPS, 2], -1.).is_ok()
        && CausalWaveSpec::new([1, MAX_STEPS + 1, 2], -1.) == Err(CausalWaveError::SequenceLength)
        && CausalWaveSpec::new([1, 20_000_000, 2], -1.) == Err(CausalWaveError::SequenceLength);
    if !sequence_bound {
        return Err("wave sequence exponent bound missing".into());
    }
    Ok(
        json!({"checks":checks,"analytic_reference":true,"sequence_bound":sequence_bound,
        "max_steps":MAX_STEPS,"scope":"Stored-contraction time/batch cancellation, not exact recurrence or language quality"}),
    )
}
