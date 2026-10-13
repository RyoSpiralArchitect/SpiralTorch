use super::*;
use st_kernel_contracts::causal_wave::MAX_DECAY;

pub(super) async fn run(d: &TensorDevice) -> Result<Value> {
    let shape = [3, 1, 2];
    let initial = [1., 0., 1., 0., 1., 0.];
    let raw = d.upload(&[1], &[0.])?;
    let drive = d.upload(&shape, &[0.; 6])?;
    let seed = d.upload(&shape, &[0.; 6])?;
    let forward = drive.causal_zspace_wave(&raw, &raw, &d.upload(&[3, 2], &initial)?, -1.)?;
    let host = CausalWaveForward::new(
        CausalWaveSpec::new(shape, -1.)?,
        &[0.; 6],
        &[0.],
        &[0.],
        &initial,
    )?;
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut checks = Vec::new();
    for exponent in [24, 48, 96] {
        let large = 2f32.powi(exponent);
        for order in permutations {
            for channel in 0..2 {
                let values = [large, 1., -large];
                let mut terminal = [0.; 6];
                for (b, i) in order.iter().enumerate() {
                    terminal[2 * b + channel] = values[*i];
                }
                let g = forward.backward(&seed, &d.upload(&[3, 2], &terminal)?)?;
                let cpu = host.backward(&[0.; 6], &terminal)?;
                let expected = if channel == 0 {
                    MAX_DECAY * 0.25
                } else {
                    (f64::from(MAX_DECAY * 0.5) * f64::from(std::f32::consts::PI)) as f32
                };
                let actual = read(if channel == 0 {
                    g.raw_decay()
                } else {
                    g.raw_phase()
                })
                .await?[0];
                let cpu = if channel == 0 {
                    cpu.raw_decay[0]
                } else {
                    cpu.raw_phase[0]
                };
                if actual.to_bits() != expected.to_bits() || cpu.to_bits() != expected.to_bits() {
                    return Err(format!("batch partial lost signal: exp={exponent}, order={order:?}, channel={channel}, gpu={actual}, cpu={cpu}, expected={expected}").into());
                }
                for t in [g.drive(), g.raw_decay(), g.raw_phase(), g.initial_state()] {
                    if read(t).await?.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite batch VJP".into());
                    }
                }
                checks.push(json!({"terminal_channel":channel,"exponent":exponent,"order":order,
                    "expected":expected,"cpu":cpu,"gpu":actual,"exact_f32":true,"finite_family":true}));
            }
        }
    }
    let mut overflow = Vec::new();
    for channel in 0..2 {
        let x = if channel == 0 { 8. } else { 1. };
        let initial = [x, 0., x, 0.];
        let drive = d.upload(&[2, 1, 2], &[0.; 4])?;
        let seed = d.upload(&[2, 1, 2], &[0.; 4])?;
        let forward = drive.causal_zspace_wave(&raw, &raw, &d.upload(&[2, 2], &initial)?, -1.)?;
        let host = CausalWaveForward::new(
            CausalWaveSpec::new([2, 1, 2], -1.)?,
            &[0.; 4],
            &[0.],
            &[0.],
            &initial,
        )?;
        let mut terminal = [0.; 4];
        terminal[channel] = f32::MAX;
        terminal[2 + channel] = -f32::MAX;
        let g = forward.backward(&seed, &d.upload(&[2, 2], &terminal)?)?;
        let cpu = host.backward(&[0.; 4], &terminal)?;
        if cpu.raw_decay != [0.] || cpu.raw_phase != [0.] {
            return Err("CPU opposite partials did not cancel".into());
        }
        for t in [g.raw_decay(), g.raw_phase()] {
            if read(t).await? != [0.] {
                return Err("GPU opposite partials did not cancel".into());
            }
        }
        for t in [g.drive(), g.initial_state()] {
            if read(t).await?.iter().any(|v| !v.is_finite()) {
                return Err("unreadable finite local VJP".into());
            }
        }
        terminal[2 + channel] = f32::MAX;
        if !matches!(
            host.backward(&[0.; 4], &terminal),
            Err(st_kernel_contracts::causal_wave::CausalWaveError::NonFinite)
        ) {
            return Err("CPU accepted nonrepresentable final sum".into());
        }
        let bad = forward.backward(&seed, &d.upload(&[2, 2], &terminal)?)?;
        for t in [
            bad.drive(),
            bad.raw_decay(),
            bad.raw_phase(),
            bad.initial_state(),
        ] {
            rejected(t).await?;
        }
        overflow.push(json!({"terminal_channel":channel,"opposite_partials_cancel":true,"finite_family":true,"same_sign_whole_family_rejected":true}));
    }
    Ok(
        json!({"checks":checks,"overflow":overflow,"analytic_reference":true,
        "scope":"36 fixed cancellation cases, two recoverable partial overflows and two final-overflow controls; not arbitrary batch-order or exponent invariance"}),
    )
}
