//! Dependency-free offline driver for the same Rust calibration module used by st-nn.
//! Input: u32le frame count, then origin/3 candidates each with 4 rows of 2 f32le logits.
//! Output: count, chosen index (u32le); origin mean, 3 minimum gains, 3 mean gains (f64le).
#[path = "../crates/st-nn/src/causal_lm/route_calibration.rs"]
mod route_calibration;
#[path = "../crates/st-nn/src/causal_lm/sum.rs"]
#[allow(dead_code)]
mod sum;

use route_calibration::{
    BinaryRouteQuartet, CalibrationRole, MatchedRouteCalibration, MAX_ROUTE_CALIBRATION_FRAMES,
};
use std::io::{self, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut argv = std::env::args().skip(1);
    let role = match (argv.next().as_deref(), argv.next()) {
        (Some("first"), None) => CalibrationRole::First,
        (Some("second"), None) => CalibrationRole::Second,
        _ => return Err("usage: native_matched_route_selector first|second".into()),
    };
    let mut input = io::stdin().lock();
    let mut b = [0; 4];
    input.read_exact(&mut b)?;
    let frames = u32::from_le_bytes(b) as usize;
    if !(1..=MAX_ROUTE_CALIBRATION_FRAMES).contains(&frames) {
        return Err("invalid frame count".into());
    }
    let mut c = MatchedRouteCalibration::<3>::new(role)?;
    for _ in 0..frames {
        let mut group = [BinaryRouteQuartet::default(); 4];
        for q in &mut group {
            for p in q {
                for x in p {
                    input.read_exact(&mut b)?;
                    *x = f32::from_le_bytes(b);
                }
            }
        }
        c.add_frame(&group[0], (&group[1..]).try_into()?)?;
    }
    if input.read(&mut b[..1])? != 0 {
        return Err("trailing calibration bytes".into());
    }
    let report = c.selection()?;
    let mut out = io::stdout().lock();
    for n in [report.frames, report.selected] {
        out.write_all(&(n as u32).to_le_bytes())?;
    }
    for x in [report.origin_mean_margin]
        .into_iter()
        .chain(report.worst_gains)
        .chain(report.mean_gains)
    {
        out.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
