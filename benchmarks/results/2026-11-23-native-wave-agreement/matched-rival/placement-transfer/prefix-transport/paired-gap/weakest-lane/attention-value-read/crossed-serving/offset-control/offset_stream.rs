//! Small lossless study bridge to the real Rust language calibration module.
use paired_offset::{PairedTokenLogits, PairedTokenOffset};
use std::error::Error;
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut words = line.split_ascii_whitespace();
        let op = words.next().ok_or("missing operation")?;
        let id = words.next().ok_or("missing cohort")?;
        let b = if op == "S" {
            PairedTokenOffset::new(f64::from_bits(u64::from_str_radix(
                words.next().ok_or("missing offset")?,
                16,
            )?))?
        } else {
            PairedTokenOffset::default()
        };
        let vals: Vec<_> = words
            .map(|w| u32::from_str_radix(w, 16).map(f32::from_bits))
            .collect::<Result<_, _>>()?;
        let width = match op {
            "F" => 8,
            "S" | "B" => 4,
            _ => return Err("unknown operation".into()),
        };
        if vals.is_empty() || vals.len() % width != 0 {
            return Err("expected nonempty complete pairs".into());
        }
        let row = |v: &[f32]| -> PairedTokenLogits { [[v[0], v[1]], [v[2], v[3]]] };
        let pairs: Vec<_> = vals.chunks_exact(width).map(row).collect();
        write!(out, "{id} {op}")?;
        match op {
            "F" => {
                let refs: Vec<_> = vals.chunks_exact(width).map(|v| row(&v[4..])).collect();
                let fit = PairedTokenOffset::fit_coverage(&pairs)?;
                let mimic = PairedTokenOffset::match_reference(&pairs, &refs)?;
                write!(
                    out,
                    " {:016x} {:016x} {} {} {}",
                    fit.offset.value().to_bits(),
                    mimic.value().to_bits(),
                    fit.pairs,
                    fit.separable,
                    fit.both_correct
                )?;
            }
            "S" => {
                for p in pairs {
                    for v in p {
                        write!(out, " {:016x}", b.adjusted_gap(v)?.to_bits())?;
                    }
                }
            }
            "B" => {
                let fit = PairedTokenOffset::fit_coverage(&pairs)?;
                write!(out, " {} {}", fit.separable, fit.both_correct)?;
            }
            _ => unreachable!(),
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}
