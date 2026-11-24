//! Calibrate on paired training prompts; serve either token on one unpaired prompt.
use st_nn::language::paired_offset::{OffsetError, PairedTokenOffset};

fn main() -> Result<(), OffsetError> {
    // Candidate order is [singular, plural] in both rows of every pair.
    let training = [[[-0.6, 0.0], [-1.4, 0.0]], [[-0.8, 0.0], [-1.6, 0.0]]];
    let fit = PairedTokenOffset::fit_coverage(&training)?;
    assert_eq!(fit.both_correct, 2);
    let gap = fit.offset.adjusted_gap([-0.7, 0.0])?;
    println!(
        "offset={} corrected_singular_minus_plural={gap}",
        fit.offset.value()
    );
    Ok(())
}
