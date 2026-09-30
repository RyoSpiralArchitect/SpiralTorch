use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use spiral_config::determinism;

// Keep the rand 0.8 StdRng sequence, but give input checkpoints an explicit
// algorithm rather than depending on the unspecified future StdRng choice.
pub(super) fn from_optional(seed: Option<u64>, label: &str) -> ChaCha12Rng {
    match seed {
        Some(seed) => ChaCha12Rng::seed_from_u64(seed),
        None => {
            let config = determinism::config();
            if config.enabled {
                ChaCha12Rng::seed_from_u64(config.seed_for(label))
            } else {
                ChaCha12Rng::from_entropy()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, RngCore};

    #[test]
    fn explicit_input_rng_preserves_existing_seeded_draws() {
        for seed in [0, 17, 29, 43, u64::MAX] {
            let mut old = StdRng::seed_from_u64(seed);
            let mut new = from_optional(Some(seed), "unused");
            for _ in 0..1000 {
                assert_eq!(old.next_u32(), new.next_u32());
                assert_eq!(old.next_u64(), new.next_u64());
                assert_eq!(old.gen::<f32>().to_bits(), new.gen::<f32>().to_bits());
                assert_eq!(old.gen_range(0..=50000_u64), new.gen_range(0..=50000_u64));
            }
        }
    }
}
