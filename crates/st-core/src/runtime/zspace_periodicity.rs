// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared periodic-suffix semantics for Z-space training and evidence.

pub(crate) const ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD: usize = 16;
pub(crate) const ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PeriodicSuffix {
    pub(crate) period: usize,
    pub(crate) token_count: usize,
    pub(crate) repeated_token_count: usize,
    pub(crate) repetition_count: usize,
}

pub(crate) fn longest_periodic_suffix(
    tokens: &[u64],
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    longest_periodic_suffix_by(
        tokens.len(),
        |index| tokens[index],
        maximum_period,
        minimum_repetitions,
    )
}

pub(crate) fn longest_periodic_suffix_with_appended_token(
    tokens: &[u64],
    appended_token: u64,
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    longest_periodic_suffix_by(
        tokens.len() + 1,
        |index| {
            if index == tokens.len() {
                appended_token
            } else {
                tokens[index]
            }
        },
        maximum_period,
        minimum_repetitions,
    )
}

fn longest_periodic_suffix_by(
    token_count: usize,
    token_at: impl Fn(usize) -> u64,
    maximum_period: usize,
    minimum_repetitions: usize,
) -> Option<PeriodicSuffix> {
    if maximum_period == 0 || minimum_repetitions < 2 {
        return None;
    }
    let maximum_period = maximum_period.min(token_count / minimum_repetitions);
    let mut best: Option<PeriodicSuffix> = None;
    for period in 1..=maximum_period {
        let mut repeated_token_count = 0usize;
        while repeated_token_count + period < token_count
            && token_at(token_count - 1 - repeated_token_count)
                == token_at(token_count - 1 - repeated_token_count - period)
        {
            repeated_token_count += 1;
        }
        let token_count = repeated_token_count + period;
        if token_count < period * minimum_repetitions {
            continue;
        }
        let candidate = PeriodicSuffix {
            period,
            token_count,
            repeated_token_count,
            repetition_count: token_count / period,
        };
        let replace = best.is_none_or(|current| {
            (
                candidate.repeated_token_count,
                candidate.token_count,
                usize::MAX - candidate.period,
            ) > (
                current.repeated_token_count,
                current.token_count,
                usize::MAX - current.period,
            )
        });
        if replace {
            best = Some(candidate);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_longest_bounded_periodic_suffix() {
        assert_eq!(
            longest_periodic_suffix(&[9, 1, 2, 1, 2, 1, 2], 16, 3),
            Some(PeriodicSuffix {
                period: 2,
                token_count: 6,
                repeated_token_count: 4,
                repetition_count: 3,
            })
        );
    }

    #[test]
    fn respects_period_and_repetition_bounds() {
        assert_eq!(longest_periodic_suffix(&[1, 2, 1, 2], 16, 3), None);
        assert_eq!(longest_periodic_suffix(&[1, 2, 1, 2, 1, 2], 1, 3), None);
        assert_eq!(
            longest_periodic_suffix(&[7, 7, 7], 16, 3).unwrap().period,
            1
        );
    }

    #[test]
    fn appended_token_uses_the_same_periodicity_semantics_without_allocating() {
        assert_eq!(
            longest_periodic_suffix_with_appended_token(&[9, 1, 2, 1, 2, 1], 2, 16, 3),
            longest_periodic_suffix(&[9, 1, 2, 1, 2, 1, 2], 16, 3)
        );
        assert_eq!(
            longest_periodic_suffix_with_appended_token(&[1, 2, 1, 2, 1], 3, 16, 3),
            None
        );
    }
}
