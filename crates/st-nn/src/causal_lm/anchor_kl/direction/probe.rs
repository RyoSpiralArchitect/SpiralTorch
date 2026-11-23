use super::CausalLmResult;

#[derive(Clone, Copy)]
pub(super) enum Choice {
    Peak,
    Next,
    At(usize),
}

// Measure every KL without retaining a whole second bank of vocabulary rows.
// Next also keeps row zero in case the last peak wraps around.
pub(super) struct ProbeRows<T> {
    choice: Choice,
    peak: usize,
    peak_kl: f64,
    first: Option<T>,
    selected: Option<T>,
}

impl<T> ProbeRows<T> {
    pub(super) fn new(choice: Choice) -> Self {
        Self {
            choice,
            peak: 0,
            peak_kl: 0.,
            first: None,
            selected: None,
        }
    }

    pub(super) fn observe(&mut self, i: usize, kl: f64, row: T) {
        if i == 0 || kl > self.peak_kl {
            self.peak = i;
            self.peak_kl = kl;
        }
        match self.choice {
            Choice::Peak if self.peak == i => self.selected = Some(row),
            Choice::At(j) if j == i => self.selected = Some(row),
            Choice::Next if i == 0 => self.first = Some(row),
            Choice::Next if i == self.peak + 1 => self.selected = Some(row),
            _ => {}
        }
    }

    pub(super) fn finish(self, count: usize) -> CausalLmResult<(usize, usize, T)> {
        if count == 0 {
            return Err("empty secant probe bank".into());
        }
        let index = match self.choice {
            Choice::Peak => self.peak,
            Choice::Next => (self.peak + 1) % count,
            Choice::At(i) => i,
        };
        let row = if matches!(self.choice, Choice::Next) && index == 0 {
            self.first
        } else {
            self.selected
        };
        Ok((
            self.peak,
            index,
            row.ok_or("missing selected secant probe")?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_selection_matches_first_max_and_wrap_including_ties() {
        for bits in 0..81 {
            let kl = [1, 3, 9, 27].map(|scale| ((bits / scale) % 3) as f64);
            let peak = (0..4).find(|&i| kl.iter().all(|&k| k <= kl[i])).unwrap();
            for (choice, expected) in [
                (Choice::Peak, peak),
                (Choice::Next, (peak + 1) % 4),
                (Choice::At(0), 0),
                (Choice::At(1), 1),
                (Choice::At(2), 2),
                (Choice::At(3), 3),
            ] {
                let mut rows = ProbeRows::new(choice);
                for (i, &v) in kl.iter().enumerate() {
                    rows.observe(i, v, i + 10);
                    assert!(matches!(choice, Choice::Next) || rows.first.is_none());
                }
                assert_eq!(rows.finish(4).unwrap(), (peak, expected, expected + 10));
            }
        }
    }
}
