//! Select an already trained route from matched binary contrasts, never from transfer examples.
use super::sum::Sum;

/// Four rows in (first role, second role) order 00, 01, 10, 11.
/// The two logits in each row correspond to label 0 and label 1, respectively.
pub type BinaryRouteQuartet = [[f32; 2]; 4];
pub const MAX_ROUTE_CALIBRATION_FRAMES: usize = 4096;
pub const MAX_CALIBRATION_ROUTES: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationRole {
    First,
    Second,
}

#[derive(Debug, PartialEq)]
pub struct MatchedRouteSelection<const N: usize> {
    pub frames: usize,
    /// Index into the caller's fixed candidate order; lowest index wins an exact tie.
    pub selected: usize,
    pub worst_gains: [f64; N],
    pub mean_gains: [f64; N],
    /// Unlabeled mean of the common origin's `logit[0] - logit[1]` on all rows.
    pub origin_mean_margin: f64,
}

/// Constant-space calibration. Callers must supply matched endpoints, distinct
/// training frames, and a stable candidate order. A result does not imply
/// positive gain or generalization; even the best candidate can be harmful.
pub struct MatchedRouteCalibration<const N: usize> {
    role: CalibrationRole,
    frames: usize,
    worst: [f64; N],
    total: [Sum; N],
    origin_total: Sum,
}

fn margins(q: &BinaryRouteQuartet) -> Result<[f64; 4], &'static str> {
    if !q.iter().flatten().all(|x| x.is_finite()) {
        return Err("nonfinite calibration logit");
    }
    Ok(q.map(|p| f64::from(p[0]) - f64::from(p[1])))
}

impl<const N: usize> MatchedRouteCalibration<N> {
    pub fn new(role: CalibrationRole) -> Result<Self, &'static str> {
        if !(1..=MAX_CALIBRATION_ROUTES).contains(&N) {
            return Err("invalid number of calibration routes");
        }
        Ok(Self {
            role,
            frames: 0,
            worst: [f64::INFINITY; N],
            total: std::array::from_fn(|_| Sum::default()),
            origin_total: Sum::default(),
        })
    }

    /// Admit a complete matched frame or leave the accumulator unchanged.
    pub fn add_frame(
        &mut self,
        origin: &BinaryRouteQuartet,
        candidates: &[BinaryRouteQuartet; N],
    ) -> Result<(), &'static str> {
        if self.frames == MAX_ROUTE_CALIBRATION_FRAMES {
            return Err("too many calibration frames");
        }
        let origin = margins(origin)?;
        let selectivity = |m: [f64; 4]| (m[1] - m[2]) * 0.5;
        let baseline = selectivity(origin);
        let mut gains = [0.; N];
        for (g, q) in gains.iter_mut().zip(candidates) {
            let gain = selectivity(margins(q)?) - baseline;
            *g = match self.role {
                CalibrationRole::First => gain,
                CalibrationRole::Second => -gain,
            };
        }
        for (i, gain) in gains.into_iter().enumerate() {
            self.worst[i] = self.worst[i].min(gain);
            self.total[i].add(gain);
        }
        for m in origin {
            self.origin_total.add(m);
        }
        self.frames += 1;
        Ok(())
    }

    pub fn selection(&self) -> Result<MatchedRouteSelection<N>, &'static str> {
        if self.frames == 0 {
            return Err("route calibration requires at least one frame");
        }
        let mean = std::array::from_fn(|i| self.total[i].value() / self.frames as f64);
        let mut selected = 0;
        for i in 1..N {
            if self.worst[i] > self.worst[selected]
                || (self.worst[i] == self.worst[selected] && mean[i] > mean[selected])
            {
                selected = i;
            }
        }
        Ok(MatchedRouteSelection {
            frames: self.frames,
            selected,
            worst_gains: self.worst,
            mean_gains: mean,
            origin_mean_margin: self.origin_total.value() / (4 * self.frames) as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(g: f32, common: f32) -> BinaryRouteQuartet {
        [
            [common, 0.],
            [common + g, 0.],
            [common - g, 0.],
            [common, 0.],
        ]
    }

    #[test]
    fn protects_worst_frame_before_average_and_reverses_role() {
        for (role, winner) in [(CalibrationRole::First, 1), (CalibrationRole::Second, 2)] {
            let mut c = MatchedRouteCalibration::new(role).unwrap();
            for values in [[11., 2., -1.], [-1., 3., -2.]] {
                c.add_frame(&frame(1., 8.), &values.map(|x| frame(x, 16.)))
                    .unwrap();
            }
            let r = c.selection().unwrap();
            assert_eq!(r.selected, winner);
            assert_eq!(r.origin_mean_margin, 8.);
            if role == CalibrationRole::First {
                assert_eq!(r.worst_gains, [-2., 1., -3.]);
                assert_eq!(r.mean_gains, [4., 1.5, -2.5]);
            }
        }
    }

    #[test]
    fn equal_minimum_uses_mean_then_original_order() {
        let mut c = MatchedRouteCalibration::new(CalibrationRole::First).unwrap();
        for v in [[1., 1., 1.], [1., 2., 2.]] {
            c.add_frame(&frame(0., 0.), &v.map(|x| frame(x, 0.)))
                .unwrap();
        }
        assert_eq!(c.selection().unwrap().selected, 1);
    }

    #[test]
    fn invalid_frame_is_atomic_even_for_unused_congruent_logit() {
        let mut c = MatchedRouteCalibration::new(CalibrationRole::First).unwrap();
        let q = frame(0., 0.);
        assert!(c.selection().is_err());
        c.add_frame(&q, &[q, q]).unwrap();
        let before = c.selection().unwrap();
        let mut bad = q;
        bad[3][1] = f32::INFINITY;
        assert!(c.add_frame(&q, &[frame(10., 0.), bad]).is_err());
        assert!(c.add_frame(&bad, &[q, q]).is_err());
        assert_eq!(c.selection().unwrap(), before);
    }

    #[test]
    fn caps_routes_and_frames_and_widens_before_subtraction() {
        assert!(MatchedRouteCalibration::<0>::new(CalibrationRole::First).is_err());
        assert!(MatchedRouteCalibration::<33>::new(CalibrationRole::First).is_err());
        let q = [[f32::MAX, -f32::MAX]; 4];
        let mut c = MatchedRouteCalibration::<1>::new(CalibrationRole::Second).unwrap();
        for _ in 0..MAX_ROUTE_CALIBRATION_FRAMES {
            c.add_frame(&q, &[q]).unwrap();
        }
        let r = c.selection().unwrap();
        assert_eq!(r.origin_mean_margin, 2. * f64::from(f32::MAX));
        assert_eq!(r.mean_gains, [0.]);
        assert!(c.add_frame(&q, &[q]).is_err());
        assert_eq!(c.selection().unwrap(), r);
    }
}
