//! Two isolated prefixes, two token candidates, one matched objective.
use super::sum::Sum;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CausalLmPairedGapObjective {
    Binary,
    Rank,
    /// Binary CE on the smaller of the two correctly oriented logit margins.
    /// At an exact tie, split the subgradient equally between the two lanes.
    WeakestLane,
    /// A caller-declared center in logit units, not a verified checkpoint identity.
    AnchoredRank {
        reference_center: f64,
        stiffness: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmWeakestLaneReport {
    pub signed_margin: f64,
    /// First and second lane respectively; both are active only at a tie.
    pub active: [bool; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CausalLmPairedGapReport {
    pub schema: &'static str,
    pub objective: CausalLmPairedGapObjective,
    pub tokens: [usize; 2],
    pub logits: [[f32; 2]; 2],
    pub gaps: [f64; 2],
    pub half_difference: f64,
    pub center: f64,
    pub center_drift: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weakest_lane: Option<CausalLmWeakestLaneReport>,
    pub binary_loss: f64,
    pub rank_loss: f64,
    pub center_penalty: f64,
    pub loss: f64,
    /// dL/dD on the two rows; the rival logit receives the opposite seed.
    pub row_derivatives: [f64; 2],
}

impl CausalLmPairedGapObjective {
    fn validate(self) -> CausalLmResult<()> {
        if let Self::AnchoredRank {
            reference_center,
            stiffness,
        } = self
        {
            if !reference_center.is_finite() || !stiffness.is_finite() || stiffness <= 0. {
                return Err(
                    "paired rank requires a finite reference center and positive finite stiffness"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

fn validate(
    vocab: usize,
    tokens: [usize; 2],
    objective: CausalLmPairedGapObjective,
) -> CausalLmResult<()> {
    if tokens[0] == tokens[1] || tokens.iter().any(|&t| t >= vocab) {
        return Err("paired token gap needs two distinct in-vocabulary tokens".into());
    }
    objective.validate()
}

fn sum<const N: usize>(xs: [f64; N]) -> f64 {
    let mut s = Sum::default();
    for x in xs {
        s.add(x);
    }
    s.value()
}

fn softplus_sigmoid(x: f64) -> (f64, f64) {
    let tail = (-x.abs()).exp();
    (
        x.max(0.) + tail.ln_1p(),
        if x >= 0. {
            1. / (1. + tail)
        } else {
            tail / (1. + tail)
        },
    )
}

fn score(
    logits: &Tensor,
    tokens: [usize; 2],
    objective: CausalLmPairedGapObjective,
) -> CausalLmResult<CausalLmPairedGapReport> {
    let (rows, cols) = logits.shape();
    validate(cols, tokens, objective)?;
    if rows != 2 || !logits.data().iter().all(|x| x.is_finite()) {
        return Err("paired token gap requires exactly two finite logit rows".into());
    }
    let converted;
    let logits = if logits.is_contiguous() {
        logits
    } else {
        converted = logits.to_layout(st_tensor::Layout::RowMajor)?;
        &converted
    };
    let (rs, cs) = logits.strides()?;
    let z = [0, 1].map(|r| tokens.map(|c| logits.data()[r * rs + c * cs]));
    let [[a, b], [c, d]] = z.map(|p| p.map(f64::from));
    let gaps = [a - b, c - d];
    // Do not subtract separately rounded gaps; their common term can dwarf the rank signal.
    let half_difference = sum([a, -b, -c, d]) * 0.5;
    let center = sum([a, -b, c, -d]) * 0.5;
    let (l0, p0) = softplus_sigmoid(-gaps[0]);
    let (l1, p1) = softplus_sigmoid(gaps[1]);
    let binary_loss = (l0 + l1) * 0.5;
    let (rank_loss, w) = softplus_sigmoid(-half_difference);
    let mut weakest_lane = None;
    let (center_drift, center_penalty, loss, row_derivatives) = match objective {
        CausalLmPairedGapObjective::Binary => (None, 0., binary_loss, [-p0 * 0.5, p1 * 0.5]),
        CausalLmPairedGapObjective::Rank => (None, 0., rank_loss, [-w * 0.5, w * 0.5]),
        CausalLmPairedGapObjective::WeakestLane => {
            // 2*C = Dp - (-Dn). Comparing rounded losses or gaps can hide the weaker lane.
            let (signed_margin, active, loss, seeds) = if center < 0. {
                (gaps[0], [true, false], l0, [-p0, 0.])
            } else if center > 0. {
                (-gaps[1], [false, true], l1, [0., p1])
            } else {
                (gaps[0], [true, true], binary_loss, [-p0 * 0.5, p1 * 0.5])
            };
            weakest_lane = Some(CausalLmWeakestLaneReport {
                signed_margin,
                active,
            });
            (None, 0., loss, seeds)
        }
        CausalLmPairedGapObjective::AnchoredRank {
            reference_center,
            stiffness,
        } => {
            let drift = sum([a * 0.5, -b * 0.5, c * 0.5, -d * 0.5, -reference_center]);
            let force = stiffness * drift;
            let penalty = (force * 0.5) * drift;
            (
                Some(drift),
                penalty,
                rank_loss + penalty,
                [(force - w) * 0.5, (force + w) * 0.5],
            )
        }
    };
    if !loss.is_finite() || !row_derivatives.iter().all(|v| v.is_finite()) {
        return Err("nonfinite paired token objective or seed".into());
    }
    Ok(CausalLmPairedGapReport {
        schema: "spiraltorch.paired_token_gap.v1",
        objective,
        tokens,
        logits: z,
        gaps,
        half_difference,
        center,
        center_drift,
        weakest_lane,
        binary_loss,
        rank_loss,
        center_penalty,
        loss,
        row_derivatives,
    })
}

pub(super) fn gradient(
    logits: &Tensor,
    tokens: [usize; 2],
    objective: CausalLmPairedGapObjective,
) -> CausalLmResult<(CausalLmPairedGapReport, Tensor)> {
    let report = score(logits, tokens, objective)?;
    let seed = report.row_derivatives.map(|x| x as f32);
    if !(report.loss as f32).is_finite() || !seed.iter().all(|v| v.is_finite()) {
        return Err("paired token loss or seed exceeds the finite f32 training domain".into());
    }
    let cols = logits.shape().1;
    let mut values = vec![0.; logits.len()];
    for (row, w) in values.chunks_exact_mut(cols).zip(seed) {
        row[tokens[0]] = w;
        row[tokens[1]] = -w;
    }
    Ok((report, Tensor::from_vec(2, cols, values)?))
}

impl NativeCausalLm {
    pub fn paired_token_gap_shape(
        &self,
        first_len: usize,
        second_len: usize,
        tokens: [usize; 2],
        objective: CausalLmPairedGapObjective,
        training: bool,
    ) -> CausalLmResult<[usize; 2]> {
        validate(self.config.vocabulary, tokens, objective)?;
        self.context_token_preference_shape(first_len, second_len, tokens[0], 0., training)
    }

    fn paired_gap_window(
        &self,
        first: &[usize],
        second: &[usize],
        tokens: [usize; 2],
        objective: CausalLmPairedGapObjective,
        training: bool,
    ) -> CausalLmResult<PreparedCausalWindow> {
        let shape =
            self.paired_token_gap_shape(first.len(), second.len(), tokens, objective, training)?;
        let mut p = self.prepare_context_pair_window([first, second], tokens, shape)?;
        p.competition = Some(objective::PreparedCompetition::PairedGap { tokens, objective });
        Ok(p)
    }

    /// First prefix favors tokens[0]; the second favors tokens[1]. No optimizer update.
    pub fn paired_token_gap(
        &self,
        first: &[usize],
        second: &[usize],
        tokens: [usize; 2],
        objective: CausalLmPairedGapObjective,
    ) -> CausalLmResult<CausalLmPairedGapReport> {
        let p = self.paired_gap_window(first, second, tokens, objective, false)?;
        let hidden = self.decode_hidden(&p.input, p.shape)?;
        score(&self.causal_logits(hidden, &p)?, tokens, objective)
    }

    /// One packed decoder VJP and a geometry-only handle sealed to this parameter snapshot.
    pub fn geometry_paired_token_gap_gradients(
        &self,
        first: &[usize],
        second: &[usize],
        tokens: [usize; 2],
        objective: CausalLmPairedGapObjective,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let p = self.paired_gap_window(first, second, tokens, objective, true)?;
        self.geometry_gradients_for_competition(&p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use CausalLmPairedGapObjective::*;

    #[test]
    fn all_objectives_match_independent_differences_and_storage_layouts() {
        let xs = [0.2f32, -0.4, 1.5, 0.6, -0.7, 0.3, 0.9, -0.1];
        for obj in [
            Binary,
            Rank,
            WeakestLane,
            AnchoredRank {
                reference_center: -0.25,
                stiffness: 1.75,
            },
        ] {
            let f = |i: usize, h: f64| {
                let mut v = xs.map(f64::from);
                v[i] += h;
                let (a, b) = (v[2] - v[0], v[6] - v[4]);
                let sp = |x: f64| x.exp().ln_1p();
                match obj {
                    Binary => (sp(-a) + sp(b)) / 2.,
                    Rank => sp((b - a) / 2.),
                    WeakestLane => sp((-a).max(b)),
                    AnchoredRank {
                        reference_center,
                        stiffness,
                    } => {
                        sp((b - a) / 2.)
                            + stiffness * ((a + b) / 2. - reference_center).powi(2) / 2.
                    }
                }
            };
            let x = Tensor::from_vec(2, 4, xs.to_vec()).unwrap();
            let (r, g) = gradient(&x, [2, 0], obj).unwrap();
            assert!((r.loss - f(0, 0.)).abs() < 1e-12);
            for (i, v) in g.data().iter().enumerate() {
                assert!((f64::from(*v) - (f(i, 1e-5) - f(i, -1e-5)) / 2e-5).abs() < 1e-7);
            }
            for layout in [
                st_tensor::Layout::ColMajor,
                st_tensor::Layout::Chimera {
                    stripes: 2,
                    tile: 2,
                },
            ] {
                let (rc, gc) = gradient(&x.to_layout(layout).unwrap(), [2, 0], obj).unwrap();
                assert_eq!(r, rc);
                assert_eq!(g.data(), gc.data());
            }
        }
    }

    #[test]
    fn weakest_lane_tracks_the_signed_winner_and_is_symmetric_at_ties() {
        for (a, n, active) in [
            (-2., 1., [true, false]),
            (1., -2., [false, true]),
            (0.75, 0.75, [true, true]),
        ] {
            let z = Tensor::from_vec(2, 2, vec![a, 0., 0., n]).unwrap();
            let r = score(&z, [0, 1], WeakestLane).unwrap();
            let q = r.weakest_lane.unwrap();
            assert_eq!(q.active, active);
            assert_eq!(q.signed_margin, f64::from(a.min(n)));
            let s = 1. / (1. + q.signed_margin.exp());
            let mass = if active[0] == active[1] { 0.5 } else { 1. };
            let weight = |on: bool| if on { s * mass } else { 0. };
            for (given, expected) in r
                .row_derivatives
                .into_iter()
                .zip([-weight(active[0]), weight(active[1])])
            {
                assert!((given - expected).abs() < 1e-15);
            }
            let mirror = Tensor::from_vec(2, 2, vec![0., n, a, 0.]).unwrap();
            let rev = score(&mirror, [1, 0], WeakestLane).unwrap();
            assert_eq!(rev.loss, r.loss);
            assert_eq!(
                rev.row_derivatives,
                [-r.row_derivatives[1], -r.row_derivatives[0]]
            );
            let legacy = serde_json::to_value(score(&z, [0, 1], Binary).unwrap()).unwrap();
            assert!(legacy.get("weakest_lane").is_none());
        }
    }

    #[test]
    fn weakest_lane_is_not_selected_from_rounded_losses_or_center_minus_radius() {
        let z = Tensor::from_vec(2, 2, vec![-f32::MAX, 1., f32::MAX, 0.]).unwrap();
        let r = score(&z, [0, 1], WeakestLane).unwrap();
        assert_eq!(r.gaps[0], -r.gaps[1]);
        assert_eq!(r.weakest_lane.unwrap().active, [true, false]);
        assert_eq!(r.row_derivatives, [-1., 0.]);
        let z = Tensor::from_vec(2, 2, vec![f32::MAX, 0., 0., 0.125]).unwrap();
        let r = score(&z, [0, 1], WeakestLane).unwrap();
        assert_eq!(r.half_difference - r.center.abs(), 0.);
        assert_eq!(r.weakest_lane.unwrap().signed_margin, 0.125);
    }

    #[test]
    fn common_terms_cancel_before_rounding_and_keep_small_seeds() {
        let x = Tensor::from_vec(2, 2, vec![f32::MAX, 1., f32::MAX, 3.]).unwrap();
        let r = score(
            &x,
            [0, 1],
            AnchoredRank {
                reference_center: f64::from(f32::MAX),
                stiffness: 1.,
            },
        )
        .unwrap();
        assert_eq!(r.gaps[0], r.gaps[1]);
        assert_eq!(r.half_difference, 1.);
        assert_eq!(r.center_drift, Some(-2.));
        let x = Tensor::from_vec(2, 2, vec![40., 0., -40., 0.]).unwrap();
        let (_, g) = gradient(&x, [0, 1], Rank).unwrap();
        assert!(g.data()[0] < 0. && g.data()[0] == -g.data()[2]);
    }

    #[test]
    fn refuses_bad_domains_and_unrepresentable_training() {
        let x = Tensor::zeros(2, 3).unwrap();
        for ids in [[0, 0], [0, 3]] {
            assert!(score(&x, ids, Rank).is_err());
        }
        for (c, w) in [(f64::NAN, 1.), (0., 0.), (1., f64::INFINITY)] {
            assert!(score(
                &x,
                [0, 1],
                AnchoredRank {
                    reference_center: c,
                    stiffness: w
                }
            )
            .is_err());
        }
        let bad = Tensor::from_vec(2, 3, vec![0., 0., 0., 0., 0., f32::NAN]).unwrap();
        assert!(score(&bad, [0, 1], Rank).is_err());
        assert!(score(&Tensor::zeros(3, 2).unwrap(), [0, 1], Binary).is_err());
        let big = AnchoredRank {
            reference_center: 1e21,
            stiffness: 1.,
        };
        assert!(score(&x, [0, 1], big).unwrap().loss.is_finite());
        assert!(gradient(&x, [0, 1], big).is_err());
    }
}
