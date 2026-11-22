//! Trade vocabulary-head replay for bounded logits without replaying the decoder.
use super::readout::{PrefixStats, SeedCursor};
use super::*;
use objective::{CompetitionReport as Report, PreparedCompetition as Plan};

/// Logical f32 vocabulary values only; excludes head weights, decoder checkpoints,
/// compact trie statistics and any allocator or matrix-kernel scratch.
#[derive(Debug, Clone, Serialize)]
pub struct CausalLmLineTrainingWork {
    pub prefix_rows: usize,
    pub vocabulary: usize,
    pub tile_rows: usize,
    pub forward_head_calls: usize,
    pub backward_head_calls: usize,
    pub replayed_head_rows: usize,
    pub peak_logit_values: usize,
    pub peak_extra_seed_values: usize,
}

impl NativeCausalLm {
    pub(in crate::causal_lm) fn geometry_line_training(
        &self,
        batch: &PreparedCausalWindow,
        tile: usize,
    ) -> CausalLmResult<NativeCausalLmAdaptationGradients> {
        let plan = batch
            .competition
            .as_ref()
            .ok_or("missing complete-line objective")?;
        let set = match plan {
            Plan::LineMenu(menu) => &menu.support,
            Plan::LineSubset(sub) => &sub.support,
            Plan::LineSet(set) => set,
            _ => return Err("expected complete-line objective".into()),
        };
        let n = set.nodes.len();
        let tile = tile.min(n);
        let vocab = self.config.vocabulary;
        let mut work = CausalLmLineTrainingWork {
            prefix_rows: n,
            vocabulary: vocab,
            tile_rows: tile,
            forward_head_calls: 1,
            backward_head_calls: 1,
            replayed_head_rows: 0,
            peak_logit_values: tile * vocab,
            peak_extra_seed_values: tile * vocab,
        };
        if tile == n {
            let mut g = self.geometry_gradients_for_competition(batch)?;
            g.line_work = Some(work);
            return Ok(g);
        }

        let (first, site) = self.first_geometry_site()?;
        let (checkpoints, hidden_rows, selected) = self.training_selected(
            batch,
            self.embed(&batch.input, batch.shape)?,
            0,
            first,
            None,
        )?;
        let normalized = self
            .final_norm
            .forward(&selected, self.config.norm_epsilon)?;
        let pack = self.backward_head.get().and_then(|r| r.as_ref().ok());
        let project = |start: usize| -> CausalLmResult<Tensor> {
            let rows: Vec<_> = (start..(start + tile).min(n)).collect();
            let z = self.project_readout_rows(&normalized.gather_rows(&rows)?, n, pack)?;
            debug_assert_eq!(z.layout(), st_tensor::Layout::RowMajor);
            Ok(z)
        };

        let mut edges = stream::EdgeRows::new(set);
        let mut capture = |start: usize| -> CausalLmResult<Tensor> {
            let z = project(start)?;
            for (j, row) in z.data().chunks_exact(vocab).enumerate() {
                edges.capture(start + j, row)?;
            }
            Ok(z)
        };
        for start in (tile..n).step_by(tile) {
            capture(start)?;
            work.forward_head_calls += 1;
        }
        // Project the leading tile last: after the bottom-up partition is
        // known, it is already the first tile needed by the top-down VJP.
        let mut leading = Some(capture(0)?);
        let stats = edges.reduce()?;
        let mut subset_seed = None;
        let gold_tokens;
        let mut ranking = None;
        let (loss, report) = match plan {
            Plan::LineMenu(menu) => {
                let report = menu.report(&stats)?;
                gold_tokens = menu.gold_tokens();
                ranking = Some(menu.recipe(&gold_tokens));
                (report.loss, Report::LineMenu(report))
            }
            Plan::LineSubset(sub) => {
                let (report, costs) = sub.summarize(&stats)?;
                subset_seed = Some(sub.recipe(&costs, report.rejected_probability_given_menu));
                (report.loss, Report::LineSubset(report))
            }
            Plan::LineSet(_) => {
                let report = set.report(stats.outcomes()[0])?;
                (report.loss, Report::LineSet(report))
            }
            _ => unreachable!("admitted line objective"),
        };
        let mut cursor = SeedCursor::new(n);
        let width = normalized.shape().1;
        let mut pulled = Vec::with_capacity(n * width);
        let mut scratch = vec![0.; vocab];
        work.peak_extra_seed_values = scratch.len();
        work.backward_head_calls = 0;
        for start in (0..n).step_by(tile) {
            let mut z = match leading.take() {
                Some(z) => z,
                None => {
                    let z = project(start)?;
                    work.forward_head_calls += 1;
                    work.replayed_head_rows += z.shape().0;
                    z
                }
            };
            for (offset, values) in z.data_mut().chunks_exact_mut(vocab).enumerate() {
                let i = start + offset;
                scratch.fill(0.);
                match &subset_seed {
                    Some(sub) => sub.write(i, values, &mut scratch),
                    None => cursor.write(
                        set,
                        stats.outcomes(),
                        stats.rows()[i],
                        ranking,
                        i,
                        values,
                        &mut scratch,
                    ),
                }
                values.copy_from_slice(&scratch);
            }
            let dh = self.vocabulary_head_pullback_rows(
                &z,
                self.readout_accumulator.backward_precision(),
                n,
            )?;
            pulled.extend_from_slice(dh.data());
            work.backward_head_calls += 1;
        }
        let parameters = self.decoder_parameter_pullback(
            checkpoints,
            hidden_rows,
            selected,
            batch.shape,
            batch.selected_rows.as_deref(),
            Tensor::from_vec(n, width, pulled)?,
            first,
            learning::DecoderTarget::GeometryFrom(&site),
            None,
        )?;
        let mut g = self.seal_adaptation_gradients(
            batch,
            CausalLmAdaptationMode::Geometry,
            loss,
            Some(report),
            parameters,
            None,
        )?;
        g.line_work = Some(work);
        Ok(g)
    }
}
