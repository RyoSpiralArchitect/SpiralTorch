//! Only an outside partition and the exact logits of outgoing edges survive a projected tile.
use super::*;

pub(super) struct EdgeRows<'a> {
    set: &'a LineSet,
    offsets: Vec<usize>,
    edges: Vec<(usize, f32)>,
    outside: Vec<Option<Group>>,
}

impl<'a> EdgeRows<'a> {
    pub fn new(set: &'a LineSet) -> Self {
        let mut offsets = Vec::with_capacity(set.nodes.len() + 1);
        let mut edges = Vec::with_capacity(set.nodes.len() - 1 + set.counts.len());
        offsets.push(0);
        for n in &set.nodes {
            edges.extend(n.edges.keys().copied().map(|k| (k, 0.)));
            offsets.push(edges.len());
        }
        Self {
            set,
            offsets,
            edges,
            outside: vec![None; set.nodes.len()],
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn capture(&mut self, i: usize, z: &[f32]) -> CausalLmResult<()> {
        if self.outside[i].is_some() {
            return Err("duplicate streamed prefix row".into());
        }
        self.outside[i] = Some(readout::outside_group(&self.set.nodes[i], z)?);
        for (id, v) in &mut self.edges[self.offsets[i]..self.offsets[i + 1]] {
            *v = z[*id];
        }
        Ok(())
    }

    pub fn finish(self, lines: &[Vec<usize>]) -> CausalLmResult<CausalLmLineScores> {
        let set = self.set;
        let edges = |i: usize| &self.edges[self.offsets[i]..self.offsets[i + 1]];
        let mut outcomes = vec![Outcome::STOP; set.nodes.len()];
        let mut rows = vec![readout::Row::default(); set.nodes.len()];
        for (i, n) in set.nodes.iter().enumerate().rev() {
            let outside = self.outside[i].ok_or("missing streamed prefix row")?;
            let (outcome, row) =
                readout::reduce_row(n, outside, &outcomes, edges(i).iter().map(|e| e.1), true);
            outcomes[i] = outcome;
            rows[i] = row;
        }
        Ok(set.scores_from_rows(lines, &outcomes, &rows, |i, k| {
            let e = edges(i);
            e[e.binary_search_by_key(&k, |x| x.0)
                .expect("admitted line edge")]
            .1
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_and_streamed(
        lines: &[Vec<usize>],
        cols: usize,
        values: Vec<f32>,
    ) -> CausalLmLineScores {
        let set = LineSet::new(lines, 0);
        let logits = Tensor::from_vec(set.nodes.len(), cols, values).unwrap();
        let dense = set.scores(lines, &logits).unwrap();
        let mut compact = EdgeRows::new(&set);
        for (i, z) in logits.data().chunks_exact(cols).enumerate().rev() {
            compact.capture(i, z).unwrap();
        }
        assert_eq!(compact.edge_count(), set.nodes.len() - 1 + lines.len());
        let s = compact.finish(lines).unwrap();
        assert_eq!(
            serde_json::to_vec(&dense).unwrap(),
            serde_json::to_vec(&s).unwrap()
        );
        s
    }

    #[test]
    fn streamed_edges_preserve_sub_ulp_winner_underflow_ties_and_full_support() {
        let lines = [vec![1, 2], vec![1, 3]];
        let r = dense_and_streamed(
            &lines,
            4,
            [[0., -1e30, 0., 0.], [0., 0., -1200., 0.], [0.; 4], [0.; 4]].concat(),
        );
        assert_eq!(r.line_nlls[0], r.line_nlls[1]);
        assert_eq!(r.winner_index, 1);
        assert_eq!((-r.conditional_nlls[0]).exp(), 0.);
        assert_eq!(dense_and_streamed(&lines, 4, vec![0.; 16]).winner_index, 0);
        let lines = [vec![1], vec![1, 1], vec![1, 2]];
        dense_and_streamed(
            &lines,
            3,
            vec![0., 1., -1., -5., 4., 3., 7., 0., -2., -4., 1., 2.],
        );
        let r = dense_and_streamed(
            &[vec![1]],
            2,
            vec![f32::MAX, -f32::MAX, -f32::MAX, f32::MAX],
        );
        assert_eq!(r.line_nlls, [4. * f64::from(f32::MAX)]);
    }

    #[test]
    fn streamed_edges_refuse_missing_duplicate_short_or_nonfinite_rows() {
        let lines = [vec![2]];
        let set = LineSet::new(&lines, 0);
        let mut rows = EdgeRows::new(&set);
        assert!(rows.capture(0, &[0., 1.]).is_err());
        assert!(rows.capture(0, &[0., f32::NAN, 2.]).is_err());
        rows.capture(0, &[0., 1., 2.]).unwrap();
        assert!(rows.capture(0, &[0., 0., 0.]).is_err());
        assert!(rows.finish(&lines).is_err());
    }
}
