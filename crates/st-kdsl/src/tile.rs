// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Enumeration helpers that generate a consistent autotuning tile search space
//! for subgroup-first kernels.  The template holds candidate values for each
//! dimension and produces the Cartesian product without allocating intermediate
//! vectors so it can be used directly by autotuners.

/// Concrete tiling choice considered by the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileConfig {
    pub workgroup: (u32, u32),
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub vector: u32,
    pub stages: u32,
}

impl TileConfig {
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        (self.workgroup.0, self.workgroup.1, 1)
    }
}

/// Builder that stores candidate dimensions for the search space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileTemplate {
    workgroups: Vec<(u32, u32)>,
    tile_m: Vec<u32>,
    tile_n: Vec<u32>,
    tile_k: Vec<u32>,
    vector: Vec<u32>,
    stages: Vec<u32>,
}

impl Default for TileTemplate {
    fn default() -> Self {
        Self {
            workgroups: vec![(32, 4), (64, 2), (128, 1)],
            tile_m: vec![64, 128],
            tile_n: vec![32, 64],
            tile_k: vec![16, 32],
            vector: vec![1, 2, 4],
            stages: vec![1, 2],
        }
    }
}

impl TileTemplate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_workgroups<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = (u32, u32)>,
    {
        self.workgroups = values.into_iter().collect();
        self
    }

    pub fn with_tile_m<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_m = values.into_iter().collect();
        self
    }

    pub fn with_tile_n<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_n = values.into_iter().collect();
        self
    }

    pub fn with_tile_k<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_k = values.into_iter().collect();
        self
    }

    pub fn with_vector<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.vector = values.into_iter().collect();
        self
    }

    pub fn with_stages<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.stages = values.into_iter().collect();
        self
    }

    pub fn push_workgroup(&mut self, workgroup: (u32, u32)) {
        self.workgroups.push(workgroup);
    }

    pub fn push_tile_m(&mut self, value: u32) {
        self.tile_m.push(value);
    }

    pub fn push_tile_n(&mut self, value: u32) {
        self.tile_n.push(value);
    }

    pub fn push_tile_k(&mut self, value: u32) {
        self.tile_k.push(value);
    }

    pub fn push_vector(&mut self, value: u32) {
        self.vector.push(value);
    }

    pub fn push_stages(&mut self, value: u32) {
        self.stages.push(value);
    }

    pub fn workgroups(&self) -> &[(u32, u32)] {
        &self.workgroups
    }

    pub fn iter(&self) -> TileIter<'_> {
        TileIter::new(self)
    }

    pub fn is_empty(&self) -> bool {
        self.workgroups.is_empty()
            || self.tile_m.is_empty()
            || self.tile_n.is_empty()
            || self.tile_k.is_empty()
            || self.vector.is_empty()
            || self.stages.is_empty()
    }
}

pub struct TileIter<'a> {
    template: &'a TileTemplate,
    indices: [usize; 6],
    finished: bool,
}

impl<'a> TileIter<'a> {
    fn new(template: &'a TileTemplate) -> Self {
        let finished = template.is_empty();
        Self {
            template,
            indices: [0; 6],
            finished,
        }
    }

    fn dims(&self) -> [usize; 6] {
        [
            self.template.workgroups.len(),
            self.template.tile_m.len(),
            self.template.tile_n.len(),
            self.template.tile_k.len(),
            self.template.vector.len(),
            self.template.stages.len(),
        ]
    }

    fn advance(&mut self) {
        for (dim, limit) in (0..self.indices.len()).rev().zip(self.dims().iter().rev()) {
            if self.indices[dim] + 1 < *limit {
                self.indices[dim] += 1;
                for reset_dim in dim + 1..self.indices.len() {
                    self.indices[reset_dim] = 0;
                }
                return;
            }
        }
        self.finished = true;
    }
}

impl<'a> Iterator for TileIter<'a> {
    type Item = TileConfig;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let cfg = TileConfig {
            workgroup: self.template.workgroups[self.indices[0]],
            tile_m: self.template.tile_m[self.indices[1]],
            tile_n: self.template.tile_n[self.indices[2]],
            tile_k: self.template.tile_k[self.indices[3]],
            vector: self.template.vector[self.indices[4]],
            stages: self.template.stages[self.indices[5]],
        };

        self.advance();
        Some(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_cartesian_product() {
        let template = TileTemplate::new()
            .with_workgroups([(64, 2), (128, 1)])
            .with_tile_m([64, 96])
            .with_tile_n([32])
            .with_tile_k([16, 32])
            .with_vector([1, 2])
            .with_stages([1, 2]);
        let configs: Vec<_> = template.iter().collect();
        assert_eq!(configs.len(), 2 * 2 * 1 * 2 * 2 * 2);
        assert_eq!(configs[0].workgroup, (64, 2));
        assert_eq!(configs[0].tile_k, 16);
        assert_eq!(configs[0].stages, 1);
        assert_eq!(configs.last().unwrap().workgroup, (128, 1));
        assert_eq!(configs.last().unwrap().vector, 2);
        assert_eq!(configs.last().unwrap().stages, 2);
    }

    #[test]
    fn handles_empty_dimension() {
        let template = TileTemplate::new().with_tile_k(Vec::<u32>::new());
        assert!(template.is_empty());
        assert_eq!(template.iter().count(), 0);
    }
}
