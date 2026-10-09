use st_kernel_contracts::{euclidean::EuclideanBiasSpec, poincare::PoincareBiasSpec};

#[derive(Clone, Copy, Debug)]
pub(super) enum PairSpec {
    Poincare(PoincareBiasSpec),
    Euclidean(EuclideanBiasSpec),
}
impl From<PoincareBiasSpec> for PairSpec {
    fn from(s: PoincareBiasSpec) -> Self {
        Self::Poincare(s)
    }
}
impl From<EuclideanBiasSpec> for PairSpec {
    fn from(s: EuclideanBiasSpec) -> Self {
        Self::Euclidean(s)
    }
}
impl PairSpec {
    pub fn shape(self) -> [usize; 3] {
        match self {
            Self::Poincare(s) => s.shape(),
            Self::Euclidean(s) => s.shape(),
        }
    }
    pub fn heads(self) -> usize {
        match self {
            Self::Poincare(s) => s.heads(),
            Self::Euclidean(s) => s.heads(),
        }
    }
    pub fn coordinates_len(self) -> usize {
        match self {
            Self::Poincare(s) => s.coordinates_len(),
            Self::Euclidean(s) => s.coordinates_len(),
        }
    }
    pub fn pairs_len(self) -> usize {
        match self {
            Self::Poincare(s) => s.pairs_len(),
            Self::Euclidean(s) => s.pairs_len(),
        }
    }
    pub fn scores_len(self) -> usize {
        match self {
            Self::Poincare(s) => s.scores_len(),
            Self::Euclidean(s) => s.scores_len(),
        }
    }
    pub fn score_shape(self) -> [usize; 4] {
        match self {
            Self::Poincare(s) => s.score_shape(),
            Self::Euclidean(s) => s.score_shape(),
        }
    }
}
