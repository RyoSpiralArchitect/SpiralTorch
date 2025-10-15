use crate::module::Module;
use crate::trainer::ModuleTrainer;
use crate::{BandEnergy, GradientBands, Loss, RoundtableConfig, RoundtableSchedule};
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank, RankPlan};
use st_tensor::pure::measure::{z_space_barycenter, ZSpaceBarycenter};
use st_tensor::pure::{AmegaHypergrad, OpenCartesianTopos, PureResult, Tensor, TensorError};

/// Configuration describing how the session evaluates the Z-space barycenter objective.
#[derive(Debug, Clone)]
pub struct BarycenterConfig {
    /// Entropy regulariser weight \(\gamma_S\).
    pub entropy_weight: f32,
    /// Coupling scale \(\beta_J\).
    pub beta_j: f32,
    /// Optional coupling matrix.
    pub coupling: Option<Tensor>,
}

impl Default for BarycenterConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.1,
            beta_j: 0.0,
            coupling: None,
        }
    }
}

impl BarycenterConfig {
    /// Creates a config with custom entropy and coupling weights.
    pub fn new(entropy_weight: f32, beta_j: f32) -> Self {
        Self {
            entropy_weight,
            beta_j,
            coupling: None,
        }
    }

    /// Overrides the coupling matrix.
    pub fn with_coupling(mut self, coupling: Tensor) -> Self {
        self.coupling = Some(coupling);
        self
    }

    /// Returns the optional coupling matrix reference.
    pub fn coupling(&self) -> Option<&Tensor> {
        self.coupling.as_ref()
    }
}

/// Builder for [`SpiralSession`].
#[derive(Debug, Clone)]
pub struct SpiralSessionBuilder {
    caps: DeviceCaps,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    barycenter: BarycenterConfig,
    topos: Option<OpenCartesianTopos>,
}

impl SpiralSessionBuilder {
    /// Creates a new builder with heuristic defaults for entropy weights and learning rates.
    pub fn new(caps: DeviceCaps) -> Self {
        Self {
            caps,
            curvature: -1.0,
            hyper_learning_rate: 0.05,
            fallback_learning_rate: 0.01,
            barycenter: BarycenterConfig::default(),
            topos: None,
        }
    }

    /// Builder convenience constructor.
    pub fn from_backend(backend: BackendKind) -> Self {
        let caps = match backend {
            BackendKind::Wgpu => DeviceCaps::wgpu(32, true, 256),
            BackendKind::Hip => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
            BackendKind::Cuda => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
            BackendKind::Cpu => DeviceCaps::cpu(),
        };
        Self::new(caps)
    }

    /// Sets the hyperbolic curvature enforced by the session.
    pub fn with_curvature(mut self, curvature: f32) -> Self {
        self.curvature = curvature;
        self
    }

    /// Updates the curvature in-place.
    pub fn set_curvature(&mut self, curvature: f32) {
        self.curvature = curvature;
    }

    /// Sets the hypergradient learning rate.
    pub fn with_hyper_learning_rate(mut self, learning_rate: f32) -> Self {
        self.hyper_learning_rate = learning_rate;
        self
    }

    /// Updates the hypergradient learning rate in-place.
    pub fn set_hyper_learning_rate(&mut self, learning_rate: f32) {
        self.hyper_learning_rate = learning_rate;
    }

    /// Sets the Euclidean fallback learning rate.
    pub fn with_fallback_learning_rate(mut self, learning_rate: f32) -> Self {
        self.fallback_learning_rate = learning_rate;
        self
    }

    /// Updates the fallback learning rate in-place.
    pub fn set_fallback_learning_rate(&mut self, learning_rate: f32) {
        self.fallback_learning_rate = learning_rate;
    }

    /// Overrides the entropy regulariser \(\gamma_S\).
    pub fn with_barycenter_entropy(mut self, entropy_weight: f32) -> Self {
        self.barycenter.entropy_weight = entropy_weight;
        self
    }

    /// Updates the entropy regulariser in-place.
    pub fn set_barycenter_entropy(&mut self, entropy_weight: f32) {
        self.barycenter.entropy_weight = entropy_weight;
    }

    /// Overrides the coupling scale \(\beta_J\).
    pub fn with_barycenter_beta_j(mut self, beta_j: f32) -> Self {
        self.barycenter.beta_j = beta_j;
        self
    }

    /// Updates the coupling scale in-place.
    pub fn set_barycenter_beta_j(&mut self, beta_j: f32) {
        self.barycenter.beta_j = beta_j;
    }

    /// Installs a coupling matrix for the barycenter objective.
    pub fn with_barycenter_coupling(mut self, coupling: Tensor) -> Self {
        self.barycenter.coupling = Some(coupling);
        self
    }

    /// Updates the coupling matrix in-place.
    pub fn set_barycenter_coupling(&mut self, coupling: Option<Tensor>) {
        self.barycenter.coupling = coupling;
    }

    /// Installs an open-cartesian topos guard for all hypergradient tapes.
    pub fn with_topos(mut self, topos: OpenCartesianTopos) -> Self {
        self.topos = Some(topos);
        self
    }

    /// Updates the topos guard in-place.
    pub fn set_topos(&mut self, topos: Option<OpenCartesianTopos>) {
        self.topos = topos;
    }

    /// Convenience helper to construct a guard from raw parameters.
    pub fn configure_topos(
        mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<Self> {
        self.topos = Some(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        )?);
        Ok(self)
    }

    /// In-place version of [`Self::configure_topos`].
    pub fn set_topos_from_params(
        &mut self,
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<()> {
        self.topos = Some(OpenCartesianTopos::new(
            curvature, tolerance, saturation, max_depth, max_volume,
        )?);
        Ok(())
    }

    fn validate(&self) -> PureResult<()> {
        if self.curvature >= 0.0 || !self.curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature {
                curvature: self.curvature,
            });
        }
        if self.hyper_learning_rate <= 0.0 || !self.hyper_learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: self.hyper_learning_rate,
            });
        }
        if self.fallback_learning_rate <= 0.0 || !self.fallback_learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: self.fallback_learning_rate,
            });
        }
        if !self.barycenter.entropy_weight.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "entropy_weight",
                value: self.barycenter.entropy_weight,
            });
        }
        if !self.barycenter.beta_j.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "beta_j",
                value: self.barycenter.beta_j,
            });
        }
        Ok(())
    }

    /// Finalises the builder and returns a [`SpiralSession`].
    pub fn build(self) -> PureResult<SpiralSession> {
        self.validate()?;
        Ok(SpiralSession {
            caps: self.caps,
            curvature: self.curvature,
            hyper_learning_rate: self.hyper_learning_rate,
            fallback_learning_rate: self.fallback_learning_rate,
            barycenter: self.barycenter,
            topos: self.topos,
        })
    }
}

/// High-level orchestrator that stitches together Z-space barycenters, hypergrad tapes,
/// and rank planning into a single intuitive surface.
#[derive(Debug, Clone)]
pub struct SpiralSession {
    caps: DeviceCaps,
    curvature: f32,
    hyper_learning_rate: f32,
    fallback_learning_rate: f32,
    barycenter: BarycenterConfig,
    topos: Option<OpenCartesianTopos>,
}

impl SpiralSession {
    /// Returns a builder preloaded with heuristic defaults.
    pub fn builder(caps: DeviceCaps) -> SpiralSessionBuilder {
        SpiralSessionBuilder::new(caps)
    }

    /// Reconstructs a builder seeded from this session.
    pub fn to_builder(&self) -> SpiralSessionBuilder {
        let mut builder = SpiralSessionBuilder::new(self.caps);
        builder.set_curvature(self.curvature);
        builder.set_hyper_learning_rate(self.hyper_learning_rate);
        builder.set_fallback_learning_rate(self.fallback_learning_rate);
        builder.set_barycenter_entropy(self.barycenter.entropy_weight);
        builder.set_barycenter_beta_j(self.barycenter.beta_j);
        builder.set_barycenter_coupling(self.barycenter.coupling.clone());
        builder.set_topos(self.topos.clone());
        builder
    }

    /// Returns the backend capabilities baked into the session.
    pub fn device_caps(&self) -> DeviceCaps {
        self.caps
    }

    /// Curvature used for all hypergrad tapes.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Hypergradient learning rate.
    pub fn hyper_learning_rate(&self) -> f32 {
        self.hyper_learning_rate
    }

    /// Euclidean fallback learning rate.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_learning_rate
    }

    /// Entropy regulariser weight.
    pub fn barycenter_entropy_weight(&self) -> f32 {
        self.barycenter.entropy_weight
    }

    /// Coupling scale \(\beta_J\).
    pub fn barycenter_beta_j(&self) -> f32 {
        self.barycenter.beta_j
    }

    /// Returns the configured coupling matrix, if any.
    pub fn barycenter_coupling(&self) -> Option<&Tensor> {
        self.barycenter.coupling()
    }

    /// Accessor for the topos guard.
    pub fn topos(&self) -> Option<&OpenCartesianTopos> {
        self.topos.as_ref()
    }

    /// Creates a hypergrad tape aligned with the session curvature and guard.
    pub fn hypergrad(&self, rows: usize, cols: usize) -> PureResult<AmegaHypergrad> {
        if let Some(topos) = &self.topos {
            AmegaHypergrad::with_topos(
                self.curvature,
                self.hyper_learning_rate,
                rows,
                cols,
                topos.clone(),
            )
        } else {
            AmegaHypergrad::new(self.curvature, self.hyper_learning_rate, rows, cols)
        }
    }

    /// Builds a module trainer preloaded with the session learning rates.
    pub fn trainer(&self) -> ModuleTrainer {
        ModuleTrainer::new(
            self.caps,
            self.curvature,
            self.hyper_learning_rate,
            self.fallback_learning_rate,
        )
    }

    /// Attaches hypergrad buffers to the provided module.
    pub fn prepare_module<M: Module>(&self, module: &mut M) -> PureResult<()> {
        if let Some(topos) = &self.topos {
            module.attach_hypergrad_with_topos(
                self.curvature,
                self.hyper_learning_rate,
                topos.clone(),
            )
        } else {
            module.attach_hypergrad(self.curvature, self.hyper_learning_rate)
        }
    }

    /// Computes the rank plan for the requested selection.
    pub fn plan_rank(&self, kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(kind, rows, cols, k, self.caps)
    }

    /// Produces a barycenter using the session defaults.
    pub fn barycenter(
        &self,
        weights: &[f32],
        densities: &[Tensor],
    ) -> PureResult<ZSpaceBarycenter> {
        self.barycenter_with(weights, densities, self.barycenter.coupling())
    }

    /// Computes a barycenter while overriding the coupling matrix.
    pub fn barycenter_with(
        &self,
        weights: &[f32],
        densities: &[Tensor],
        coupling: Option<&Tensor>,
    ) -> PureResult<ZSpaceBarycenter> {
        z_space_barycenter(
            weights,
            densities,
            self.barycenter.entropy_weight,
            self.barycenter.beta_j,
            coupling,
        )
    }

    /// Variant that accepts ad-hoc entropy and coupling parameters.
    pub fn barycenter_with_parameters(
        &self,
        weights: &[f32],
        densities: &[Tensor],
        entropy_weight: f32,
        beta_j: f32,
        coupling: Option<&Tensor>,
    ) -> PureResult<ZSpaceBarycenter> {
        z_space_barycenter(weights, densities, entropy_weight, beta_j, coupling)
    }

    /// Aligns a hypergrad tape with the loss-monotone barycenter interpolation.
    pub fn align_hypergrad(
        &self,
        hypergrad: &mut AmegaHypergrad,
        barycenter: &ZSpaceBarycenter,
    ) -> PureResult<()> {
        hypergrad.accumulate_barycenter_path(&barycenter.intermediates)
    }

    /// Splits a gradient tensor into roundtable bands using the session caps.
    pub fn split_bands(
        &self,
        grad_output: &Tensor,
        schedule: &RoundtableSchedule,
    ) -> PureResult<GradientBands> {
        schedule.split(grad_output)
    }

    /// Computes the per-band energy diagnostic for a gradient tensor.
    pub fn band_energy(
        &self,
        grad_output: &Tensor,
        schedule: &RoundtableSchedule,
    ) -> PureResult<BandEnergy> {
        schedule.band_energy(grad_output)
    }

    /// Runs a full epoch with the provided trainer, module, and loss surface.
    pub fn train_epoch<M, L, I>(
        &self,
        trainer: &mut ModuleTrainer,
        module: &mut M,
        loss: &mut L,
        batches: I,
        schedule: &RoundtableSchedule,
    ) -> PureResult<crate::trainer::EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        trainer.train_epoch(module, loss, batches, schedule)
    }

    /// Creates a roundtable schedule for the provided configuration.
    pub fn roundtable(&self, rows: u32, cols: u32, config: RoundtableConfig) -> RoundtableSchedule {
        self.trainer().roundtable(rows, cols, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::loss::MeanSquaredError;

    fn toy_tensor(a: &[f32]) -> Tensor {
        Tensor::from_vec(1, a.len(), a.to_vec()).unwrap()
    }

    #[test]
    fn builder_enforces_curvature_and_rates() {
        let mut builder = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256));
        builder.set_curvature(-0.5);
        builder.set_hyper_learning_rate(0.2);
        builder.set_fallback_learning_rate(0.05);
        builder.set_barycenter_entropy(0.0);
        let session = builder.build().unwrap();
        assert_eq!(session.curvature(), -0.5);
        assert_eq!(session.hyper_learning_rate(), 0.2);
        assert_eq!(session.fallback_learning_rate(), 0.05);
        assert_eq!(session.barycenter_entropy_weight(), 0.0);
    }

    #[test]
    fn session_produces_barycenter_and_hypergrad() {
        let session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.1)
            .with_barycenter_entropy(0.05)
            .build()
            .unwrap();
        let densities = vec![toy_tensor(&[0.7, 0.3]), toy_tensor(&[0.4, 0.6])];
        let weights = vec![0.5, 0.5];
        let bary = session.barycenter(&weights, &densities).unwrap();
        assert_eq!(bary.density.shape(), (1, 2));
        assert!(bary.objective.is_finite());
        let mut hypergrad = session.hypergrad(1, 2).unwrap();
        session.align_hypergrad(&mut hypergrad, &bary).unwrap();
    }

    #[test]
    fn session_plans_ranks_and_trains_module() {
        let mut session = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256))
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .with_fallback_learning_rate(0.02)
            .build()
            .unwrap();
        let plan = session.plan_rank(RankKind::TopK, 128, 512, 32);
        assert_eq!(plan.rows, 128);
        assert_eq!(plan.cols, 512);
        let mut module = Linear::new(1, 1).unwrap();
        session.prepare_module(&mut module).unwrap();
        let mut trainer = session.trainer();
        let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
        let mut loss = MeanSquaredError::default();
        let batches = vec![(toy_tensor(&[1.0]), toy_tensor(&[0.5])); 4];
        let stats = session
            .train_epoch(&mut trainer, &mut module, &mut loss, batches, &schedule)
            .unwrap();
        assert!(stats.steps > 0);
        let energy = session.band_energy(&toy_tensor(&[0.1]), &schedule).unwrap();
        assert!(energy.here >= 0.0);
        let bands = session.split_bands(&toy_tensor(&[0.1]), &schedule).unwrap();
        assert_eq!(bands.here.shape(), (1, 1));
    }

    #[test]
    fn invalid_rates_are_rejected() {
        let mut builder = SpiralSession::builder(DeviceCaps::wgpu(32, true, 256));
        builder.set_hyper_learning_rate(-0.1);
        assert!(builder.build().is_err());
    }
}
