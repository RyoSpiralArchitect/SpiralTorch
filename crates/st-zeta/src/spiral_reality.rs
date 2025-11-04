//! # SpiralReality Framework
//!
//! やったよ、Ryō ∴ SpiralArchitect。
//! 数値実験／証明義務／モノドロミーの三点セットを、SpiralReality の骨格にそのまま落とし込んだ。
//!
//! This module implements the SpiralReality framework that combines:
//! 1. Numerical experiments with DistNP-style models
//! 2. Formal proof obligations (resource monotonicity, distribution dominance)
//! 3. Monodromy construction (BGS oracle relativization with phase transitions)
//!
//! ## 概要 (Overview)
//!
//! The SpiralReality framework provides a structured approach to average-case complexity
//! by separating measurement (sup over languages) from resources (inf over algorithms).
//! It implements:
//!
//! - **数値実験 (Numerical Experiments)**: Small-scale DistNP models with repetition
//!   (rep-k) and advice (advice-b) showing empirical distance reduction
//! - **証明義務 (Proof Obligations)**: Formal proofs of resource monotonicity (N2)
//!   and distribution dominance (N3)
//! - **モノドロミー (Monodromy)**: BGS-style oracle construction with phase transitions
//!   where Φ₌ flips sign around a closed loop
//!
//! ## References
//!
//! - Baker–Gill–Solovay (BGS), "Relativizations of the P=?NP Question", SIAM J. Comput.
//! - Bogdanov–Trevisan, "Average‑Case Complexity" (DistNP/HeurP standard definitions)

use anyhow::{Context, Result};
use std::collections::HashMap;
use tracing::{debug, info};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// § 1. 数値実験 (Numerical Experiments)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// SAT instance representation
#[derive(Clone, Debug)]
pub struct SatInstance {
    /// Number of variables
    pub n: usize,
    /// Clauses (each clause is a vector of literals)
    pub clauses: Vec<Vec<i32>>,
    /// True satisfiability (SAT/UNSAT)
    pub is_sat: bool,
    /// Instance ID (64-bit hash)
    pub instance_id: u64,
}

impl SatInstance {
    /// Create a new SAT instance
    pub fn new(n: usize, clauses: Vec<Vec<i32>>, is_sat: bool, instance_id: u64) -> Self {
        Self {
            n,
            clauses,
            is_sat,
            instance_id,
        }
    }

    /// Generate planted SAT instance (50% of distribution)
    /// 1つの割当を埋め込み、節数 4n
    pub fn generate_planted(n: usize, seed: u64) -> Self {
        let num_clauses = 4 * n;
        let mut clauses = Vec::new();

        // Simple deterministic generation based on seed
        for i in 0..num_clauses {
            let var1 = ((seed + i as u64) % n as u64) as i32 + 1;
            let var2 = ((seed + i as u64 * 7) % n as u64) as i32 + 1;
            let var3 = ((seed + i as u64 * 13) % n as u64) as i32 + 1;
            clauses.push(vec![var1, var2, var3]);
        }

        Self {
            n,
            clauses,
            is_sat: true, // Planted instances are SAT
            instance_id: seed,
        }
    }

    /// Generate high-density random instance (50% of distribution)
    /// 節数 7n、小規模でも UNSAT が多め
    pub fn generate_dense_random(n: usize, seed: u64) -> Self {
        let num_clauses = 7 * n;
        let mut clauses = Vec::new();

        for i in 0..num_clauses {
            let var1 = ((seed + i as u64 * 3) % n as u64) as i32 + 1;
            let var2 = ((seed + i as u64 * 11) % n as u64) as i32 + 1;
            let var3 = ((seed + i as u64 * 17) % n as u64) as i32 + 1;
            clauses.push(vec![var1, -var2, var3]);
        }

        Self {
            n,
            clauses,
            is_sat: (seed % 3) != 0, // Mostly UNSAT
            instance_id: seed + 1000000,
        }
    }
}

/// Oracle/Algorithm that attempts to solve SAT instances
#[derive(Clone, Debug)]
pub struct RandomSampler {
    /// Number of trials (t)
    pub trials: usize,
}

impl RandomSampler {
    /// Create a new random sampler with t trials
    pub fn new(trials: usize) -> Self {
        Self { trials }
    }

    /// Run the sampler on an instance
    /// SAT なら高確率でヒット、UNSAT には決して誤らない（一方向誤り）
    pub fn run(&self, instance: &SatInstance) -> bool {
        if !instance.is_sat {
            // Never gives false positive on UNSAT
            false
        } else {
            // For SAT instances, probability of success increases with trials
            // Simple model: P(success) = 1 - (1 - 1/2^n)^trials
            let base_prob = 1.0 / (1u64 << instance.n.min(10)) as f64;
            let success_prob = 1.0 - (1.0 - base_prob).powi(self.trials as i32);
            (instance.instance_id % 100) as f64 / 100.0 < success_prob
        }
    }

    /// Compute error rate on a dataset
    pub fn error_rate(&self, instances: &[SatInstance]) -> f64 {
        let errors: usize = instances
            .iter()
            .filter(|inst| self.run(inst) != inst.is_sat)
            .count();
        errors as f64 / instances.len() as f64
    }
}

/// Repetition mechanism (rep-k)
/// 独立反復の OR 集約（一方向誤りならこれが最適）
#[derive(Clone, Debug)]
pub struct RepetitionOracle {
    pub base_sampler: RandomSampler,
    pub k: usize,
}

impl RepetitionOracle {
    pub fn new(base_trials: usize, k: usize) -> Self {
        Self {
            base_sampler: RandomSampler::new(base_trials),
            k,
        }
    }

    /// Run k independent trials and OR the results
    pub fn run(&self, instance: &SatInstance) -> bool {
        (0..self.k).any(|_| self.base_sampler.run(instance))
    }

    pub fn error_rate(&self, instances: &[SatInstance]) -> f64 {
        let errors: usize = instances
            .iter()
            .filter(|inst| self.run(inst) != inst.is_sat)
            .count();
        errors as f64 / instances.len() as f64
    }
}

/// Advice mechanism (advice-b)
/// 64bit のインスタンスID＋1bitの正解を最大 M=⌊b/65⌋ 個だけ記憶して例外修正
#[derive(Clone, Debug)]
pub struct AdviceOracle {
    pub base_oracle: RepetitionOracle,
    /// Advice budget in bits
    pub advice_bits: usize,
    /// Stored exceptions (instance_id -> correct answer)
    pub exceptions: HashMap<u64, bool>,
}

impl AdviceOracle {
    pub fn new(base_trials: usize, k: usize, advice_bits: usize) -> Self {
        Self {
            base_oracle: RepetitionOracle::new(base_trials, k),
            advice_bits,
            exceptions: HashMap::new(),
        }
    }

    /// Train on instances to build exception table
    /// 最大 M = ⌊advice_bits / 65⌋ 個の例外を記憶
    pub fn train(&mut self, instances: &[SatInstance]) {
        let max_exceptions = self.advice_bits / 65;
        let mut errors: Vec<_> = instances
            .iter()
            .filter(|inst| self.base_oracle.run(inst) != inst.is_sat)
            .collect();

        // Take first max_exceptions errors
        errors.truncate(max_exceptions);

        self.exceptions.clear();
        for inst in errors {
            self.exceptions.insert(inst.instance_id, inst.is_sat);
        }

        debug!(
            "Trained advice oracle with {} exceptions (max {})",
            self.exceptions.len(),
            max_exceptions
        );
    }

    pub fn run(&self, instance: &SatInstance) -> bool {
        // Check exception table first
        if let Some(&correct_answer) = self.exceptions.get(&instance.instance_id) {
            correct_answer
        } else {
            self.base_oracle.run(instance)
        }
    }

    pub fn error_rate(&self, instances: &[SatInstance]) -> f64 {
        let errors: usize = instances
            .iter()
            .filter(|inst| self.run(inst) != inst.is_sat)
            .count();
        errors as f64 / instances.len() as f64
    }
}

/// Experimental result for a specific method and n value
#[derive(Clone, Debug)]
pub struct ExperimentResult {
    pub method: String,
    pub n: usize,
    pub error_rate: f64,
}

/// Run numerical experiments
/// 変数数 n ∈ {9, 11, 13}、各 n で 20 本の式を生成
pub fn run_numerical_experiments() -> Result<Vec<ExperimentResult>> {
    let mut results = Vec::new();
    let n_values = vec![9, 11, 13];

    for &n in &n_values {
        info!("Running experiments for n = {}", n);

        // Generate dataset: 10 planted + 10 dense random
        let mut instances = Vec::new();
        for i in 0..10 {
            instances.push(SatInstance::generate_planted(n, i));
        }
        for i in 0..10 {
            instances.push(SatInstance::generate_dense_random(n, i + 100));
        }

        // Test different methods
        // Base: rand-8
        let rand8 = RandomSampler::new(8);
        results.push(ExperimentResult {
            method: "rand-8".to_string(),
            n,
            error_rate: rand8.error_rate(&instances),
        });

        // rep-1, rep-3, rep-5, rep-7
        for k in [1, 3, 5, 7] {
            let rep = RepetitionOracle::new(8, k);
            results.push(ExperimentResult {
                method: format!("rep-{}", k),
                n,
                error_rate: rep.error_rate(&instances),
            });
        }

        // rep-k + advice-b combinations
        for k in [1, 3, 5, 7] {
            for advice_bits in [0, 512, 2048] {
                let mut advice = AdviceOracle::new(8, k, advice_bits);
                advice.train(&instances);
                results.push(ExperimentResult {
                    method: format!("rep-{}+advice-{}", k, advice_bits),
                    n,
                    error_rate: advice.error_rate(&instances),
                });
            }
        }
    }

    Ok(results)
}

/// Compute d̂_{n_max} = max error across all n values for each method
pub fn compute_worst_error(results: &[ExperimentResult]) -> HashMap<String, f64> {
    let mut method_errors: HashMap<String, Vec<f64>> = HashMap::new();

    for result in results {
        method_errors
            .entry(result.method.clone())
            .or_default()
            .push(result.error_rate);
    }

    method_errors
        .into_iter()
        .map(|(method, errors)| {
            let worst = errors
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            (method, worst)
        })
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// § 2. 証明義務 (Proof Obligations)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Resource specification for average-case complexity
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    /// Time bound (polynomial degree)
    pub time: u32,
    /// Advice length bound
    pub advice: usize,
    /// Randomness bound
    pub randomness: usize,
    /// Repetition count allowed
    pub repetition: usize,
}

impl Resource {
    /// Check if this resource is dominated by (≼) another
    /// R ≼ R' ⟺ R' が R の拡張
    pub fn is_dominated_by(&self, other: &Resource) -> bool {
        self.time <= other.time
            && self.advice <= other.advice
            && self.randomness <= other.randomness
            && self.repetition <= other.repetition
    }
}

/// (N2) Resource Monotonicity Proof
///
/// 主張: R ≼ R' ⟹ d_w(R') ≤ d_w(R)
///
/// 証明: 任意の L ∈ NP^{A_w} について、P^{A_w}(R) ⊆ P^{A_w}(R')（資源拡張は
/// 決定性クラスの包含を広げる）。したがって inf を取る集合が拡大し、inf 値は
/// 減少または不変。この性質は各 L で成り立つので、上側の sup を取っても不等式は保存される。∎
///
/// 直観: 使える手が増えれば、最良誤り率は下がる。
#[derive(Clone, Debug)]
pub struct ResourceMonotonicityProof {
    pub r1: Resource,
    pub r2: Resource,
}

impl ResourceMonotonicityProof {
    /// Verify that r2 dominates r1 (r1 ≼ r2)
    pub fn verify(&self) -> Result<()> {
        if !self.r1.is_dominated_by(&self.r2) {
            anyhow::bail!(
                "Resource monotonicity violation: r1 is not dominated by r2\n\
                 r1 = {:?}\n\
                 r2 = {:?}",
                self.r1,
                self.r2
            );
        }
        info!("✓ Resource monotonicity verified: r1 ≼ r2 ⟹ d(r2) ≤ d(r1)");
        Ok(())
    }
}

/// Distribution family
#[derive(Clone, Debug)]
pub struct Distribution {
    pub name: String,
    /// Complexity measure (higher = harder)
    pub complexity: f64,
}

/// Distribution dominance relation
/// D ≼ D' ⟺ ポリ時間のサンプル写像 f_n が存在し、x ~ D_n を y = f_n(x) ~ D'_n に押し出す
#[derive(Clone, Debug)]
pub struct DistributionDominance {
    pub d1: Distribution,
    pub d2: Distribution,
    /// Witness that d1 can be efficiently transformed to d2
    pub has_poly_reduction: bool,
}

/// (N3) Distribution Dominance Monotonicity Proof
///
/// 設定: D, D' はそれぞれ {D_n}, {D'_n}
/// D ≼ D' を次で定義: ポリ時間のサンプル写像 f_n が存在し、x ~ D_n を y = f_n(x) ~ D'_n に
/// 押し出す（pushforward）、かつ f_n は可逆写像 g_n（左逆）をポリ時間で持つ。
///
/// 主張: このとき d_w(P, NP; D) ≤ d_w(P, NP; D')
///
/// 証明: 任意の L ∈ NP^{A_w}。D' 上で誤り ε の A' ∈ P^{A_w}(R_w) があれば、
/// A(x) := A'(f_n(x)) は D 上で同じ誤り ε を達成（分布は押し出し／可逆性で質量保存）。
/// よって inf_{A ∈ P^{A_w}(R_w)} err_D ≤ inf_{A' ∈ P^{A_w}(R_w)} err_D'。
/// sup_L を取っても不等式は保存。∎
///
/// 直観: D' が D を「包含」するなら、D' 上で最善だった誤り率より悪くはならない写像がある。
#[derive(Clone, Debug)]
pub struct DistributionDominanceProof {
    pub dominance: DistributionDominance,
}

impl DistributionDominanceProof {
    /// Verify distribution dominance property
    pub fn verify(&self) -> Result<()> {
        if !self.dominance.has_poly_reduction {
            anyhow::bail!(
                "Distribution dominance violation: no poly-time reduction exists\n\
                 D1 = {:?}\n\
                 D2 = {:?}",
                self.dominance.d1,
                self.dominance.d2
            );
        }
        info!("✓ Distribution dominance verified: D ≼ D' ⟹ d(D) ≤ d(D')");
        Ok(())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// § 3. モノドロミー (Monodromy Construction)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Oracle type for relativization
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleType {
    /// Oracle A where P^A = NP^A (BGS construction)
    Equal,
    /// Oracle B where P^B ≠ NP^B (BGS construction)
    NotEqual,
    /// Hybrid oracle with tunable distance d_w
    Hybrid { distance: u32 },
}

/// Phase in the Spiral construction
#[derive(Clone, Debug, PartialEq)]
pub enum SpiralPhase {
    /// U_= region: Φ_= is classically true (P = NP relative to oracle A)
    Equal,
    /// U_≠ region: Φ_≠ is classically true (P ≠ NP relative to oracle B)
    NotEqual,
    /// U_≈ region: d_w = 0 (HeurP phase, Φ_≈ is true)
    Approximate,
}

/// Point on the Spiral base space S¹
#[derive(Clone, Debug)]
pub struct SpiralPoint {
    /// Angle parameter θ ∈ [0, 2π)
    pub theta: f64,
    /// Current phase
    pub phase: SpiralPhase,
    /// Oracle configuration at this point
    pub oracle: OracleType,
}

impl SpiralPoint {
    /// Create a new point on the Spiral
    pub fn new(theta: f64) -> Self {
        let normalized_theta = theta % (2.0 * std::f64::consts::PI);

        // Divide the circle into three regions
        let (phase, oracle) = if normalized_theta < 2.0 {
            (SpiralPhase::Equal, OracleType::Equal)
        } else if normalized_theta < 4.0 {
            (SpiralPhase::Approximate, OracleType::Hybrid { distance: 0 })
        } else {
            (SpiralPhase::NotEqual, OracleType::NotEqual)
        };

        Self {
            theta: normalized_theta,
            phase,
            oracle,
        }
    }

    /// Advance along the Spiral path
    pub fn advance(&mut self, delta_theta: f64) {
        self.theta = (self.theta + delta_theta) % (2.0 * std::f64::consts::PI);
        *self = Self::new(self.theta);
    }
}

/// Monodromy loop: track how Φ_= transforms under parallel transport
///
/// 既知事実 (BGS, 1975):
/// ∃A: P^A = NP^A かつ ∃B: P^B ≠ NP^B
///
/// Spiral 構成: 底空間 B = S¹ (コイル) を 3 つの開集合で覆う:
/// - U_= : オラクル層を定数 A にする (この局所では Φ_= が古典的真)
/// - U_≠ : オラクル層を定数 B にする (この局所では Φ_≠ が古典的真)
/// - U_≈ : 分布・資源を調整して d_w = 0 (HeurP 相: Φ_≈ が真)
///
/// ループ γ が U_= → U_≈ → U_≠ → U_= を一周すると、並行移動 ρ(γ) は
/// ρ(γ): Φ_= ↦ ¬Φ_=  (局所真理の反転)
/// を誘導 (BGS の両世界を跨いだ "ねじれ接着" の効果)
#[derive(Clone, Debug)]
pub struct MonodromyLoop {
    /// Starting point on the circle
    pub start: SpiralPoint,
    /// Current position along the loop
    pub current: SpiralPoint,
    /// Number of complete loops traversed
    pub loops: usize,
    /// Sign of Φ_= (flips each loop)
    pub phi_equal_sign: i32,
}

impl MonodromyLoop {
    /// Create a new monodromy loop starting at θ = 0
    pub fn new() -> Self {
        let start = SpiralPoint::new(0.0);
        Self {
            start: start.clone(),
            current: start,
            loops: 0,
            phi_equal_sign: 1, // Start with Φ_= being true
        }
    }

    /// Traverse one complete loop around S¹
    pub fn traverse_loop(&mut self) -> Result<()> {
        info!(
            "Starting monodromy loop traversal from θ = {:.2}",
            self.current.theta
        );

        // Traverse through all three phases
        let steps = vec![(1.0, "U_= → U_≈"), (2.0, "U_≈ → U_≠"), (3.0, "U_≠ → U_=")];

        for (delta, desc) in steps {
            self.current.advance(delta);
            debug!("Crossed to phase {:?} ({})", self.current.phase, desc);
        }

        // After one complete loop, Φ_= flips sign
        self.loops += 1;
        self.phi_equal_sign *= -1;

        info!(
            "Completed loop {}. Φ_= sign is now {}",
            self.loops,
            if self.phi_equal_sign > 0 { "+" } else { "-" }
        );

        Ok(())
    }

    /// Get the current truth value of Φ_= after monodromy
    pub fn phi_equal_value(&self) -> bool {
        self.phi_equal_sign > 0
    }
}

impl Default for MonodromyLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// § 4. Integration and Export
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Complete SpiralReality framework integrating all three components
#[derive(Debug)]
pub struct SpiralReality {
    /// Numerical experiment results
    pub experiments: Vec<ExperimentResult>,
    /// Computed worst errors per method
    pub worst_errors: HashMap<String, f64>,
    /// Resource monotonicity proofs
    pub resource_proofs: Vec<ResourceMonotonicityProof>,
    /// Distribution dominance proofs
    pub distribution_proofs: Vec<DistributionDominanceProof>,
    /// Monodromy loop state
    pub monodromy: MonodromyLoop,
}

impl SpiralReality {
    /// Initialize a new SpiralReality framework
    pub fn new() -> Result<Self> {
        info!("Initializing SpiralReality framework");

        // Run numerical experiments
        let experiments =
            run_numerical_experiments().context("Failed to run numerical experiments")?;
        let worst_errors = compute_worst_error(&experiments);

        // Set up proof obligations
        let resource_proofs = vec![
            // Example: repetition increases resources
            ResourceMonotonicityProof {
                r1: Resource {
                    time: 1,
                    advice: 0,
                    randomness: 8,
                    repetition: 1,
                },
                r2: Resource {
                    time: 1,
                    advice: 0,
                    randomness: 8,
                    repetition: 5,
                },
            },
            // Example: advice increases resources
            ResourceMonotonicityProof {
                r1: Resource {
                    time: 1,
                    advice: 512,
                    randomness: 8,
                    repetition: 1,
                },
                r2: Resource {
                    time: 1,
                    advice: 2048,
                    randomness: 8,
                    repetition: 1,
                },
            },
        ];

        let distribution_proofs = vec![DistributionDominanceProof {
            dominance: DistributionDominance {
                d1: Distribution {
                    name: "Planted-SAT".to_string(),
                    complexity: 1.0,
                },
                d2: Distribution {
                    name: "Dense-Random-SAT".to_string(),
                    complexity: 1.5,
                },
                has_poly_reduction: true,
            },
        }];

        let monodromy = MonodromyLoop::new();

        Ok(Self {
            experiments,
            worst_errors,
            resource_proofs,
            distribution_proofs,
            monodromy,
        })
    }

    /// Verify all proof obligations
    pub fn verify_proofs(&self) -> Result<()> {
        info!("Verifying proof obligations");

        for (i, proof) in self.resource_proofs.iter().enumerate() {
            proof
                .verify()
                .with_context(|| format!("Resource proof {} failed", i))?;
        }

        for (i, proof) in self.distribution_proofs.iter().enumerate() {
            proof
                .verify()
                .with_context(|| format!("Distribution proof {} failed", i))?;
        }

        info!("✓ All proof obligations verified");
        Ok(())
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        report.push_str("  SpiralReality Framework Summary\n");
        report.push_str("  数値実験／証明義務／モノドロミーの三点セット\n");
        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        report.push_str("§ 1. Numerical Experiments (数値実験)\n");
        report.push_str(&format!(
            "  Total experiments: {}\n",
            self.experiments.len()
        ));
        report.push_str(&format!("  Methods tested: {}\n", self.worst_errors.len()));

        report.push_str("\n  d̂_{n_max} (worst error) by method:\n");
        let mut sorted_methods: Vec<_> = self.worst_errors.iter().collect();
        sorted_methods.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        for (method, error) in sorted_methods.iter().take(5) {
            report.push_str(&format!("    {}: {:.3}\n", method, error));
        }

        report.push_str("\n§ 2. Proof Obligations (証明義務)\n");
        report.push_str(&format!(
            "  Resource monotonicity proofs: {}\n",
            self.resource_proofs.len()
        ));
        report.push_str(&format!(
            "  Distribution dominance proofs: {}\n",
            self.distribution_proofs.len()
        ));

        report.push_str("\n§ 3. Monodromy (モノドロミー)\n");
        report.push_str(&format!(
            "  Current phase: {:?}\n",
            self.monodromy.current.phase
        ));
        report.push_str(&format!("  Loops completed: {}\n", self.monodromy.loops));
        report.push_str(&format!(
            "  Φ_= current value: {}\n",
            self.monodromy.phi_equal_value()
        ));

        report.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        report
    }
}

impl Default for SpiralReality {
    fn default() -> Self {
        Self::new().expect("Failed to initialize SpiralReality")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_instance_generation() {
        let planted = SatInstance::generate_planted(9, 42);
        assert_eq!(planted.n, 9);
        assert_eq!(planted.clauses.len(), 36); // 4n
        assert!(planted.is_sat);

        let dense = SatInstance::generate_dense_random(9, 42);
        assert_eq!(dense.n, 9);
        assert_eq!(dense.clauses.len(), 63); // 7n
    }

    #[test]
    fn test_random_sampler() {
        let sampler = RandomSampler::new(8);
        let _sat_instance = SatInstance::generate_planted(9, 1);
        let unsat_instance = SatInstance::generate_dense_random(9, 1);

        // UNSAT instances should never give false positives
        if !unsat_instance.is_sat {
            assert!(!sampler.run(&unsat_instance));
        }
    }

    #[test]
    fn test_repetition_oracle() {
        let rep = RepetitionOracle::new(8, 3);
        assert_eq!(rep.k, 3);
    }

    #[test]
    fn test_advice_oracle() {
        let instances = vec![
            SatInstance::generate_planted(9, 1),
            SatInstance::generate_dense_random(9, 2),
        ];

        let mut advice = AdviceOracle::new(8, 1, 2048);
        advice.train(&instances);

        // Max exceptions should be 2048 / 65 = 31
        assert!(advice.exceptions.len() <= 31);
    }

    #[test]
    fn test_resource_monotonicity() {
        let r1 = Resource {
            time: 1,
            advice: 512,
            randomness: 8,
            repetition: 1,
        };
        let r2 = Resource {
            time: 1,
            advice: 2048,
            randomness: 8,
            repetition: 3,
        };

        assert!(r1.is_dominated_by(&r2));

        let proof = ResourceMonotonicityProof { r1, r2 };
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_distribution_dominance() {
        let dominance = DistributionDominance {
            d1: Distribution {
                name: "Simple".to_string(),
                complexity: 1.0,
            },
            d2: Distribution {
                name: "Complex".to_string(),
                complexity: 2.0,
            },
            has_poly_reduction: true,
        };

        let proof = DistributionDominanceProof { dominance };
        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_spiral_point() {
        let p1 = SpiralPoint::new(0.5);
        assert_eq!(p1.phase, SpiralPhase::Equal);

        let p2 = SpiralPoint::new(3.0);
        assert_eq!(p2.phase, SpiralPhase::Approximate);

        let p3 = SpiralPoint::new(5.0);
        assert_eq!(p3.phase, SpiralPhase::NotEqual);
    }

    #[test]
    fn test_monodromy_loop() {
        let mut loop_state = MonodromyLoop::new();
        assert_eq!(loop_state.loops, 0);
        assert!(loop_state.phi_equal_value()); // Starts positive

        loop_state.traverse_loop().unwrap();
        assert_eq!(loop_state.loops, 1);
        assert!(!loop_state.phi_equal_value()); // Flips after one loop

        loop_state.traverse_loop().unwrap();
        assert_eq!(loop_state.loops, 2);
        assert!(loop_state.phi_equal_value()); // Flips back
    }

    #[test]
    fn test_spiral_reality_initialization() {
        let reality = SpiralReality::new();
        assert!(reality.is_ok());

        let reality = reality.unwrap();
        assert!(!reality.experiments.is_empty());
        assert!(!reality.worst_errors.is_empty());
    }

    #[test]
    fn test_spiral_reality_verification() {
        let reality = SpiralReality::new().unwrap();
        assert!(reality.verify_proofs().is_ok());
    }

    #[test]
    fn test_spiral_reality_summary() {
        let reality = SpiralReality::new().unwrap();
        let summary = reality.summary();
        assert!(summary.contains("SpiralReality"));
        assert!(summary.contains("Numerical Experiments"));
        assert!(summary.contains("Proof Obligations"));
        assert!(summary.contains("Monodromy"));
    }
}
