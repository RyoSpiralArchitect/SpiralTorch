// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{HashMap, HashSet, VecDeque};

pub type NodeId = u64;

/// Describes a single operation in the causal graph together with the
/// estimated contribution it has on the final objective.
#[derive(Clone, Debug)]
pub struct CausalNode {
    pub id: NodeId,
    pub label: String,
    pub parents: Vec<NodeId>,
    /// Estimated absolute influence this node has on the final result.
    pub effect: f32,
    /// Relative cost/latency budget associated with the node.
    pub cost: f32,
}

impl CausalNode {
    pub fn new(id: NodeId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            parents: Vec::new(),
            effect: 1.0,
            cost: 1.0,
        }
    }

    pub fn with_parents(mut self, parents: Vec<NodeId>) -> Self {
        self.parents = parents;
        self
    }

    pub fn with_effect(mut self, effect: f32) -> Self {
        self.effect = effect.max(0.0);
        self
    }

    pub fn with_cost(mut self, cost: f32) -> Self {
        self.cost = cost.max(0.0);
        self
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CausalCompileError {
    #[error("duplicate node id {0}")]
    DuplicateNode(NodeId),
    #[error("unknown dependency {0}")]
    UnknownDependency(NodeId),
    #[error("cycle detected while compiling causal graph")]
    CycleDetected,
    #[error("skip threshold must be non-negative, got {0}")]
    NegativeThreshold(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum SkipReason {
    LowEffect,
    BudgetExceeded,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExecutionDecision {
    Execute { aggregated_effect: f32 },
    Skip(SkipReason),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionStep {
    pub id: NodeId,
    pub label: String,
    pub parents: Vec<NodeId>,
    pub decision: ExecutionDecision,
    pub cost: f32,
}

#[derive(Clone, Debug)]
pub struct CompiledPlan {
    steps: Vec<ExecutionStep>,
    skip_threshold: f32,
    total_budget: Option<f32>,
    spent_budget: f32,
    skipped: HashSet<NodeId>,
}

impl CompiledPlan {
    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }

    pub fn should_execute(&self, id: NodeId) -> bool {
        !self.skipped.contains(&id)
    }

    pub fn extend_budget(&mut self, delta: f32) {
        if delta <= 0.0 {
            return;
        }
        match self.total_budget {
            Some(ref mut total) => {
                *total += delta;
            }
            None => {
                self.total_budget = Some(self.spent_budget + delta);
            }
        }
    }

    pub fn adapt_with_observation(&mut self, id: NodeId, new_effect: f32) {
        if let Some(step) = self.steps.iter_mut().find(|s| s.id == id) {
            let currently_executed = matches!(step.decision, ExecutionDecision::Execute { .. });
            let mut available_budget = self
                .total_budget
                .map(|total| (total - self.spent_budget).max(0.0));

            let decision = if new_effect.abs() < self.skip_threshold {
                if currently_executed {
                    self.spent_budget = (self.spent_budget - step.cost).max(0.0);
                }
                self.skipped.insert(id);
                ExecutionDecision::Skip(SkipReason::LowEffect)
            } else if let Some(remaining) = available_budget.as_mut() {
                if !currently_executed && step.cost > *remaining {
                    self.skipped.insert(id);
                    ExecutionDecision::Skip(SkipReason::BudgetExceeded)
                } else {
                    if !currently_executed {
                        *remaining -= step.cost;
                        self.spent_budget += step.cost;
                    }
                    self.skipped.remove(&id);
                    ExecutionDecision::Execute {
                        aggregated_effect: new_effect,
                    }
                }
            } else {
                if !currently_executed {
                    self.spent_budget += step.cost;
                }
                self.skipped.remove(&id);
                ExecutionDecision::Execute {
                    aggregated_effect: new_effect,
                }
            };
            step.decision = decision;
        }
    }
}

#[derive(Default)]
pub struct CausalGraph {
    nodes: HashMap<NodeId, CausalNode>,
}

impl CausalGraph {
    pub fn insert(&mut self, node: CausalNode) -> Result<(), CausalCompileError> {
        if self.nodes.contains_key(&node.id) {
            return Err(CausalCompileError::DuplicateNode(node.id));
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    pub fn compile_plan(
        &self,
        skip_threshold: f32,
        latency_budget: Option<f32>,
    ) -> Result<CompiledPlan, CausalCompileError> {
        if skip_threshold < 0.0 {
            return Err(CausalCompileError::NegativeThreshold(skip_threshold));
        }

        for node in self.nodes.values() {
            for parent in &node.parents {
                if !self.nodes.contains_key(parent) {
                    return Err(CausalCompileError::UnknownDependency(*parent));
                }
            }
        }

        let order = self.topological_order()?;
        let mut remaining_budget = latency_budget;
        let mut steps = Vec::with_capacity(order.len());
        let mut active_effect: HashMap<NodeId, f32> = HashMap::new();
        let mut skipped = HashSet::new();
        let mut skip_reasons: HashMap<NodeId, SkipReason> = HashMap::new();
        let mut spent_budget = 0.0;

        for id in order {
            let node = &self.nodes[&id];
            let parent_effect = if node.parents.is_empty() {
                1.0
            } else {
                node.parents
                    .iter()
                    .map(|p| {
                        if let Some(effect) = active_effect.get(p) {
                            *effect
                        } else if matches!(skip_reasons.get(p), Some(SkipReason::LowEffect)) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .fold(1.0, |acc, e| acc * e)
            };
            let aggregate = parent_effect.abs() * node.effect.abs();
            let decision = if aggregate < skip_threshold {
                skipped.insert(id);
                skip_reasons.insert(id, SkipReason::LowEffect);
                ExecutionDecision::Skip(SkipReason::LowEffect)
            } else if let Some(budget) = remaining_budget.as_mut() {
                if node.cost > *budget {
                    skipped.insert(id);
                    skip_reasons.insert(id, SkipReason::BudgetExceeded);
                    ExecutionDecision::Skip(SkipReason::BudgetExceeded)
                } else {
                    *budget -= node.cost;
                    spent_budget += node.cost;
                    skipped.remove(&id);
                    skip_reasons.remove(&id);
                    ExecutionDecision::Execute {
                        aggregated_effect: aggregate,
                    }
                }
            } else {
                skipped.remove(&id);
                skip_reasons.remove(&id);
                ExecutionDecision::Execute {
                    aggregated_effect: aggregate,
                }
            };
            if matches!(decision, ExecutionDecision::Execute { .. }) {
                active_effect.insert(id, aggregate.max(1e-6));
            } else {
                active_effect.remove(&id);
            match decision {
                ExecutionDecision::Execute { .. } => {
                    active_effect.insert(id, aggregate.max(1e-6));
                }
                ExecutionDecision::Skip(SkipReason::LowEffect) => {
                    active_effect.insert(id, 1.0);
                }
                _ => {
                    active_effect.insert(id, 0.0);
                }
            }
            steps.push(ExecutionStep {
                id,
                label: node.label.clone(),
                parents: node.parents.clone(),
                decision,
                cost: node.cost,
            });
        }

        Ok(CompiledPlan {
            steps,
            skip_threshold,
            total_budget: latency_budget,
            spent_budget,
            skipped,
        })
    }

    fn topological_order(&self) -> Result<Vec<NodeId>, CausalCompileError> {
        let mut indegree: HashMap<NodeId, usize> =
            self.nodes.values().map(|node| (node.id, 0)).collect();
        for node in self.nodes.values() {
            for parent in &node.parents {
                *indegree.entry(node.id).or_default() += 1;
                if !indegree.contains_key(parent) {
                    indegree.insert(*parent, 0);
                }
            }
        }
        let mut queue: VecDeque<NodeId> = indegree
            .iter()
            .filter_map(|(&id, &deg)| if deg == 0 { Some(id) } else { None })
            .collect();
        let mut order = Vec::new();

        let mut remaining = indegree.clone();
        while let Some(id) = queue.pop_front() {
            order.push(id);
            for child in self.nodes.values().filter(|n| n.parents.contains(&id)) {
                if let Some(entry) = remaining.get_mut(&child.id) {
                    *entry -= 1;
                    if *entry == 0 {
                        queue.push_back(child.id);
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(CausalCompileError::CycleDetected);
        }

        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> CausalGraph {
        let mut graph = CausalGraph::default();
        graph
            .insert(CausalNode::new(1, "input").with_effect(1.0).with_cost(0.1))
            .unwrap();
        graph
            .insert(
                CausalNode::new(2, "spectral")
                    .with_parents(vec![1])
                    .with_effect(0.8)
                    .with_cost(0.3),
            )
            .unwrap();
        graph
            .insert(
                CausalNode::new(3, "refine")
                    .with_parents(vec![2])
                    .with_effect(0.05)
                    .with_cost(0.5),
            )
            .unwrap();
        graph
            .insert(
                CausalNode::new(4, "decode")
                    .with_parents(vec![2, 3])
                    .with_effect(1.2)
                    .with_cost(0.4),
            )
            .unwrap();
        graph
    }

    #[test]
    fn skips_low_effect_nodes() {
        let graph = sample_graph();
        let plan = graph.compile_plan(0.1, None).unwrap();
        assert_eq!(plan.steps().len(), 4);
        assert!(plan.should_execute(1));
        assert!(plan.should_execute(2));
        assert!(!plan.should_execute(3));
        assert!(plan.should_execute(4));
    }

    #[test]
    fn enforces_budget_and_updates() {
        let graph = sample_graph();
        let mut plan = graph.compile_plan(0.01, Some(0.6)).unwrap();
        assert!(plan.should_execute(1));
        assert!(plan.should_execute(2));
        assert!(!plan.should_execute(3));
        // decode would exceed the remaining budget (0.6 - 0.1 - 0.3 = 0.2 < 0.4)
        assert!(!plan.should_execute(4));

        // provide a runtime observation showing decode is critical; adapts if budget allows
        plan.adapt_with_observation(4, 0.9);
        assert!(!plan.should_execute(4));

        // mark the refine stage as negligible which frees its budget
        plan.adapt_with_observation(3, 0.0);
        assert!(!plan.should_execute(3));

        // grant a bit more headroom from the runtime monitor
        plan.extend_budget(0.3);

        // freeing budget allows decode to be re-enabled
        plan.adapt_with_observation(4, 1.1);
        assert!(plan.should_execute(4));
    }

    #[test]
    fn reject_cycles() {
        let mut graph = CausalGraph::default();
        graph
            .insert(CausalNode::new(1, "a").with_parents(vec![2]))
            .unwrap();
        graph
            .insert(CausalNode::new(2, "b").with_parents(vec![1]))
            .unwrap();
        assert_eq!(
            graph.compile_plan(0.0, None).unwrap_err(),
            CausalCompileError::CycleDetected
        );
    }
}
