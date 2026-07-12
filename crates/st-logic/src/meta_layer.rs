// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Meta-narrative layer that bridges Z-space causality with categorical
//! semantics so higher-level studios inherit a temporal sense of meaning.
//!
//! The layer is composed of three cooperating pieces:
//!
//! * [`CausalSet`] — Maintains a partially ordered set of narrative beats.
//! * [`TemporalLogicEngine`] — Advances the causal frontier and yields beats in
//!   a manner consistent with the partial order.
//! * [`ToposLogicBridge`] — Attaches a sheaf of meanings to each beat so the
//!   resolved narrative honours categorical restrictions.
//!
//! [`MetaNarrativeLayer`] stitches the components together and exposes a
//! high-level interface that QuantumRealityStudio can consult when deciding
//! what it should speak about next.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Deref;

use crate::quantum_reality::ZSpace;
use st_core::maxwell::MaxwellZPulse;
use thiserror::Error;

/// Error emitted when a causal relation cannot be established.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CausalError {
    /// Event identifiers must contain at least one non-whitespace character.
    #[error("causal event identifier cannot be empty")]
    EmptyEventId,
    /// The provided identifier is already registered inside the set.
    #[error("causal event `{0}` is already registered")]
    DuplicateEvent(String),
    /// Resolved identifiers are immutable history and cannot be registered again.
    #[error("causal event `{0}` has already been resolved")]
    ResolvedEvent(String),
    /// Either the source or the destination event is missing.
    #[error("causal event `{0}` is not registered")]
    UnknownEvent(String),
    /// The requested dependency would introduce a cycle.
    #[error("causal relation `{0}` -> `{1}` would create a cycle")]
    CyclicDependency(String, String),
    /// An event cannot depend on itself.
    #[error("causal event cannot depend on itself")]
    SelfDependency,
}

/// Structural inconsistency detected inside a causal set or temporal frontier.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CausalInvariantError {
    #[error("causal node key `{key}` does not match embedded identifier `{node_id}`")]
    NodeIdMismatch { key: String, node_id: String },
    #[error("resolved causal event `{0}` is still live")]
    ResolvedNode(String),
    #[error("live causal event `{0}` has no successor index")]
    MissingSuccessorIndex(String),
    #[error("successor index exists for non-live event `{0}`")]
    UnknownSuccessorIndex(String),
    #[error("causal successor `{parent}` -> `{child}` points to a non-live child")]
    UnknownSuccessor { parent: String, child: String },
    #[error("causal successor `{parent}` -> `{child}` is missing from the child parent set")]
    MissingParentLink { parent: String, child: String },
    #[error("causal event `{child}` references unknown parent `{parent}`")]
    DanglingParent { parent: String, child: String },
    #[error("causal event `{child}` references live parent `{parent}` without a successor link")]
    MissingSuccessorLink { parent: String, child: String },
    #[error("causal graph contains a dependency cycle")]
    CyclicGraph,
    #[error("causal frontier contains duplicate event `{0}`")]
    DuplicateFrontierEvent(String),
    #[error("causal frontier contains non-live event `{0}`")]
    UnknownFrontierEvent(String),
    #[error("causal frontier contains blocked event `{0}`")]
    BlockedFrontierEvent(String),
    #[error("ready causal event `{0}` is missing from the frontier")]
    MissingFrontierEvent(String),
}

/// Node stored inside the [`CausalSet`].
#[derive(Clone, Debug)]
pub struct CausalNode<T> {
    id: String,
    payload: T,
    parents: BTreeSet<String>,
}

impl<T> CausalNode<T> {
    fn new(id: String, payload: T) -> Self {
        Self {
            id,
            payload,
            parents: BTreeSet::new(),
        }
    }

    /// Returns the event identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the event payload.
    pub fn payload(&self) -> &T {
        &self.payload
    }

    /// Iterates over predecessor identifiers in deterministic order.
    pub fn parent_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.parents.iter().map(String::as_str)
    }

    /// Returns true when all parent identifiers have been resolved.
    fn is_ready(&self, resolved: &BTreeSet<String>) -> bool {
        self.parents.iter().all(|parent| resolved.contains(parent))
    }
}

/// Partially ordered set describing the causal relations between narrative
/// beats.
#[derive(Clone, Debug)]
pub struct CausalSet<T> {
    nodes: BTreeMap<String, CausalNode<T>>,
    successors: BTreeMap<String, BTreeSet<String>>,
    resolved: BTreeSet<String>,
}

impl<T> CausalSet<T> {
    /// Creates an empty causal set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of registered events.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true when no events have been registered.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of event identifiers retained as resolved history.
    pub fn resolved_len(&self) -> usize {
        self.resolved.len()
    }

    /// Returns true when an identifier has already been resolved.
    pub fn was_resolved(&self, id: &str) -> bool {
        self.resolved.contains(id)
    }

    /// Inserts a new event into the causal set.
    pub fn insert(&mut self, id: impl Into<String>, payload: T) -> Result<(), CausalError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(CausalError::EmptyEventId);
        }
        if self.nodes.contains_key(&id) {
            return Err(CausalError::DuplicateEvent(id));
        }
        if self.resolved.contains(&id) {
            return Err(CausalError::ResolvedEvent(id));
        }
        self.successors.entry(id.clone()).or_default();
        self.nodes.insert(id.clone(), CausalNode::new(id, payload));
        Ok(())
    }

    /// Registers that `earlier` must precede `later`.
    /// A resolved predecessor is accepted as already-satisfied history.
    pub fn add_dependency(
        &mut self,
        earlier: impl AsRef<str>,
        later: impl AsRef<str>,
    ) -> Result<(), CausalError> {
        let earlier = earlier.as_ref();
        let later = later.as_ref();
        if earlier == later {
            return Err(CausalError::SelfDependency);
        }
        let earlier_is_live = if self.nodes.contains_key(earlier) {
            true
        } else if self.resolved.contains(earlier) {
            false
        } else {
            return Err(CausalError::UnknownEvent(earlier.to_string()));
        };
        self.ensure_live(later)?;
        if earlier_is_live && self.would_create_cycle(earlier, later) {
            return Err(CausalError::CyclicDependency(
                earlier.to_string(),
                later.to_string(),
            ));
        }
        let Some(later_node) = self.nodes.get_mut(later) else {
            return Err(CausalError::UnknownEvent(later.to_string()));
        };
        later_node.parents.insert(earlier.to_string());
        if earlier_is_live {
            self.successors
                .entry(earlier.to_string())
                .or_default()
                .insert(later.to_string());
        }
        Ok(())
    }

    /// Returns a reference to the stored node.
    pub fn get(&self, id: &str) -> Option<&CausalNode<T>> {
        self.nodes.get(id)
    }

    /// Returns true when a live event is ready to resolve.
    pub fn is_ready(&self, id: &str) -> bool {
        self.nodes
            .get(id)
            .is_some_and(|node| node.is_ready(&self.resolved))
    }

    /// Resolves an event and retains its identifier as immutable history.
    pub fn resolve(&mut self, id: &str) -> Option<(String, T)> {
        let node = self.nodes.remove(id)?;
        self.successors.remove(id);
        for parent in &node.parents {
            if let Some(successors) = self.successors.get_mut(parent) {
                successors.remove(id);
            }
        }
        self.resolved.insert(node.id.clone());
        Some((node.id, node.payload))
    }

    /// Compatibility alias for [`Self::resolve`].
    pub fn take(&mut self, id: &str) -> Option<(String, T)> {
        self.resolve(id)
    }

    /// Verifies the bidirectional DAG index and resolution history.
    pub fn validate(&self) -> Result<(), CausalInvariantError> {
        for (id, node) in &self.nodes {
            if id != &node.id {
                return Err(CausalInvariantError::NodeIdMismatch {
                    key: id.clone(),
                    node_id: node.id.clone(),
                });
            }
            if self.resolved.contains(id) {
                return Err(CausalInvariantError::ResolvedNode(id.clone()));
            }
            if !self.successors.contains_key(id) {
                return Err(CausalInvariantError::MissingSuccessorIndex(id.clone()));
            }
            for parent in &node.parents {
                if self.resolved.contains(parent) {
                    continue;
                }
                if !self.nodes.contains_key(parent) {
                    return Err(CausalInvariantError::DanglingParent {
                        parent: parent.clone(),
                        child: id.clone(),
                    });
                }
                if !self
                    .successors
                    .get(parent)
                    .is_some_and(|children| children.contains(id))
                {
                    return Err(CausalInvariantError::MissingSuccessorLink {
                        parent: parent.clone(),
                        child: id.clone(),
                    });
                }
            }
        }

        for (parent, children) in &self.successors {
            if !self.nodes.contains_key(parent) {
                return Err(CausalInvariantError::UnknownSuccessorIndex(parent.clone()));
            }
            for child in children {
                let Some(child_node) = self.nodes.get(child) else {
                    return Err(CausalInvariantError::UnknownSuccessor {
                        parent: parent.clone(),
                        child: child.clone(),
                    });
                };
                if !child_node.parents.contains(parent) {
                    return Err(CausalInvariantError::MissingParentLink {
                        parent: parent.clone(),
                        child: child.clone(),
                    });
                }
            }
        }
        self.validate_acyclic()?;
        Ok(())
    }

    fn validate_acyclic(&self) -> Result<(), CausalInvariantError> {
        let mut indegree = self
            .nodes
            .iter()
            .map(|(id, node)| {
                let live_parents = node
                    .parents
                    .iter()
                    .filter(|parent| self.nodes.contains_key(parent.as_str()))
                    .count();
                (id.as_str(), live_parents)
            })
            .collect::<BTreeMap<_, _>>();
        let mut ready = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect::<VecDeque<_>>();
        let mut visited = 0usize;

        while let Some(parent) = ready.pop_front() {
            visited += 1;
            let Some(children) = self.successors.get(parent) else {
                continue;
            };
            for child in children {
                let Some(degree) = indegree.get_mut(child.as_str()) else {
                    continue;
                };
                if *degree == 0 {
                    continue;
                }
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(child);
                }
            }
        }

        if visited == self.nodes.len() {
            Ok(())
        } else {
            Err(CausalInvariantError::CyclicGraph)
        }
    }

    fn ensure_live(&self, id: &str) -> Result<(), CausalError> {
        if self.nodes.contains_key(id) {
            Ok(())
        } else if self.resolved.contains(id) {
            Err(CausalError::ResolvedEvent(id.to_string()))
        } else {
            Err(CausalError::UnknownEvent(id.to_string()))
        }
    }

    fn ready_ids(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_ready(&self.resolved))
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn would_create_cycle(&self, parent: &str, child: &str) -> bool {
        if parent == child {
            return true;
        }
        let mut stack: Vec<String> = vec![child.to_string()];
        let mut visited = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) {
                continue;
            }
            if current == parent {
                return true;
            }
            if let Some(children) = self.successors.get(&current) {
                stack.extend(children.iter().cloned());
            }
        }
        false
    }
}

impl<T> Default for CausalSet<T> {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            successors: BTreeMap::new(),
            resolved: BTreeSet::new(),
        }
    }
}

/// Advances a [`CausalSet`] according to temporal logic rules.
#[derive(Clone, Debug)]
pub struct TemporalLogicEngine<T> {
    causal: CausalSet<T>,
    frontier: VecDeque<String>,
}

impl<T> TemporalLogicEngine<T> {
    /// Builds an engine from an existing causal set.
    pub fn from_causal_set(causal: CausalSet<T>) -> Self {
        let mut engine = Self {
            frontier: VecDeque::new(),
            causal,
        };
        engine.refresh_frontier();
        engine
    }

    /// Creates an empty engine.
    pub fn new() -> Self {
        Self::from_causal_set(CausalSet::new())
    }

    /// Returns the number of events still waiting to be emitted.
    pub fn pending(&self) -> usize {
        self.causal.len()
    }

    /// Returns the underlying causal set.
    pub fn causal(&self) -> &CausalSet<T> {
        &self.causal
    }

    /// Returns a mutation guard that refreshes the frontier when dropped.
    pub fn causal_mut(&mut self) -> CausalSetMut<'_, T> {
        CausalSetMut { engine: self }
    }

    /// Registers a new event.
    pub fn insert_event(&mut self, id: impl Into<String>, payload: T) -> Result<(), CausalError> {
        self.causal.insert(id.into(), payload)?;
        self.refresh_frontier();
        Ok(())
    }

    /// Adds a dependency inside the underlying causal set.
    pub fn add_dependency(
        &mut self,
        earlier: impl AsRef<str>,
        later: impl AsRef<str>,
    ) -> Result<(), CausalError> {
        self.causal.add_dependency(earlier, later)?;
        self.refresh_frontier();
        Ok(())
    }

    /// Resolves an event without returning it through the normal frontier.
    /// Descendants that depended on it become eligible immediately.
    pub fn discard_event(&mut self, id: &str) -> Option<(String, T)> {
        let resolved = self.causal.resolve(id);
        self.refresh_frontier();
        resolved
    }

    /// Returns true when an identifier has already left the live causal set.
    pub fn was_resolved(&self, id: &str) -> bool {
        self.causal.was_resolved(id)
    }

    /// Returns the next ready event if one exists.
    pub fn next_ready(&mut self) -> Option<(String, T)> {
        self.next_matching(|_| true)
    }

    /// Returns the next ready event that satisfies the provided predicate.
    pub fn next_matching<F>(&mut self, mut predicate: F) -> Option<(String, T)>
    where
        F: FnMut(&T) -> bool,
    {
        self.refresh_frontier();
        let index = self.frontier.iter().position(|id| {
            self.causal
                .get(id)
                .is_some_and(|node| self.causal.is_ready(id) && predicate(&node.payload))
        })?;
        let id = self.frontier.remove(index)?;
        let resolved = self.causal.resolve(&id);
        self.refresh_frontier();
        resolved
    }

    fn refresh_frontier(&mut self) {
        self.frontier.retain(|id| self.causal.is_ready(id));
        let ready = self.causal.ready_ids();
        for id in ready {
            if self.frontier.iter().any(|existing| existing == &id) {
                continue;
            }
            self.frontier.push_back(id);
        }
    }

    /// Verifies both the causal graph and the derived ready frontier.
    pub fn validate(&self) -> Result<(), CausalInvariantError> {
        self.causal.validate()?;
        let mut seen = BTreeSet::new();
        for id in &self.frontier {
            if !seen.insert(id.as_str()) {
                return Err(CausalInvariantError::DuplicateFrontierEvent(id.clone()));
            }
            if self.causal.get(id).is_none() {
                return Err(CausalInvariantError::UnknownFrontierEvent(id.clone()));
            }
            if !self.causal.is_ready(id) {
                return Err(CausalInvariantError::BlockedFrontierEvent(id.clone()));
            }
        }
        for id in self.causal.ready_ids() {
            if !seen.contains(id.as_str()) {
                return Err(CausalInvariantError::MissingFrontierEvent(id));
            }
        }
        Ok(())
    }
}

/// Mutable causal-set view that re-derives the engine frontier on drop.
pub struct CausalSetMut<'a, T> {
    engine: &'a mut TemporalLogicEngine<T>,
}

impl<T> CausalSetMut<'_, T> {
    /// Inserts a live event through the guarded causal set.
    pub fn insert(&mut self, id: impl Into<String>, payload: T) -> Result<(), CausalError> {
        self.engine.causal.insert(id, payload)
    }

    /// Adds a dependency through the guarded causal set.
    pub fn add_dependency(
        &mut self,
        earlier: impl AsRef<str>,
        later: impl AsRef<str>,
    ) -> Result<(), CausalError> {
        self.engine.causal.add_dependency(earlier, later)
    }

    /// Resolves a live event through the guarded causal set.
    pub fn resolve(&mut self, id: &str) -> Option<(String, T)> {
        self.engine.causal.resolve(id)
    }

    /// Compatibility alias for [`Self::resolve`].
    pub fn take(&mut self, id: &str) -> Option<(String, T)> {
        self.resolve(id)
    }

    /// Returns mutable access to a live payload without exposing graph indices.
    pub fn payload_mut(&mut self, id: &str) -> Option<&mut T> {
        self.engine
            .causal
            .nodes
            .get_mut(id)
            .map(|node| &mut node.payload)
    }
}

impl<T> Deref for CausalSetMut<'_, T> {
    type Target = CausalSet<T>;

    fn deref(&self) -> &Self::Target {
        &self.engine.causal
    }
}

impl<T> Drop for CausalSetMut<'_, T> {
    fn drop(&mut self) {
        self.engine.refresh_frontier();
    }
}

impl<T> Default for TemporalLogicEngine<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Section of the meaning sheaf associated with a narrative beat.
#[derive(Clone, Debug, PartialEq)]
pub struct MeaningSection {
    cover: Vec<String>,
    tags: Vec<String>,
    weight: f32,
    zspace_signature: Option<Vec<f64>>,
    min_similarity: f32,
}

impl MeaningSection {
    /// Creates a section supported on the provided cover.
    pub fn new(cover: Vec<String>, tags: Vec<String>, weight: f32) -> Self {
        let mut cover = cover
            .into_iter()
            .filter(|label| !label.trim().is_empty())
            .map(|label| label.trim().to_string())
            .collect::<Vec<_>>();
        if cover.is_empty() {
            cover.push("*".to_string());
        }
        let mut tags = tags
            .into_iter()
            .filter(|tag| !tag.trim().is_empty())
            .map(|tag| tag.trim().to_string())
            .collect::<Vec<_>>();
        tags.sort();
        tags.dedup();
        Self {
            cover,
            tags,
            weight: weight.max(0.0),
            zspace_signature: None,
            min_similarity: 0.0,
        }
    }

    /// Convenience constructor for a section concentrated on a single open.
    pub fn for_open(open: impl Into<String>, tags: Vec<String>, weight: f32) -> Self {
        Self::new(vec![open.into()], tags, weight)
    }

    /// Attaches a Z-space signature that must be matched before the section
    /// contributes to the collapsed sheaf.
    pub fn with_zspace_signature(mut self, signature: Vec<f64>, min_similarity: f32) -> Self {
        self.zspace_signature = Some(signature);
        self.min_similarity = min_similarity.clamp(0.0, 1.0);
        self
    }

    fn supports(&self, open: &str) -> bool {
        self.cover.iter().any(|label| label == "*" || label == open)
    }
}

fn signature_similarity(expected: &[f64], observed: &ZSpace) -> f64 {
    let expected_norm = expected
        .iter()
        .fold(0.0, |acc, value| acc + value * value)
        .sqrt();
    let observed_norm = observed.norm();
    if expected_norm <= f64::EPSILON || observed_norm <= f64::EPSILON {
        return 0.0;
    }
    let dot = expected
        .iter()
        .zip(observed.signature.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f64>();
    (dot / (expected_norm * observed_norm)).clamp(-1.0, 1.0)
}

/// Sheaf of meanings assigned to a causal event.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeaningSheaf {
    sections: Vec<MeaningSection>,
}

impl MeaningSheaf {
    /// Creates an empty sheaf.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a section to the sheaf.
    pub fn add_section(&mut self, section: MeaningSection) {
        self.sections.push(section);
    }

    /// Builder variant for chaining calls.
    pub fn with_section(mut self, section: MeaningSection) -> Self {
        self.add_section(section);
        self
    }

    /// Collapses the sheaf over `open`, returning the induced tags and their
    /// aggregated weights.
    pub fn collapse(
        &self,
        open: &str,
        threshold: f32,
        zspace: Option<&ZSpace>,
    ) -> (Vec<(String, f32)>, f32) {
        let mut aggregated: BTreeMap<String, f32> = BTreeMap::new();
        for section in &self.sections {
            if !section.supports(open) {
                continue;
            }
            let mut weight = section.weight;
            if let Some(expected) = section.zspace_signature.as_ref() {
                let observed = match zspace {
                    Some(observed) => observed,
                    None => continue,
                };
                let similarity = signature_similarity(expected, observed);
                if similarity < section.min_similarity as f64 {
                    continue;
                }
                weight *= similarity as f32;
            }
            if weight <= 0.0 {
                continue;
            }
            for tag in &section.tags {
                *aggregated.entry(tag.clone()).or_insert(0.0) += weight;
            }
        }
        let mut entries: Vec<(String, f32)> = aggregated
            .into_iter()
            .filter(|(_, weight)| *weight >= threshold)
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total = entries.iter().fold(0.0, |acc, (_, w)| acc + *w);
        (entries, total)
    }
}

/// Maintains the mapping between causal events and their meaning sheaves.
#[derive(Clone, Debug, Default)]
pub struct ToposLogicBridge {
    sheaves: BTreeMap<String, MeaningSheaf>,
}

impl ToposLogicBridge {
    /// Creates an empty bridge.
    pub fn new() -> Self {
        Self::default()
    }

    /// Associates a sheaf with the provided event identifier.
    pub fn attach(&mut self, event_id: impl Into<String>, sheaf: MeaningSheaf) {
        self.sheaves.insert(event_id.into(), sheaf);
    }

    /// Returns the collapsed tags for the event over the specified open.
    pub fn resolve(
        &self,
        event_id: &str,
        open: &str,
        threshold: f32,
        zspace: Option<&ZSpace>,
    ) -> (Vec<(String, f32)>, f32) {
        self.sheaves
            .get(event_id)
            .map(|sheaf| sheaf.collapse(open, threshold, zspace))
            .unwrap_or_default()
    }
}

/// Narrative beat emitted by the meta layer once causal constraints are
/// satisfied.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedNarrative {
    /// Identifier of the causal event that produced this beat.
    pub event_id: String,
    /// Channel that should receive the narrative.
    pub channel: String,
    /// Finalised tag list honouring the sheaf restrictions.
    pub tags: Vec<String>,
    /// Intensity computed from the pulse and sheaf weights.
    pub intensity: f32,
    /// Z-space signature observed when the beat was resolved.
    pub z_space: ZSpace,
}

/// Description of a narrative beat stored inside the causal set.
#[derive(Clone, Debug, PartialEq)]
pub struct NarrativeBeat {
    channel: Option<String>,
    context: String,
    base_tags: Vec<String>,
    intensity_scale: f32,
    floor: f32,
    sheaf_threshold: f32,
    zspace_gate: Option<ZSpaceGate>,
}

impl NarrativeBeat {
    /// Creates a beat bound to the optional channel and context.
    pub fn new(
        channel: Option<String>,
        context: impl Into<String>,
        base_tags: Vec<String>,
    ) -> Self {
        let mut tags = base_tags
            .into_iter()
            .filter(|tag| !tag.trim().is_empty())
            .map(|tag| tag.trim().to_string())
            .collect::<Vec<_>>();
        tags.sort();
        tags.dedup();
        Self {
            channel,
            context: context.into(),
            base_tags: tags,
            intensity_scale: 1.0,
            floor: 0.0,
            sheaf_threshold: 0.0,
            zspace_gate: None,
        }
    }

    /// Restricts the beat to a specific channel.
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    /// Sets the scaling applied to the detected Z magnitude.
    pub fn with_intensity_scale(mut self, scale: f32) -> Self {
        self.intensity_scale = scale.max(0.0);
        self
    }

    /// Sets the minimum intensity returned by the beat.
    pub fn with_floor(mut self, floor: f32) -> Self {
        self.floor = floor.max(0.0);
        self
    }

    /// Sets the sheaf threshold controlling which tags are included.
    pub fn with_sheaf_threshold(mut self, threshold: f32) -> Self {
        self.sheaf_threshold = threshold.max(0.0);
        self
    }

    /// Restricts the beat to pulses whose Z-space signature matches the
    /// provided prototype.
    pub fn with_zspace_gate(
        mut self,
        signature: Vec<f64>,
        min_similarity: f32,
        min_norm: f32,
    ) -> Self {
        self.zspace_gate = Some(ZSpaceGate::new(signature, min_similarity, min_norm));
        self
    }

    fn matches_channel(&self, channel: &str) -> bool {
        self.channel
            .as_deref()
            .map(|expected| expected == channel)
            .unwrap_or(true)
    }

    fn allows_zspace(&self, zspace: &ZSpace) -> bool {
        self.zspace_gate
            .as_ref()
            .map(|gate| gate.allows(zspace))
            .unwrap_or(true)
    }

    fn matches(&self, channel: &str, zspace: &ZSpace) -> bool {
        self.matches_channel(channel) && self.allows_zspace(zspace)
    }
}

/// Gate that restricts a narrative beat to Z-space signatures matching the
/// provided prototype.
#[derive(Clone, Debug, PartialEq)]
pub struct ZSpaceGate {
    signature: Vec<f64>,
    min_similarity: f32,
    min_norm: f32,
}

impl ZSpaceGate {
    /// Creates a new gate with the provided prototype and thresholds.
    pub fn new(signature: Vec<f64>, min_similarity: f32, min_norm: f32) -> Self {
        Self {
            signature,
            min_similarity: min_similarity.clamp(0.0, 1.0),
            min_norm: min_norm.max(0.0),
        }
    }

    /// Returns true when the observed signature satisfies the gate.
    pub fn allows(&self, observed: &ZSpace) -> bool {
        let similarity = signature_similarity(&self.signature, observed);
        let norm = observed.norm();
        similarity >= self.min_similarity as f64 && norm >= self.min_norm as f64
    }
}

/// High-level orchestrator binding together the causal and categorical layers.
#[derive(Clone, Debug, Default)]
pub struct MetaNarrativeLayer {
    engine: TemporalLogicEngine<NarrativeBeat>,
    bridge: ToposLogicBridge,
}

impl MetaNarrativeLayer {
    /// Creates an empty meta layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Injects a pre-built engine and bridge.
    pub fn from_parts(
        engine: TemporalLogicEngine<NarrativeBeat>,
        bridge: ToposLogicBridge,
    ) -> Self {
        Self { engine, bridge }
    }

    /// Returns a mutable reference to the underlying engine.
    pub fn engine_mut(&mut self) -> &mut TemporalLogicEngine<NarrativeBeat> {
        &mut self.engine
    }

    /// Returns a mutable reference to the categorical bridge.
    pub fn bridge_mut(&mut self) -> &mut ToposLogicBridge {
        &mut self.bridge
    }

    /// Resolves the next narrative beat compatible with the provided channel
    /// and pulse.
    pub fn next_with_pulse(
        &mut self,
        channel: &str,
        pulse: &MaxwellZPulse,
    ) -> Option<ResolvedNarrative> {
        let magnitude = pulse.magnitude() as f32;
        let observed_zspace = ZSpace::from(pulse);
        self.engine
            .next_matching(|beat| beat.matches(channel, &observed_zspace))
            .map(|(event_id, beat)| {
                let target_channel = beat.channel.clone().unwrap_or_else(|| channel.to_string());
                let base_intensity = (magnitude * beat.intensity_scale).max(beat.floor);
                let (sheaf_tags, sheaf_weight) = self.bridge.resolve(
                    &event_id,
                    &beat.context,
                    beat.sheaf_threshold,
                    Some(&observed_zspace),
                );
                let mut tags = beat.base_tags.clone();
                for (tag, _) in &sheaf_tags {
                    if !tags.iter().any(|existing| existing == tag) {
                        tags.push(tag.clone());
                    }
                }
                tags.sort();
                let intensity = if sheaf_weight > 0.0 {
                    base_intensity * (1.0 + sheaf_weight)
                } else {
                    base_intensity
                };
                ResolvedNarrative {
                    event_id,
                    channel: target_channel,
                    tags,
                    intensity,
                    z_space: observed_zspace.clone(),
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pulse(z: f64) -> MaxwellZPulse {
        MaxwellZPulse {
            blocks: 16,
            mean: 0.12,
            standard_error: 0.05,
            z_score: z,
            band_energy: (0.4, 0.3, 0.2),
            z_bias: 0.18,
        }
    }

    #[test]
    fn causal_set_detects_cycles() {
        let mut set = CausalSet::<()>::new();
        set.insert("a", ()).unwrap();
        set.insert("b", ()).unwrap();
        set.add_dependency("a", "b").unwrap();
        let err = set.add_dependency("b", "a").unwrap_err();
        assert_eq!(err, CausalError::CyclicDependency("b".into(), "a".into()));
    }

    #[test]
    fn engine_respects_partial_order() {
        let mut engine = TemporalLogicEngine::new();
        engine
            .insert_event(
                "intro",
                NarrativeBeat::new(Some("alpha".into()), "root", vec!["hello".into()]),
            )
            .unwrap();
        engine
            .insert_event(
                "bridge",
                NarrativeBeat::new(Some("alpha".into()), "root", vec!["world".into()]),
            )
            .unwrap();
        engine.add_dependency("intro", "bridge").unwrap();

        let first = engine.next_ready().expect("intro ready");
        assert_eq!(first.0, "intro");
        let second = engine.next_ready().expect("bridge ready");
        assert_eq!(second.0, "bridge");
        assert!(engine.next_ready().is_none());
    }

    #[test]
    fn engine_respects_dependencies_added_after_reverse_insertion() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("later", "later").unwrap();
        engine.insert_event("earlier", "earlier").unwrap();
        engine.add_dependency("earlier", "later").unwrap();

        assert_eq!(engine.next_ready(), Some(("earlier".into(), "earlier")));
        assert_eq!(engine.next_ready(), Some(("later".into(), "later")));
    }

    #[test]
    fn reverse_inserted_dense_dag_stays_valid_while_draining() {
        const EVENTS: usize = 32;
        let mut engine = TemporalLogicEngine::new();
        for index in (0..EVENTS).rev() {
            engine
                .insert_event(format!("event-{index:02}"), index)
                .unwrap();
        }
        for index in 1..EVENTS {
            engine
                .add_dependency(
                    format!("event-{:02}", index - 1),
                    format!("event-{index:02}"),
                )
                .unwrap();
            if index >= 3 {
                engine
                    .add_dependency(
                        format!("event-{:02}", index - 3),
                        format!("event-{index:02}"),
                    )
                    .unwrap();
            }
        }

        for expected in 0..EVENTS {
            assert_eq!(engine.validate(), Ok(()));
            let (id, payload) = engine.next_ready().expect("next DAG event");
            assert_eq!(id, format!("event-{expected:02}"));
            assert_eq!(payload, expected);
        }
        assert_eq!(engine.validate(), Ok(()));
        assert!(engine.next_ready().is_none());
    }

    #[test]
    fn causal_set_resolution_is_first_class_history() {
        let mut set = CausalSet::new();
        set.insert("child", "child").unwrap();
        set.insert("parent", "parent").unwrap();
        set.add_dependency("parent", "child").unwrap();
        assert!(!set.is_ready("child"));

        assert_eq!(set.resolve("parent"), Some(("parent".into(), "parent")));

        assert!(set.was_resolved("parent"));
        assert_eq!(set.resolved_len(), 1);
        assert!(set.is_ready("child"));
        assert_eq!(set.validate(), Ok(()));
    }

    #[test]
    fn resolved_event_identifiers_cannot_be_reused() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("once", 1).unwrap();
        assert_eq!(engine.next_ready(), Some(("once".into(), 1)));

        let error = engine.insert_event("once", 2).unwrap_err();

        assert_eq!(error, CausalError::ResolvedEvent("once".into()));
        assert_eq!(engine.pending(), 0);
        assert_eq!(engine.validate(), Ok(()));
    }

    #[test]
    fn resolved_history_can_parent_a_late_arriving_event() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("parent", 1).unwrap();
        assert_eq!(engine.next_ready(), Some(("parent".into(), 1)));
        engine.insert_event("child", 2).unwrap();

        engine.add_dependency("parent", "child").unwrap();
        assert_eq!(
            engine.add_dependency("child", "parent"),
            Err(CausalError::ResolvedEvent("parent".into()))
        );

        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.next_ready(), Some(("child".into(), 2)));
    }

    #[test]
    fn causal_mutation_guard_refreshes_blocked_frontier_entries() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("later", 2).unwrap();
        engine.insert_event("earlier", 1).unwrap();
        {
            let mut causal = engine.causal_mut();
            causal.add_dependency("earlier", "later").unwrap();
        }

        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.next_ready(), Some(("earlier".into(), 1)));
        assert_eq!(engine.next_ready(), Some(("later".into(), 2)));
    }

    #[test]
    fn causal_mutation_guard_updates_payload_without_exposing_graph_indices() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("ready", 1).unwrap();
        {
            let mut causal = engine.causal_mut();
            *causal.payload_mut("ready").unwrap() = 2;
            assert!(causal.payload_mut("missing").is_none());
        }

        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.next_ready(), Some(("ready".into(), 2)));
    }

    #[test]
    fn discarding_an_event_unblocks_its_descendants() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("child", 2).unwrap();
        engine.insert_event("parent", 1).unwrap();
        engine.add_dependency("parent", "child").unwrap();

        assert_eq!(engine.discard_event("parent"), Some(("parent".into(), 1)));

        assert!(engine.was_resolved("parent"));
        assert_eq!(engine.next_ready(), Some(("child".into(), 2)));
        assert_eq!(engine.validate(), Ok(()));
    }

    #[test]
    fn failed_cycle_insertion_leaves_graph_unchanged() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("a", 1).unwrap();
        engine.insert_event("b", 2).unwrap();
        engine.add_dependency("a", "b").unwrap();

        let error = engine.add_dependency("b", "a").unwrap_err();

        assert_eq!(error, CausalError::CyclicDependency("b".into(), "a".into()));
        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.next_ready(), Some(("a".into(), 1)));
        assert_eq!(engine.next_ready(), Some(("b".into(), 2)));
    }

    #[test]
    fn validation_detects_an_internally_corrupted_cycle() {
        let mut set = CausalSet::new();
        set.insert("a", 1).unwrap();
        set.insert("b", 2).unwrap();
        set.add_dependency("a", "b").unwrap();
        set.nodes.get_mut("a").unwrap().parents.insert("b".into());
        set.successors.get_mut("b").unwrap().insert("a".into());

        assert_eq!(set.validate(), Err(CausalInvariantError::CyclicGraph));
    }

    #[test]
    fn predicate_panic_does_not_consume_a_ready_event() {
        let mut engine = TemporalLogicEngine::new();
        engine.insert_event("ready", 1).unwrap();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.next_matching::<_>(|_| panic!("predicate failed"));
        }));

        assert!(panic.is_err());
        assert_eq!(engine.pending(), 1);
        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.next_ready(), Some(("ready".into(), 1)));
    }

    #[test]
    fn event_ids_and_nodes_are_inspectable_without_mutation() {
        let mut set = CausalSet::new();
        set.insert("parent", 1).unwrap();
        set.insert("child", 2).unwrap();
        set.add_dependency("parent", "child").unwrap();

        let node = set.get("child").expect("child node");
        assert_eq!(node.id(), "child");
        assert_eq!(node.payload(), &2);
        assert_eq!(node.parent_ids().collect::<Vec<_>>(), vec!["parent"]);
    }

    #[test]
    fn empty_event_ids_are_rejected_without_mutation() {
        let mut set = CausalSet::new();

        assert_eq!(set.insert("  ", 1), Err(CausalError::EmptyEventId));
        assert!(set.is_empty());
        assert_eq!(set.validate(), Ok(()));
    }

    #[test]
    fn meta_layer_resolves_sheaf_tags() {
        let mut engine = TemporalLogicEngine::new();
        engine
            .insert_event(
                "beat",
                NarrativeBeat::new(Some("alpha".into()), "canvas", vec!["base".into()])
                    .with_intensity_scale(0.5)
                    .with_sheaf_threshold(0.1),
            )
            .unwrap();
        let mut bridge = ToposLogicBridge::new();
        let signature = ZSpace::from(sample_pulse(3.2)).signature;
        let sheaf = MeaningSheaf::new().with_section(
            MeaningSection::for_open("canvas", vec!["extra".into(), "base".into()], 0.2)
                .with_zspace_signature(signature, 0.6),
        );
        bridge.attach("beat", sheaf);
        let mut layer = MetaNarrativeLayer::from_parts(engine, bridge);
        let resolved = layer
            .next_with_pulse("alpha", &sample_pulse(3.2))
            .expect("resolved");
        assert_eq!(resolved.channel, "alpha");
        assert!(resolved.tags.contains(&"base".into()));
        assert!(resolved.tags.contains(&"extra".into()));
        assert!(resolved.intensity > 0.0);
        assert!((resolved.z_space.signature[3] - 3.2).abs() < 1e-6);
    }

    #[test]
    fn meta_layer_respects_zspace_gate() {
        let mut engine = TemporalLogicEngine::new();
        engine
            .insert_event(
                "blocked",
                NarrativeBeat::new(Some("alpha".into()), "canvas", vec!["base".into()])
                    .with_zspace_gate(vec![0.0, 0.0, 0.0, -10.0], 0.9, 0.1),
            )
            .unwrap();
        engine
            .insert_event(
                "allowed",
                NarrativeBeat::new(Some("alpha".into()), "canvas", vec!["pass".into()])
                    .with_zspace_gate(vec![16.0, 0.12, 0.05, 3.2], 0.5, 0.1),
            )
            .unwrap();
        let mut layer = MetaNarrativeLayer::from_parts(engine, ToposLogicBridge::new());
        let resolved = layer
            .next_with_pulse("alpha", &sample_pulse(3.2))
            .expect("gate should admit second beat");
        assert_eq!(resolved.event_id, "allowed");
        assert!(resolved.tags.contains(&"pass".into()));
    }
}
