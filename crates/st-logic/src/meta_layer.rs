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

use st_core::maxwell::MaxwellZPulse;
use thiserror::Error;

/// Error emitted when a causal relation cannot be established.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CausalError {
    /// The provided identifier is already registered inside the set.
    #[error("causal event `{0}` is already registered")]
    DuplicateEvent(String),
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

    /// Returns true when all parent identifiers appear inside `emitted`.
    fn is_ready(&self, emitted: &BTreeSet<String>) -> bool {
        self.parents.iter().all(|parent| emitted.contains(parent))
    }
}

/// Partially ordered set describing the causal relations between narrative
/// beats.
#[derive(Clone, Debug)]
pub struct CausalSet<T> {
    nodes: BTreeMap<String, CausalNode<T>>,
    successors: BTreeMap<String, BTreeSet<String>>,
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

    /// Inserts a new event into the causal set.
    pub fn insert(&mut self, id: impl Into<String>, payload: T) -> Result<(), CausalError> {
        let id = id.into();
        if self.nodes.contains_key(&id) {
            return Err(CausalError::DuplicateEvent(id));
        }
        self.successors.entry(id.clone()).or_default();
        self.nodes.insert(id.clone(), CausalNode::new(id, payload));
        Ok(())
    }

    /// Registers that `earlier` must precede `later`.
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
        if !self.nodes.contains_key(earlier) {
            return Err(CausalError::UnknownEvent(earlier.to_string()));
        }
        if !self.nodes.contains_key(later) {
            return Err(CausalError::UnknownEvent(later.to_string()));
        }
        if self.would_create_cycle(earlier, later) {
            return Err(CausalError::CyclicDependency(
                earlier.to_string(),
                later.to_string(),
            ));
        }
        let later_node = self
            .nodes
            .get_mut(later)
            .expect("later node exists after earlier validation");
        later_node.parents.insert(earlier.to_string());
        self.successors
            .entry(earlier.to_string())
            .or_default()
            .insert(later.to_string());
        Ok(())
    }

    /// Returns a reference to the stored node.
    pub fn get(&self, id: &str) -> Option<&CausalNode<T>> {
        self.nodes.get(id)
    }

    /// Removes an event from the causal set, returning its payload.
    pub fn take(&mut self, id: &str) -> Option<(String, T)> {
        let node = self.nodes.remove(id)?;
        self.successors.remove(id);
        for successors in self.successors.values_mut() {
            successors.remove(id);
        }
        Some((node.id, node.payload))
    }

    fn ready_ids(&self, emitted: &BTreeSet<String>) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(id, node)| !emitted.contains(id.as_str()) && node.is_ready(emitted))
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
        }
    }
}

/// Advances a [`CausalSet`] according to temporal logic rules.
#[derive(Clone, Debug)]
pub struct TemporalLogicEngine<T> {
    causal: CausalSet<T>,
    emitted: BTreeSet<String>,
    frontier: VecDeque<String>,
}

impl<T> TemporalLogicEngine<T> {
    /// Builds an engine from an existing causal set.
    pub fn from_causal_set(causal: CausalSet<T>) -> Self {
        let mut engine = Self {
            frontier: VecDeque::new(),
            emitted: BTreeSet::new(),
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

    /// Returns a mutable reference to the underlying causal set.
    pub fn causal_mut(&mut self) -> &mut CausalSet<T> {
        &mut self.causal
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
        self.retain_frontier();
        self.refresh_frontier();
        Ok(())
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
        for idx in 0..self.frontier.len() {
            if let Some(id) = self.frontier.get(idx).cloned() {
                if let Some(node) = self.causal.get(&id) {
                    if !predicate(&node.payload) {
                        continue;
                    }
                    let id = self.frontier.remove(idx).unwrap_or(id);
                    let (id, payload) = self.causal.take(&id).expect("node exists at removal time");
                    self.emitted.insert(id.clone());
                    self.refresh_frontier();
                    return Some((id, payload));
                } else {
                    // Remove stale identifiers.
                    self.frontier.remove(idx);
                }
            }
        }
        None
    }

    fn refresh_frontier(&mut self) {
        let ready = self.causal.ready_ids(&self.emitted);
        for id in ready {
            if self.emitted.contains(&id) {
                continue;
            }
            if self.frontier.iter().any(|existing| existing == &id) {
                continue;
            }
            self.frontier.push_back(id);
        }
        self.retain_frontier();
    }

    fn retain_frontier(&mut self) {
        self.frontier
            .retain(|id| self.causal.get(id.as_str()).is_some() && !self.emitted.contains(id));
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
        }
    }

    /// Convenience constructor for a section concentrated on a single open.
    pub fn for_open(open: impl Into<String>, tags: Vec<String>, weight: f32) -> Self {
        Self::new(vec![open.into()], tags, weight)
    }

    fn supports(&self, open: &str) -> bool {
        self.cover.iter().any(|label| label == "*" || label == open)
    }
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
    pub fn collapse(&self, open: &str, threshold: f32) -> (Vec<(String, f32)>, f32) {
        let mut aggregated: BTreeMap<String, f32> = BTreeMap::new();
        for section in &self.sections {
            if !section.supports(open) {
                continue;
            }
            for tag in &section.tags {
                *aggregated.entry(tag.clone()).or_insert(0.0) += section.weight;
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
    pub fn resolve(&self, event_id: &str, open: &str, threshold: f32) -> (Vec<(String, f32)>, f32) {
        self.sheaves
            .get(event_id)
            .map(|sheaf| sheaf.collapse(open, threshold))
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

    fn matches_channel(&self, channel: &str) -> bool {
        self.channel
            .as_deref()
            .map(|expected| expected == channel)
            .unwrap_or(true)
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
        self.engine
            .next_matching(|beat| beat.matches_channel(channel))
            .map(|(event_id, beat)| {
                let target_channel = beat.channel.clone().unwrap_or_else(|| channel.to_string());
                let base_intensity = (magnitude * beat.intensity_scale).max(beat.floor);
                let (sheaf_tags, sheaf_weight) =
                    self.bridge
                        .resolve(&event_id, &beat.context, beat.sheaf_threshold);
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
        let sheaf = MeaningSheaf::new().with_section(MeaningSection::for_open(
            "canvas",
            vec!["extra".into(), "base".into()],
            0.2,
        ));
        bridge.attach("beat", sheaf);
        let mut layer = MetaNarrativeLayer::from_parts(engine, bridge);
        let resolved = layer
            .next_with_pulse("alpha", &sample_pulse(3.2))
            .expect("resolved");
        assert_eq!(resolved.channel, "alpha");
        assert!(resolved.tags.contains(&"base".into()));
        assert!(resolved.tags.contains(&"extra".into()));
        assert!(resolved.intensity > 0.0);
    }
}
