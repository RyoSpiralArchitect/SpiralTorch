// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::Arc;

#[derive(Clone, Debug)]
struct BankEntry<T> {
    id: Arc<str>,
    value: T,
}

/// Generic registry storing values keyed by identifier in insertion order.
#[derive(Clone, Debug)]
pub struct GaugeBank<T> {
    entries: Vec<BankEntry<T>>,
}

impl<T> Default for GaugeBank<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<T> GaugeBank<T> {
    /// Creates an empty bank.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a value with the provided identifier.
    pub fn register(&mut self, id: impl Into<String>, value: T) -> bool {
        let id: Arc<str> = Arc::from(id.into());
        if self
            .entries
            .iter()
            .any(|entry| entry.id.as_ref() == id.as_ref())
        {
            return false;
        }
        self.entries.push(BankEntry { id, value });
        true
    }

    /// Builder-style registration that returns `self` for chaining.
    pub fn with_registered(mut self, id: impl Into<String>, value: T) -> Self {
        let _ = self.register(id, value);
        self
    }

    /// Returns a reference to the value identified by `id`.
    pub fn get(&self, id: &str) -> Option<&T> {
        self.entries
            .iter()
            .find(|entry| entry.id.as_ref() == id)
            .map(|entry| &entry.value)
    }

    /// Returns a mutable reference to the value identified by `id`.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut T> {
        self.entries
            .iter_mut()
            .find(|entry| entry.id.as_ref() == id)
            .map(|entry| &mut entry.value)
    }

    /// Removes a value from the bank and returns it if present.
    pub fn remove(&mut self, id: &str) -> Option<T> {
        if let Some(idx) = self
            .entries
            .iter()
            .position(|entry| entry.id.as_ref() == id)
        {
            Some(self.entries.remove(idx).value)
        } else {
            None
        }
    }

    /// Iterates over identifiers and values in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &T)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.id.as_ref(), &entry.value))
    }

    /// Iterates over identifiers and mutable values in insertion order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut T)> + '_ {
        self.entries
            .iter_mut()
            .map(|entry| (entry.id.as_ref(), &mut entry.value))
    }

    /// Provides access to the underlying identifier handles and values.
    pub fn entries(&self) -> impl Iterator<Item = (&Arc<str>, &T)> + '_ {
        self.entries.iter().map(|entry| (&entry.id, &entry.value))
    }

    /// Returns the identifiers stored in the bank.
    pub fn ids(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.iter().map(|entry| entry.id.as_ref())
    }

    /// Returns the number of registered values.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no values are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clones the stored values into a vector preserving insertion order.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.entries
            .iter()
            .map(|entry| entry.value.clone())
            .collect()
    }

    /// Consumes the bank and returns the stored values in insertion order.
    pub fn into_vec(self) -> Vec<T> {
        self.entries.into_iter().map(|entry| entry.value).collect()
    }

    /// Consumes the bank and returns identifier-value pairs in insertion order.
    pub fn into_entries(self) -> Vec<(Arc<str>, T)> {
        self.entries
            .into_iter()
            .map(|entry| (entry.id, entry.value))
            .collect()
    }
}

impl<T> IntoIterator for GaugeBank<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}
