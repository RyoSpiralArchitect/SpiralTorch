//! Fixed-width, single-document causal contexts for external-tokenizer clients.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Clients must check this bound before eagerly converting document lengths.
pub const MAX_DOCUMENTS: usize = 1_048_576;
const MAX_WINDOWS: u64 = 1_048_576;
const MAX_RECIPE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    context_length: u32,
    document_lengths: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CausalDocumentWindow {
    pub document: u32,
    pub start: u32,
    pub length: u32,
}

/// No concatenation, padding, tokenizer, EOS insertion or content authentication.
/// Each full context is disjoint; its first token is context, not a target.
/// Incomplete tails are dropped per document, never borrowed from the next one.
#[derive(Clone, Debug, Serialize)]
pub struct CausalDocumentWindows {
    schema: &'static str,
    policy: &'static str,
    context_length: u32,
    document_lengths: Vec<u32>,
    windows: Vec<CausalDocumentWindow>,
    total_tokens: u64,
    dropped_tokens: u64,
    target_tokens: u64,
}

impl CausalDocumentWindows {
    pub fn from_lengths(document_lengths: Vec<u32>, context_length: u32) -> Result<Self, String> {
        if context_length < 2 {
            return Err("causal contexts need at least two tokens".into());
        }
        if document_lengths.len() > MAX_DOCUMENTS {
            return Err("document count exceeds budget".into());
        }
        let count: u64 = document_lengths
            .iter()
            .map(|n| u64::from(n / context_length))
            .sum();
        if count > MAX_WINDOWS {
            return Err("window count exceeds budget".into());
        }
        let total_tokens = document_lengths.iter().map(|&n| u64::from(n)).sum();
        let dropped_tokens = document_lengths
            .iter()
            .map(|&n| u64::from(n % context_length))
            .sum();
        let mut windows = Vec::new();
        windows
            .try_reserve_exact(count as usize)
            .map_err(|_| "window allocation failed")?;
        for (document, &length) in document_lengths.iter().enumerate() {
            for i in 0..length / context_length {
                windows.push(CausalDocumentWindow {
                    document: document as u32,
                    start: i * context_length,
                    length: context_length,
                });
            }
        }
        Ok(Self {
            schema: "spiraltorch.causal_document_windows.v1",
            policy: "disjoint_full_contexts_drop_tail_per_document.v1",
            context_length,
            document_lengths,
            windows,
            total_tokens,
            dropped_tokens,
            target_tokens: count * u64::from(context_length - 1),
        })
    }

    pub fn create_json(payload: &[u8]) -> Result<Self, String> {
        if payload.len() > MAX_RECIPE_BYTES {
            return Err("window recipe exceeds byte budget".into());
        }
        let recipe: Recipe = serde_json::from_slice(payload).map_err(|e| e.to_string())?;
        Self::from_lengths(recipe.document_lengths, recipe.context_length)
    }

    pub fn windows(&self) -> &[CausalDocumentWindow] {
        &self.windows
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }

    /// Identity of the window plan only, NOT an identity of token/document content.
    pub fn plan_id(&self) -> Result<String, String> {
        Ok(format!(
            "sha256:{:x}",
            Sha256::digest(self.to_json()?.as_bytes())
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_never_borrow_from_short_documents_or_tails() {
        let plan = CausalDocumentWindows::from_lengths(vec![0, 3, 5, 12], 4).unwrap();
        assert_eq!(
            plan.windows
                .iter()
                .map(|w| (w.document, w.start, w.length))
                .collect::<Vec<_>>(),
            vec![(2, 0, 4), (3, 0, 4), (3, 4, 4), (3, 8, 4)]
        );
        assert_eq!(
            (plan.total_tokens, plan.dropped_tokens, plan.target_tokens),
            (20, 4, 12)
        );
        assert!(CausalDocumentWindows::from_lengths(vec![3, 3], 4)
            .unwrap()
            .windows
            .is_empty());
    }

    #[test]
    fn lengths_and_boundaries_are_bound_without_claiming_content_identity() {
        let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../tests/fixtures/causal_document_windows.fixture.json"
        ))
        .unwrap();
        let shared =
            CausalDocumentWindows::create_json(&serde_json::to_vec(&fixture["recipe"]).unwrap())
                .unwrap();
        assert_eq!(shared.plan_id().unwrap(), fixture["plan_id"]);
        let a = CausalDocumentWindows::from_lengths(vec![4, 4], 4).unwrap();
        let b = CausalDocumentWindows::from_lengths(vec![8], 4).unwrap();
        assert_ne!(a.plan_id().unwrap(), b.plan_id().unwrap());
        let c =
            CausalDocumentWindows::create_json(br#"{"document_lengths":[4,4],"context_length":4}"#)
                .unwrap();
        assert_eq!(a.plan_id().unwrap(), c.plan_id().unwrap());
    }

    #[test]
    fn rejects_invalid_recipes_before_window_allocation() {
        assert!(CausalDocumentWindows::from_lengths(vec![4], 1).is_err());
        assert!(CausalDocumentWindows::from_lengths(vec![u32::MAX], 2).is_err());
        assert!(CausalDocumentWindows::from_lengths(vec![0; MAX_DOCUMENTS + 1], 2).is_err());
        for value in [
            r#"{"document_lengths":[-1],"context_length":4}"#,
            r#"{"document_lengths":[4.5],"context_length":4}"#,
            r#"{"document_lengths":[4],"context_length":true}"#,
            r#"{"document_lengths":[4],"context_length":4,"padding":true}"#,
        ] {
            assert!(CausalDocumentWindows::create_json(value.as_bytes()).is_err());
        }
        let boundary = CausalDocumentWindows::from_lengths(vec![u32::MAX], u32::MAX).unwrap();
        assert_eq!(boundary.windows[0].start, 0);
        assert_eq!(boundary.target_tokens, u64::from(u32::MAX - 1));
    }
}
