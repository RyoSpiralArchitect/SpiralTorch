//! Bounded transport for pinned, explicitly labeled evaluation JSON.
use super::{CausalLmResult, NativeCausalLm, MAX_DOCUMENTS};
use serde::{
    de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use std::{cell::Cell, fmt, marker::PhantomData};

pub struct CausalLmLabeledAnswerCorpus {
    pub token_documents: Vec<Vec<usize>>,
    pub labels: Vec<Vec<i64>>,
}

impl CausalLmLabeledAnswerCorpus {
    pub const MAX_JSON_BYTES: usize = 256 * 1024 * 1024;

    /// Apply the same document and token limits as native corpus evaluation.
    /// Model-specific positions, vocabulary, and work budgets are checked at scoring.
    pub fn from_json(bytes: &[u8], owners: usize) -> CausalLmResult<Self> {
        if bytes.len() > Self::MAX_JSON_BYTES {
            return Err("labeled answer corpus exceeds JSON byte budget".into());
        }
        Ok(parse(
            bytes,
            NativeCausalLm::evaluation_token_limit(owners)?,
            MAX_DOCUMENTS,
        )?)
    }
}

fn parse(
    bytes: &[u8],
    tokens: usize,
    documents: usize,
) -> Result<CausalLmLabeledAnswerCorpus, serde_json::Error> {
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let result = decoder.deserialize_map(Corpus { tokens, documents })?;
    decoder.end()?;
    Ok(result)
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Field {
    TokenDocuments,
    Labels,
}

struct Corpus {
    tokens: usize,
    documents: usize,
}

impl<'de> Visitor<'de> for Corpus {
    type Value = CausalLmLabeledAnswerCorpus;

    fn expecting(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("an object with token_documents and labels")
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let (mut token_documents, mut labels) = (None, None);
        while let Some(key) = map.next_key()? {
            match key {
                Field::TokenDocuments if token_documents.is_none() => {
                    token_documents = Some(self.read_rows(&mut map)?)
                }
                Field::Labels if labels.is_none() => labels = Some(self.read_rows(&mut map)?),
                Field::TokenDocuments => return Err(Error::duplicate_field("token_documents")),
                Field::Labels => return Err(Error::duplicate_field("labels")),
            }
        }
        let token_documents =
            token_documents.ok_or_else(|| Error::missing_field("token_documents"))?;
        let labels = labels.ok_or_else(|| Error::missing_field("labels"))?;
        if token_documents.len() != labels.len()
            || token_documents
                .iter()
                .zip(&labels)
                .any(|(doc, row)| doc.len() != row.len())
        {
            return Err(Error::custom("labels must match token document shape"));
        }
        Ok(CausalLmLabeledAnswerCorpus {
            token_documents,
            labels,
        })
    }
}

impl Corpus {
    fn read_rows<'de, T: Copy + Deserialize<'de>, M: MapAccess<'de>>(
        &self,
        map: &mut M,
    ) -> Result<Vec<Vec<T>>, M::Error> {
        let documents = Cell::new(self.documents);
        let tokens = Cell::new(self.tokens);
        map.next_value_seed(LimitedList {
            item: LimitedList {
                item: Scalar(PhantomData),
                remaining: &tokens,
                exceeded: "labeled answer token budget exceeded",
            },
            remaining: &documents,
            exceeded: "labeled answer document budget exceeded",
        })
    }
}

#[derive(Clone, Copy)]
struct Scalar<T>(PhantomData<T>);

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for Scalar<T> {
    type Value = T;

    fn deserialize<D: Deserializer<'de>>(self, input: D) -> Result<T, D::Error> {
        T::deserialize(input)
    }
}

#[derive(Clone, Copy)]
struct LimitedList<'a, S> {
    item: S,
    remaining: &'a Cell<usize>,
    exceeded: &'static str,
}

struct Charged<'a, S>(LimitedList<'a, S>);

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for Charged<'_, S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, input: D) -> Result<Self::Value, D::Error> {
        let budget = self.0.remaining;
        budget.set(
            budget
                .get()
                .checked_sub(1)
                .ok_or_else(|| Error::custom(self.0.exceeded))?,
        );
        self.0.item.deserialize(input)
    }
}

impl<'de, S: Copy + DeserializeSeed<'de>> DeserializeSeed<'de> for LimitedList<'_, S> {
    type Value = Vec<S::Value>;

    fn deserialize<D: Deserializer<'de>>(self, input: D) -> Result<Self::Value, D::Error> {
        input.deserialize_seq(self)
    }
}

impl<'de, S: Copy + DeserializeSeed<'de>> Visitor<'de> for LimitedList<'_, S> {
    type Value = Vec<S::Value>;

    fn expecting(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str("a bounded array")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Charged(self))? {
            items.push(item);
        }
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labeled_corpus_accepts_both_orders_at_exact_limits() {
        for raw in [
            r#"{"token_documents":[[2,4],[],[3]],"labels":[[-100,4],[],[3]]}"#,
            r#"{"labels":[[-100,4],[],[3]],"token_documents":[[2,4],[],[3]]}"#,
        ] {
            let corpus = parse(raw.as_bytes(), 3, 3).unwrap();
            assert_eq!(corpus.token_documents, [vec![2, 4], vec![], vec![3]]);
            assert_eq!(corpus.labels, [vec![-100, 4], vec![], vec![3]]);
        }
    }

    #[test]
    fn labeled_corpus_rejects_extra_values_before_decoding_them() {
        for (raw, budget) in [
            (r#"{"token_documents":[[1],[false]]}"#, "token"),
            (r#"{"labels":[[1],[false]]}"#, "token"),
            (r#"{"token_documents":[[],[],{}]}"#, "document"),
            (r#"{"labels":[[],[],{}]}"#, "document"),
        ] {
            let err = parse(raw.as_bytes(), 1, 2).err().unwrap();
            assert!(err
                .to_string()
                .contains(&format!("{budget} budget exceeded")));
        }
    }

    #[test]
    fn labeled_corpus_rejects_ambiguous_or_unpaired_json() {
        for raw in [
            r#"{"token_documents":[[true]],"labels":[[1]]}"#,
            r#"{"token_documents":[[-1]],"labels":[[1]]}"#,
            r#"{"token_documents":[[1]],"labels":[[1.0]]}"#,
            r#"{"token_documents":[[1]],"labels":[[]]}"#,
            r#"{"token_documents":[[1]],"labels":[]}"#,
            r#"{"token_documents":[],"token_documents":[],"labels":[]}"#,
            r#"{"token_documents":[],"labels":[],"extra":[]}"#,
            r#"{"token_documents":[]}"#,
            r#"{"token_documents":[],"labels":[]} null"#,
        ] {
            assert!(parse(raw.as_bytes(), 3, 3).is_err(), "{raw}");
        }
    }
}
