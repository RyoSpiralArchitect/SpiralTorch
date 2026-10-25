//! Borrowed wire views preserve raw evidence without a second value tree.
use super::*;
use serde::Serialize;
use std::io::{self, Write};

fn decimal_counter<S: serde::Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_str(value)
}

#[derive(Serialize)]
struct ForwardWire<'a> {
    #[serde(serialize_with = "decimal_counter")]
    input_generation: u64,
    #[serde(serialize_with = "decimal_counter")]
    submitted_forward: u64,
    #[serde(serialize_with = "decimal_counter")]
    parameter_revision: u64,
    #[serde(serialize_with = "decimal_counter")]
    submitted_backwards_at_capture: u64,
    parameter_shapes: &'a [Vec<usize>],
    parameter_bits: &'a [Vec<u32>],
    activation_shapes: &'a [Vec<usize>],
    activation_bits: &'a [Vec<u32>],
    forward_flags: &'a [u32],
    prediction_flags: u32,
}

#[derive(Serialize)]
struct NamedWire<'a> {
    name: &'a str,
    model_slots: &'a [usize],
    diagnostic: ForwardWire<'a>,
}

#[derive(Serialize)]
struct ModelWire<'a> {
    schema: &'static str,
    value_encoding: &'static str,
    counter_encoding: &'static str,
    diagnostic_only: bool,
    release_ready: bool,
    #[serde(serialize_with = "decimal_counter")]
    submitted_forward: u64,
    #[serde(serialize_with = "decimal_counter")]
    parameter_revision: u64,
    captured_bytes: usize,
    model_parameter_shapes: &'a [Vec<usize>],
    direct_model_slots: &'a [usize],
    graphs: Vec<NamedWire<'a>>,
}

struct BoundedJson {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedJson {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        let end = self
            .bytes
            .len()
            .checked_add(input.len())
            .filter(|&end| end <= self.limit)
            .ok_or_else(|| io::Error::other("graph diagnostic JSON byte budget"))?;
        if end > self.bytes.capacity() {
            let capacity = self
                .bytes
                .capacity()
                .max(1024)
                .saturating_mul(2)
                .min(self.limit)
                .max(end);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .map_err(|_| io::Error::other("graph diagnostic JSON allocation"))?;
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ByteDecoderGraphDiagnostic {
    /// Serialize raw u32 bits and guards without converting them to floats.
    /// u64 counters use decimal strings, preserving precision in browser JSON.
    /// No invalid-value filtering or numerical acceptance is performed.
    pub fn to_json(&self) -> Result<String, InferenceError> {
        self.to_json_with_limit(BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_JSON_BYTES)
    }

    /// Bounds encoded bytes while streaming; an oversized result is never
    /// returned. This is not a bound on peak host memory or Python objects.
    pub fn to_json_with_limit(&self, max_bytes: usize) -> Result<String, InferenceError> {
        if max_bytes == 0 || max_bytes > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_JSON_BYTES {
            return Err(InferenceError::ByteDecoder(
                "graph diagnostic JSON byte budget",
            ));
        }
        if self.graphs.len() > BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_GRAPHS {
            return Err(InferenceError::ByteDecoder("graph diagnostic graph limit"));
        }
        let graphs = self
            .graphs
            .iter()
            .map(|graph| {
                let d = &graph.diagnostic;
                NamedWire {
                    name: &graph.name,
                    model_slots: &graph.model_slots,
                    diagnostic: ForwardWire {
                        input_generation: d.input_generation,
                        submitted_forward: d.submitted_forward,
                        parameter_revision: d.parameter_revision,
                        submitted_backwards_at_capture: d.submitted_backwards_at_capture,
                        parameter_shapes: &d.parameter_shapes,
                        parameter_bits: &d.parameter_bits,
                        activation_shapes: &d.activation_shapes,
                        activation_bits: &d.activation_bits,
                        forward_flags: &d.forward_flags,
                        prediction_flags: d.prediction_flags,
                    },
                }
            })
            .collect();
        let wire = ModelWire {
            schema: "spiraltorch.byte_decoder_graph_diagnostic.v1",
            value_encoding: "f32_raw_u32_bits",
            counter_encoding: "u64_decimal_strings",
            diagnostic_only: true,
            release_ready: false,
            submitted_forward: self.submitted_forward,
            parameter_revision: self.parameter_revision,
            captured_bytes: self.captured_bytes,
            model_parameter_shapes: &self.model_parameter_shapes,
            direct_model_slots: &self.direct_model_slots,
            graphs,
        };
        let mut writer = BoundedJson {
            bytes: Vec::new(),
            limit: max_bytes,
        };
        serde_json::to_writer(&mut writer, &wire)?;
        String::from_utf8(writer.bytes)
            .map_err(|_| InferenceError::ByteDecoder("graph diagnostic JSON encoding"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ByteDecoderGraphDiagnostic {
        // Synthetic serializer evidence only, including invalid numerical bits.
        ByteDecoderGraphDiagnostic {
            submitted_forward: u64::MAX,
            parameter_revision: 1,
            captured_bytes: 48,
            model_parameter_shapes: vec![vec![2], vec![1]],
            direct_model_slots: vec![1],
            graphs: vec![ByteDecoderNamedGraphDiagnostic {
                name: "fixture.head".into(),
                model_slots: vec![0],
                diagnostic: GraphForwardDiagnostic {
                    input_generation: 8,
                    submitted_forward: 9,
                    parameter_revision: 2,
                    submitted_backwards_at_capture: 0,
                    parameter_shapes: vec![vec![2]],
                    parameter_bits: vec![vec![0x7fc0_0042, 0x8000_0000]],
                    activation_shapes: vec![vec![1, 2], vec![1, 2]],
                    activation_bits: vec![vec![0x7f80_0000, 0xff80_0000], vec![0x3f80_0000, 0]],
                    forward_flags: vec![1, 2, 0, 0, 0],
                    prediction_flags: 3,
                },
            }],
        }
    }

    #[test]
    fn diagnostic_json_preserves_bits_guards_and_owner_graph_revisions() {
        let value: serde_json::Value = serde_json::from_str(&fixture().to_json().unwrap()).unwrap();
        assert_eq!(
            value["schema"],
            "spiraltorch.byte_decoder_graph_diagnostic.v1"
        );
        assert_eq!(value["value_encoding"], "f32_raw_u32_bits");
        assert_eq!(value["counter_encoding"], "u64_decimal_strings");
        assert_eq!(value["submitted_forward"], u64::MAX.to_string());
        assert_eq!(value["parameter_revision"], "1");
        assert_eq!(value["direct_model_slots"], serde_json::json!([1]));
        assert_eq!(value["release_ready"], false);
        let graph = &value["graphs"][0]["diagnostic"];
        assert_eq!(graph["parameter_revision"], "2");
        assert_eq!(graph["input_generation"], "8");
        assert_eq!(graph["submitted_forward"], "9");
        assert_eq!(graph["submitted_backwards_at_capture"], "0");
        assert_eq!(
            graph["parameter_bits"],
            serde_json::json!([[0x7fc0_0042u32, 0x8000_0000u32]])
        );
        assert_eq!(
            graph["activation_bits"][0],
            serde_json::json!([0x7f80_0000u32, 0xff80_0000u32])
        );
        assert_eq!(graph["prediction_flags"], 3);
        assert_eq!(graph["forward_flags"], serde_json::json!([1, 2, 0, 0, 0]));
    }

    #[test]
    fn diagnostic_json_limit_is_inclusive_and_never_returns_partial_output() {
        let diagnostic = fixture();
        let text = diagnostic.to_json().unwrap();
        assert_eq!(diagnostic.to_json_with_limit(text.len()).unwrap(), text);
        assert!(diagnostic.to_json_with_limit(text.len() - 1).is_err());
        assert!(diagnostic.to_json_with_limit(0).is_err());
        assert!(diagnostic
            .to_json_with_limit(BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_JSON_BYTES + 1)
            .is_err());
    }

    #[test]
    fn diagnostic_json_writer_rejects_overflow_before_appending() {
        let mut writer = BoundedJson {
            bytes: Vec::new(),
            limit: 4,
        };
        writer.write_all(b"12").unwrap();
        assert!(writer.write_all(b"345").is_err());
        assert_eq!(writer.bytes, b"12");
        writer.write_all(b"34").unwrap();
        assert_eq!(writer.bytes, b"1234");
        assert!(writer.write_all(b"5").is_err());
    }

    #[test]
    fn diagnostic_json_rejects_excess_graph_metadata() {
        let mut diagnostic = fixture();
        diagnostic.graphs =
            vec![diagnostic.graphs[0].clone(); BYTE_DECODER_GRAPH_DIAGNOSTIC_MAX_GRAPHS + 1];
        assert!(diagnostic.to_json().is_err());
    }
}
