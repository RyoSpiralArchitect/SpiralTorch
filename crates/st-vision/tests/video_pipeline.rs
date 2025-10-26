// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)

use st_tensor::{PureResult, Tensor};
use st_vision::{
    DecodedFrame, VideoDecoder, VideoPipeline, VideoPipelineConfig, VideoPipelineOutput,
};

struct SyntheticDecoder {
    frames: Vec<DecodedFrame>,
    index: usize,
}

impl SyntheticDecoder {
    fn new(frames: Vec<DecodedFrame>) -> Self {
        Self { frames, index: 0 }
    }
}

impl VideoDecoder for SyntheticDecoder {
    fn next_frame(&mut self) -> PureResult<Option<DecodedFrame>> {
        if self.index >= self.frames.len() {
            Ok(None)
        } else {
            let frame = self.frames[self.index].clone();
            self.index += 1;
            Ok(Some(frame))
        }
    }
}

fn tensor(rows: usize, cols: usize, values: &[f32]) -> Tensor {
    Tensor::from_vec(rows, cols, values.to_vec()).expect("tensor")
}

fn synthetic_frames() -> Vec<DecodedFrame> {
    vec![
        DecodedFrame {
            timestamp: 0.0,
            tensor: tensor(2, 2, &[0.0, 0.05, 0.1, 0.2]),
        },
        DecodedFrame {
            timestamp: 1.0 / 30.0,
            tensor: tensor(2, 2, &[0.2, 0.3, 0.3, 0.5]),
        },
        DecodedFrame {
            timestamp: 2.0 / 30.0,
            tensor: tensor(2, 2, &[0.4, 0.45, 0.6, 0.7]),
        },
    ]
}

#[test]
fn pipeline_emits_temporal_z_dynamics() {
    let decoder = SyntheticDecoder::new(synthetic_frames());
    let mut pipeline = VideoPipeline::new(
        decoder,
        VideoPipelineConfig {
            temporal_alpha: 0.5,
            resonance_decay: 0.3,
            motion_gain: 1.0,
            digest_window: 2,
            quiescence_threshold: 0.05,
        },
    );

    let mut outputs: Vec<VideoPipelineOutput> = Vec::new();
    while let Some(output) = pipeline.next().expect("frame") {
        outputs.push(output);
    }

    assert_eq!(outputs.len(), 3);
    for (idx, output) in outputs.iter().enumerate() {
        assert_eq!(output.stream.volume.depth(), 2);
        assert_eq!(output.stream.volume.height(), 2);
        assert_eq!(output.stream.volume.width(), 2);
        assert_eq!(output.z_dynamics.smoothed_weights.len(), 2);
        assert_eq!(output.z_dynamics.per_depth_energy.len(), 2);
        assert_eq!(output.z_dynamics.spectral_response.len(), 2);
        assert!(output
            .atlas_frame
            .metrics
            .iter()
            .any(|metric| metric.name == "z.motion_energy"));
        assert!(output
            .atlas_frame
            .notes
            .iter()
            .any(|note| note == "video.pipeline.z_dynamics"));
        let snapshot = output.stream.chrono_snapshot.as_ref().expect("snapshot");
        assert_eq!(snapshot.summary().frames, idx + 1);
        if idx > 0 {
            assert!(snapshot.dt() > 0.0);
        }
        let (rows, cols) = output.resonance_envelope.shape();
        assert_eq!(rows, 2);
        assert_eq!(cols, 4);
        assert!(output.temporal_digest.frames >= idx + 1);
        assert!(output.window_digest.frames >= 1);
    }

    let motion_zero = outputs[0].motion_embedding.data();
    assert!(motion_zero.iter().all(|value| value.abs() < 1e-6));
    let later_motion = outputs[2].motion_embedding.data();
    assert!(later_motion.iter().any(|value| value.abs() > 0.0));
    let entropy_metric = outputs[2]
        .atlas_frame
        .metrics
        .iter()
        .find(|metric| metric.name == "z.weight_entropy")
        .expect("entropy metric");
    assert!(entropy_metric.value.is_finite());
    assert!(pipeline.temporal_digest_window(2).frames >= 1);
}
