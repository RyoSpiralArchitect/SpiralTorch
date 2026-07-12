use pyo3::prelude::*;

use st_core::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};

fn zsource_to_str(source: ZSource) -> &'static str {
    match source {
        ZSource::Microlocal => "microlocal",
        ZSource::Maxwell => "maxwell",
        ZSource::Graph => "graph",
        ZSource::Desire => "desire",
        ZSource::GW => "gw",
        ZSource::RealGrad => "realgrad",
        ZSource::Other(tag) => tag,
    }
}

fn support_to_tuple(support: ZSupport) -> (f32, f32, f32) {
    (support.leading, support.central, support.trailing)
}

fn scale_to_tuple(scale: Option<ZScale>) -> Option<(f32, f32)> {
    scale.map(|value| (value.physical_radius, value.log_radius))
}

#[pyclass(module = "spiraltorch.psi", name = "ZPulseSnapshot")]
#[derive(Clone)]
pub(crate) struct PyZPulse {
    inner: ZPulse,
}

impl PyZPulse {
    #[cfg(any(feature = "nn", feature = "text"))]
    pub(crate) fn from_pulse(inner: ZPulse) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyZPulse {
    #[getter]
    pub fn source(&self) -> &'static str {
        zsource_to_str(self.inner.source)
    }

    #[getter]
    pub fn ts(&self) -> u64 {
        self.inner.ts
    }

    #[getter]
    pub fn tempo(&self) -> f32 {
        self.inner.tempo
    }

    #[getter]
    pub fn band_energy(&self) -> (f32, f32, f32) {
        self.inner.band_energy
    }

    #[getter]
    pub fn drift(&self) -> f32 {
        self.inner.drift
    }

    #[getter]
    pub fn z_bias(&self) -> f32 {
        self.inner.z_bias
    }

    #[getter]
    pub fn density_fluctuation(&self) -> f32 {
        self.inner.density_fluctuation
    }

    #[getter]
    pub fn support(&self) -> (f32, f32, f32) {
        support_to_tuple(self.inner.support)
    }

    #[getter]
    pub fn scale(&self) -> Option<(f32, f32)> {
        scale_to_tuple(self.inner.scale)
    }

    #[getter]
    pub fn quality(&self) -> f32 {
        self.inner.quality
    }

    #[getter]
    pub fn stderr(&self) -> f32 {
        self.inner.stderr
    }

    #[getter]
    pub fn latency_ms(&self) -> f32 {
        self.inner.latency_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_preserves_the_canonical_pulse_fields() {
        let snapshot = PyZPulse {
            inner: ZPulse {
                source: ZSource::Graph,
                ts: 42,
                tempo: 0.75,
                band_energy: (0.5, 0.3, 0.2),
                density_fluctuation: 0.25,
                drift: -0.125,
                z_bias: 0.375,
                support: ZSupport::new(3.0, 2.0, 1.0),
                scale: ZScale::from_components(2.0, 2.0_f32.ln()),
                quality: 0.9,
                stderr: 0.05,
                latency_ms: 1.5,
            },
        };

        assert_eq!(snapshot.source(), "graph");
        assert_eq!(snapshot.ts(), 42);
        assert_eq!(snapshot.tempo(), 0.75);
        assert_eq!(snapshot.band_energy(), (0.5, 0.3, 0.2));
        assert_eq!(snapshot.density_fluctuation(), 0.25);
        assert_eq!(snapshot.drift(), -0.125);
        assert_eq!(snapshot.z_bias(), 0.375);
        assert_eq!(snapshot.support(), (3.0, 2.0, 1.0));
        assert_eq!(snapshot.scale(), Some((2.0, 2.0_f32.ln())));
        assert_eq!(snapshot.quality(), 0.9);
        assert_eq!(snapshot.stderr(), 0.05);
        assert_eq!(snapshot.latency_ms(), 1.5);
    }
}
