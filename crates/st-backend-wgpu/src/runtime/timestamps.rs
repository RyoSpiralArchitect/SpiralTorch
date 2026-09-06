//! Opt-in pass timestamps. No wall-clock substitution or default-runtime mutation.

use super::{empty_buffer, ReadbackLease, WgpuContext, WgpuRuntimeError};

#[derive(Clone, Debug, PartialEq)]
pub struct PassTimestamp {
    pub start_tick: u64,
    pub end_tick: u64,
    pub elapsed_ns: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PassTimestamps {
    pub timestamp_period_ns: f64,
    pub passes: Vec<PassTimestamp>,
}

fn invalid(message: impl Into<String>) -> WgpuRuntimeError {
    WgpuRuntimeError::InvalidTimestamps {
        message: message.into(),
    }
}

fn query_count(passes: u32) -> Result<u32, WgpuRuntimeError> {
    passes
        .checked_mul(2)
        .filter(|n| *n > 0 && *n <= wgpu::QUERY_SET_MAX_QUERIES)
        .ok_or_else(|| invalid("pass count exceeds the nonzero query-set budget"))
}

fn decode(bytes: &[u8], count: u32, period: f64) -> Result<PassTimestamps, WgpuRuntimeError> {
    if !period.is_finite()
        || period <= 0.0
        || bytes.len() != count as usize * 8
        || count == 0
        || !count.is_multiple_of(2)
    {
        return Err(invalid("invalid timestamp period or readback length"));
    }
    let mut passes = Vec::with_capacity(count as usize / 2);
    for pair in bytes.as_chunks::<16>().0 {
        let start_tick = u64::from_le_bytes(pair[..8].try_into().unwrap());
        let end_tick = u64::from_le_bytes(pair[8..].try_into().unwrap());
        // Subtract integers before conversion: absolute device clocks exceed 2^53.
        let ticks = end_tick
            .checked_sub(start_tick)
            .ok_or_else(|| invalid("timestamp order reversed or wrapped"))?;
        let elapsed_ns = ticks as f64 * period;
        if !elapsed_ns.is_finite() {
            return Err(invalid("timestamp duration overflowed"));
        }
        passes.push(PassTimestamp {
            start_tick,
            end_tick,
            elapsed_ns,
        });
    }
    Ok(PassTimestamps {
        timestamp_period_ns: period,
        passes,
    })
}

/// Owned query/resolve buffers for an instrumented dispatch, including its chunks.
pub(crate) struct PassTimestampRecorder {
    context: WgpuContext,
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: ReadbackLease,
    count: u32,
    period: f64,
}

impl PassTimestampRecorder {
    pub(crate) fn new(context: WgpuContext, passes: u32) -> Result<Self, WgpuRuntimeError> {
        let count = query_count(passes)?;
        if !context
            .device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Err(WgpuRuntimeError::TimestampQueriesUnavailable);
        }
        let period = f64::from(context.queue().get_timestamp_period());
        if !period.is_finite() || period <= 0.0 {
            return Err(invalid("device timestamp period is unavailable"));
        }
        let resolve = empty_buffer::<u64>(
            context.device(),
            "profile.timestamp.resolve",
            count as usize,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let staging = ReadbackLease::unpooled(empty_buffer::<u64>(
            context.device(),
            "profile.timestamp.readback",
            count as usize,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?);
        let queries = context
            .device()
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile.pass.timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
        Ok(Self {
            context,
            queries,
            resolve,
            staging,
            count,
            period,
        })
    }

    pub(crate) fn writes(&self, pass: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        assert!(pass < self.count / 2);
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(pass * 2),
            end_of_pass_write_index: Some(pass * 2 + 1),
        }
    }

    pub(crate) fn resolve(self, encoder: &mut wgpu::CommandEncoder) -> TimestampReadback {
        encoder.resolve_query_set(&self.queries, 0..self.count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            self.staging.buffer(),
            0,
            u64::from(self.count) * 8,
        );
        TimestampReadback {
            context: self.context,
            staging: self.staging,
            count: self.count,
            period: self.period,
        }
    }
}

/// A pending timestamp copy owns its storage independently of later submissions.
pub struct TimestampReadback {
    context: WgpuContext,
    staging: ReadbackLease,
    count: u32,
    period: f64,
}

impl TimestampReadback {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(mut self) -> Result<PassTimestamps, WgpuRuntimeError> {
        let bytes = self.staging.read(
            &self.context,
            std::time::Duration::from_secs(30),
            "profile.pass.timestamps",
        )?;
        decode(&bytes, self.count, self.period)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(self) -> Result<PassTimestamps, WgpuRuntimeError> {
        let bytes = self
            .staging
            .read_async(self.context, "profile.pass.timestamps")
            .await?;
        decode(&bytes, self.count, self.period)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn timestamps_preserve_integer_deltas_and_zero_quantized_intervals() {
        let start = (1u64 << 60) + 1;
        let times = decode(&bytes(&[start, start + 3, start + 3, start + 3]), 4, 2.5).unwrap();
        assert_eq!(times.passes[0].elapsed_ns, 7.5);
        assert_eq!(times.passes[0].start_tick, start);
        assert_eq!(times.passes[1].elapsed_ns, 0.0);
    }

    #[test]
    fn malformed_timestamps_fail_without_wall_clock_fallback() {
        for period in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(decode(&bytes(&[1, 3]), 2, period).is_err());
        }
        assert!(decode(&bytes(&[3, 1]), 2, 1.0).is_err());
        assert!(decode(&bytes(&[1]), 2, 1.0).is_err());
        assert!(decode(&[], 0, 1.0).is_err());
        assert!(decode(&bytes(&[1]), 1, 1.0).is_err());
    }

    #[test]
    fn query_budget_is_checked_before_allocation() {
        assert_eq!(query_count(2048).unwrap(), 4096);
        assert_eq!(
            query_count(wgpu::QUERY_SET_MAX_QUERIES / 2).unwrap(),
            wgpu::QUERY_SET_MAX_QUERIES
        );
        for count in [0, wgpu::QUERY_SET_MAX_QUERIES / 2 + 1, u32::MAX] {
            assert!(query_count(count).is_err());
        }
    }
}
