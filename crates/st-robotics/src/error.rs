use std::fmt;

/// Error type covering invalid robotics runtime configuration or inputs.
#[derive(Debug)]
pub enum RoboticsError {
    /// Attempted to register a sensor channel that already exists.
    ChannelExists(String),
    /// Requested a channel that has not been registered.
    ChannelMissing(String),
    /// Channel was configured with a zero dimension.
    InvalidDimension { channel: String },
    /// Invalid smoothing coefficient provided for a channel.
    InvalidSmoothingCoefficient { channel: String, alpha: f32 },
    /// Payload provided with incorrect dimensionality for a channel.
    DimensionMismatch {
        channel: String,
        expected: usize,
        actual: usize,
    },
    /// Calibration bias did not match the channel dimensionality.
    BiasLengthMismatch {
        channel: String,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for RoboticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelExists(name) => write!(f, "sensor channel '{name}' already registered"),
            Self::ChannelMissing(name) => write!(f, "sensor channel '{name}' is not registered"),
            Self::InvalidDimension { channel } => {
                write!(
                    f,
                    "sensor channel '{channel}' must have a non-zero dimension"
                )
            }
            Self::InvalidSmoothingCoefficient { channel, alpha } => write!(
                f,
                "smoothing coefficient for channel '{channel}' must be in the range (0, 1]; got {alpha}"
            ),
            Self::DimensionMismatch {
                channel,
                expected,
                actual,
            } => write!(
                f,
                "payload for channel '{channel}' must contain {expected} values (got {actual})",
            ),
            Self::BiasLengthMismatch {
                channel,
                expected,
                actual,
            } => write!(
                f,
                "bias for channel '{channel}' must contain {expected} values (got {actual})",
            ),
        }
    }
}

impl std::error::Error for RoboticsError {}
