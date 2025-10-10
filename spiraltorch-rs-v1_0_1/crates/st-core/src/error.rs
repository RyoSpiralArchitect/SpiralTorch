use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpiralError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("dtype error: {0}")]
    DType(String),
    #[error("device error: {0}")]
    Device(String),
}

pub type Result<T> = std::result::Result<T, SpiralError>;

pub fn shape<S: Into<String>>(msg: S) -> SpiralError { SpiralError::Shape(msg.into()) }
pub fn dtype<S: Into<String>>(msg: S) -> SpiralError { SpiralError::DType(msg.into()) }
pub fn device<S: Into<String>>(msg: S) -> SpiralError { SpiralError::Device(msg.into()) }
