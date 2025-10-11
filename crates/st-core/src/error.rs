
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Device error: {0}")]
    Device(String),
    #[error("DType error: {0}")]
    DType(String),
    #[error("Shape error: {0}")]
    Shape(String),
    #[error("Other: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn device(msg:&str)->Error { Error::Device(msg.to_string()) }
pub fn dtype(msg:&str)->Error { Error::DType(msg.to_string()) }
pub fn shape(msg:&str)->Error { Error::Shape(msg.to_string()) }
