
use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Error)]
pub enum Error {
    #[error("Device error: {0}")] Device(String),
    #[error("Shape error: {0}")] Shape(String),
    #[error("Type error: {0}")] DType(String),
    #[error("Other: {0}")] Other(String),
}
pub fn device(m:&str)->Error{ Error::Device(m.to_string()) }
pub fn shape(m:&str)->Error{ Error::Shape(m.to_string()) }
pub fn dtype(m:&str)->Error{ Error::DType(m.to_string()) }
