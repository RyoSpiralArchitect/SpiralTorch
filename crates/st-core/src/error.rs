use thiserror::Error;
#[derive(Error, Debug)]
pub enum Error {
    #[error("device error: {0}")] Device(String),
    #[error("invalid argument: {0}")] Invalid(String),
    #[error("internal error: {0}")] Internal(String),
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn device(msg: &str) -> Error { Error::Device(msg.to_string()) }
pub fn invalid(msg: &str) -> Error { Error::Invalid(msg.to_string()) }
pub fn internal(msg: &str) -> Error { Error::Internal(msg.to_string()) }
