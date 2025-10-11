use thiserror::Error;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Msg(String),
}
pub fn msg(msg: &str) -> Error { Error::Msg(msg.to_string()) }
pub fn device(msg: &str) -> Error { Error::Msg(format!("device: {msg}")) }
