use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct Error { msg: String }
impl Error { pub fn new<S: Into<String>>(s: S) -> Self { Self { msg: s.into() } } }
impl Display for Error { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.msg) } }
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn device<S: Into<String>>(s: S) -> Error { Error::new(s) }
pub fn shape<S: Into<String>>(s: S) -> Error { Error::new(s) }
