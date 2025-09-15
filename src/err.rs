use std::fmt;

#[derive(Debug)]
pub struct SyntaxError {
    pub message: String,
}

impl SyntaxError {
    pub fn new(msg: impl Into<String>) -> Self {
        SyntaxError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SyntaxError: {}", self.message)
    }
}

impl std::error::Error for SyntaxError {}

#[derive(Debug)]
pub enum MathMapError {
    Parse(SyntaxError),
}

impl std::fmt::Display for MathMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathMapError::Parse(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for MathMapError {}
