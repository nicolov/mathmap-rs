use std::fmt;

#[derive(Debug, Clone)]
pub struct SyntaxError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl SyntaxError {
    pub fn with_pos(msg: impl Into<String>, line: usize, column: usize) -> Self {
        SyntaxError {
            message: msg.into(),
            line,
            column,
        }
    }
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SyntaxError: {} on line {}, col {}", self.message, self.line, self.column)
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
