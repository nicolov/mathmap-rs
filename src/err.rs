use std::fmt;
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct ErrorWithPos {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl ErrorWithPos {
    pub fn with_pos(msg: impl Into<String>, line: usize, column: usize) -> Self {
        ErrorWithPos {
            message: msg.into(),
            line,
            column,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntaxError(pub ErrorWithPos);

#[derive(Debug, Clone)]
pub struct RuntimeError(pub ErrorWithPos);

#[derive(Debug, Clone)]
pub struct TypeError(pub ErrorWithPos);

impl Deref for SyntaxError {
    type Target = ErrorWithPos;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for RuntimeError {
    type Target = ErrorWithPos;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for TypeError {
    type Target = ErrorWithPos;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SyntaxError: {} on line {}, col {}",
            self.message, self.line, self.column
        )
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RuntimeError: {} on line {}, col {}",
            self.message, self.line, self.column
        )
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TypeError: {} on line {}, col {}",
            self.message, self.line, self.column
        )
    }
}

impl std::error::Error for SyntaxError {}

impl std::error::Error for RuntimeError {}

impl std::error::Error for TypeError {}

impl SyntaxError {
    pub fn with_pos(msg: impl Into<String>, line: usize, column: usize) -> Self {
        Self(ErrorWithPos::with_pos(msg, line, column))
    }
}

impl RuntimeError {
    pub fn with_pos(msg: impl Into<String>, line: usize, column: usize) -> Self {
        Self(ErrorWithPos::with_pos(msg, line, column))
    }
}

impl TypeError {
    pub fn with_pos(msg: impl Into<String>, line: usize, column: usize) -> Self {
        Self(ErrorWithPos::with_pos(msg, line, column))
    }
}

#[derive(Debug)]
pub enum MathMapError {
    Syntax(SyntaxError),
    Runtime(RuntimeError),
    Type(TypeError),
}

impl std::fmt::Display for MathMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MathMapError::Syntax(e) => write!(f, "{}", e),
            MathMapError::Runtime(e) => write!(f, "{}", e),
            MathMapError::Type(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for MathMapError {}

impl From<SyntaxError> for MathMapError {
    fn from(value: SyntaxError) -> Self {
        MathMapError::Syntax(value)
    }
}

impl From<RuntimeError> for MathMapError {
    fn from(value: RuntimeError) -> Self {
        MathMapError::Runtime(value)
    }
}

impl From<TypeError> for MathMapError {
    fn from(value: TypeError) -> Self {
        MathMapError::Type(value)
    }
}
