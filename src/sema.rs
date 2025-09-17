// Semantic analysis and type checking.

#![allow(dead_code)]

use crate::ast::Expression;
use std::error::Error;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Type {
    #[default]
    Unknown,
    Int,
    Tuple(usize),
}

pub struct SemanticAnalyzer {}

impl SemanticAnalyzer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Parser;

    // #[test]
    // fn analyze_int_constant() -> Result<(), Box<dyn Error>> {
    //     let input = "1 + 2";
    //     let mut parser = Parser::new(input);
    //     let ast = parser.parse_expression(1)?;
    //     dbg!(&ast);
    //     let typed_expr = SemanticAnalyzer::analyze_expr(&ast);
    //     assert_eq!(typed_expr.ty, Type::Int);
    //     Ok(())
    // }
}
