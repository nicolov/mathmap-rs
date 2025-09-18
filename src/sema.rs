// Semantic analysis and type checking.

#![allow(dead_code)]

use crate::ast;
use crate::err::TypeError;
use ast::Expression;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Type {
    #[default]
    Unknown,
    Int,
    Tuple(usize),
}

impl Type {
    pub fn as_wgsl(&self) -> &str {
        match self {
            Type::Int => "i32",
            Type::Tuple(1) => "f32",
            Type::Tuple(4) => "vec4<f32>",
            _ => todo!(),
        }
    }
}

pub struct SemanticAnalyzer {}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {}
    }

    fn analyze_expr(&mut self, expr: &mut Expression) -> Result<(), TypeError> {
        match expr {
            // Types of int and float literals are already filled in by the parser.
            Expression::IntConst { .. } => Ok(()),
            Expression::FloatConst { .. } => Ok(()),
            // Expression::TupleConst { .. } => Ok(()),
            Expression::FunctionCall { name, args, ty } => {
                for arg in &mut *args {
                    self.analyze_expr(arg)?;
                }
                match name.as_str() {
                    "__add" => {
                        // For now just go through the args and promote to float if any of them is.
                        // todo: handle broadcasting, tuples, and implicit casts.
                        // this needs a representation for function signatures.
                        let mut t = Type::Int;
                        for arg in args {
                            match arg {
                                Expression::IntConst { .. } => {}
                                Expression::FloatConst { .. } => t = Type::Tuple(1),
                                _ => {
                                    return Err(TypeError::with_pos(
                                        "expected int or float constant",
                                        0,
                                        0,
                                    ));
                                }
                            }
                        }
                        *ty = t;
                        Ok(())
                    }
                    "rgbColor" => {
                        assert!(args.len() == 3);
                        *ty = Type::Tuple(4);
                        Ok(())
                    }
                    _ => {
                        unimplemented!("unimplemented function {}", name);
                    }
                }
            }
            _ => {
                todo!();
            }
        }
    }

    pub fn analyze_filter(&mut self, filter: &mut ast::Filter) -> Result<(), TypeError> {
        if filter.exprs.is_empty() {
            return Err(TypeError::with_pos("empty filter", 0, 0));
        }
        for expr in &mut filter.exprs {
            self.analyze_expr(expr)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Expression as E;
    use ast::Parser;
    use std::error::Error;

    fn analyze_expr(src: &str) -> Result<Expression, Box<dyn Error>> {
        let mut parser = Parser::new(src);
        let mut ast = parser.parse_expression(1)?;
        let mut sema = SemanticAnalyzer::new();
        sema.analyze_expr(&mut ast)?;
        Ok(ast)
    }

    #[test]
    fn int_constant() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1")?;
        if let E::IntConst { ty, .. } = expr {
            assert_eq!(ty, Type::Int);
        } else {
            panic!("expected int constant");
        }
        Ok(())
    }

    #[test]
    fn add_ints() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1 + 2")?;
        if let E::FunctionCall { ty, .. } = expr {
            assert_eq!(ty, Type::Int);
        } else {
            panic!("expected function call");
        }
        Ok(())
    }

    #[test]
    fn add_int_and_float() -> Result<(), Box<dyn Error>> {
        let expr = analyze_expr("1 + 2.0")?;
        if let E::FunctionCall { ty, .. } = expr {
            assert_eq!(ty, Type::Tuple(1));
        } else {
            panic!("expected function call");
        }
        Ok(())
    }
}
