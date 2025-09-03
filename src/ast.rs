#![allow(dead_code)]

use crate::lexer;
use std::fmt;

#[derive(Debug)]
struct OpInfo {
    precedence: u8,
    associativity: Associativity,
    name: &'static str,
}

#[derive(Debug, PartialEq)]
enum Associativity {
    Left,
    Right,
}


fn get_op_info(op: &lexer::Token) -> Option<OpInfo> {
    match op {
        lexer::Token::Less => Some(OpInfo {
            precedence: 4,
            associativity: Associativity::Left,
            name: "__less",
        }),
        lexer::Token::Plus => Some(OpInfo {
            precedence: 5,
            associativity: Associativity::Left,
            name: "__add",
        }),
        lexer::Token::Minus => Some(OpInfo {
            precedence: 5,
            associativity: Associativity::Left,
            name: "__sub",
        }),
        lexer::Token::Star => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__mul",
        }),
        lexer::Token::Slash => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__div",
        }),
        lexer::Token::Percent => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__mod",
        }),
        lexer::Token::Assign => Some(OpInfo {
            precedence: 2,
            associativity: Associativity::Right,
            name: "__assign",
        }),
        _ => None,
    }
}

#[derive(Debug, PartialEq)]
pub enum Expression {
    // From exprtree.h
    IntConst {
        value: i64,
    },
    FloatConst {
        value: f32,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    Variable {
        name: String,
    },
    If {
        condition: Box<Expression>,
        then: Box<Expression>,
        else_: Option<Box<Expression>>,
    },
    Assignment {
        name: String,
        value: Box<Expression>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Filter {
    pub name: String,
    pub exprs: Vec<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct Module {
    pub filters: Vec<Filter>,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    pub fn new(msg: impl Into<String>) -> Self {
        ParseError {
            message: msg.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

pub struct Parser<'a> {
    tokens: std::iter::Peekable<lexer::Lexer<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            tokens: lexer::Lexer::new(input).peekable(),
        }
    }

    fn expect(&mut self, expected: lexer::Token) -> Result<(), ParseError> {
        let t = self.tokens.next();
        match t {
            Some(t) if t == expected => Ok(()),
            Some(t) => Err(ParseError::new(format!(
                "Expected {:?}, got {:?}",
                expected, t
            ))),
            None => Err(ParseError::new("Unexpected end of input")),
        }
    }

    fn consume_ident(&mut self) -> Result<String, ParseError> {
        let t = self.tokens.next();
        match t {
            Some(lexer::Token::Ident(s)) => Ok(s.to_string()),
            Some(t) => Err(ParseError::new(format!("Expected identifier, got {:?}", t))),
            None => Err(ParseError::new("Unexpected end of input")),
        }
    }

    fn parse_atom(&mut self) -> Result<Expression, ParseError> {
        let mut expr = match self.tokens.peek() {
            Some(lexer::Token::Minus) => {
                self.tokens.next();
                let operand = self.parse_expression(100)?;
                let args = vec![operand];
                Ok(Expression::FunctionCall {
                    name: "__neg".to_string(),
                    args,
                })
            }
            Some(lexer::Token::LParen) => {
                self.expect(lexer::Token::LParen)?;
                // Start parsing a subexpression inside parentheses.
                let subexpr = self.parse_expression(1)?;
                self.expect(lexer::Token::RParen)?;
                Ok(subexpr)
            }
            Some(lexer::Token::Ident(s)) => {
                let name = s.to_string();
                self.tokens.next();
                Ok(Expression::Variable { name: name })
            }
            Some(lexer::Token::NumberLit(s)) => {
                if s.contains(".") {
                    match s.parse::<f32>() {
                        Ok(x) => {
                            self.tokens.next(); // consume number
                            Ok(Expression::FloatConst { value: x })
                        }
                        Err(_) => Err(ParseError::new(format!("Invalid float literal: {}", s))),
                    }
                } else {
                    match s.parse::<i64>() {
                        Ok(x) => {
                            self.tokens.next(); // consume number
                            Ok(Expression::IntConst { value: x })
                        }
                        Err(_) => Err(ParseError::new(format!("Invalid int literal: {}", s))),
                    }
                }
            }
            Some(lexer::Token::If) => {
                self.tokens.next();
                let condition_expr = self.parse_expression(1)?;
                self.expect(lexer::Token::Then)?;
                let then_expr = self.parse_expression(1)?;

                let else_expr = if self.tokens.peek() == Some(&lexer::Token::Else) {
                    self.tokens.next();
                    Some(Box::new(self.parse_expression(1)?))
                } else {
                    None
                };

                self.expect(lexer::Token::End)?;

                Ok(Expression::If {
                    condition: Box::new(condition_expr),
                    then: Box::new(then_expr),
                    else_: else_expr,
                })
            }
            None => Err(ParseError::new(
                "Unexpected end of input while parsing expression",
            )),
            _ => Err(ParseError::new(format!(
                "Unexpected token in expression: {:?}",
                self.tokens.peek()
            ))),
        }?;

        // We have a choice between handling postfix operators ([], function calls, etc..)
        // here or in parse_expression. Apparently, the most common option is to do it here
        // so we don't have to pollute precedence tables with extra tokens. This works
        // because postfix operators have the highest precedence. Note that
        // https://pdubroy.github.io/200andchange/precedence-climbing/ does it the other way
        // and puts function call handling in parse_expression.
        loop {
            match self.tokens.peek() {
                Some(lexer::Token::LParen) => {
                    if let Expression::Variable { ref name } = expr {
                        self.tokens.next(); // consume '('
                        let mut args = Vec::new();
                        if self.tokens.peek() != Some(&lexer::Token::RParen) {
                            loop {
                                args.push(self.parse_expression(1)?);
                                if self.tokens.peek() == Some(&lexer::Token::Comma) {
                                    self.tokens.next(); // consume ','
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(lexer::Token::RParen)?;
                        expr = Expression::FunctionCall {
                            name: name.clone(),
                            args,
                        };
                    } else {
                        return Err(ParseError::new(
                            "Only identifiers can be called as functions",
                        ));
                    }
                }
                Some(lexer::Token::Colon) => {
                    if let Expression::Variable { ref name } = expr {
                        self.expect(lexer::Token::Colon)?;
                        self.expect(lexer::Token::LBracket)?;
                        let mut args = Vec::new();
                        while self.tokens.peek() != Some(&lexer::Token::RBracket) {
                            args.push(self.parse_expression(1)?);
                            if self.tokens.peek() == Some(&lexer::Token::Comma) {
                                self.tokens.next(); // consume ','
                            } else {
                                break;
                            }
                        }
                        self.expect(lexer::Token::RBracket)?;

                        let fn_name = match name.as_str() {
                            "rgba" => Ok("rgbaColor"),
                            _ => Err(ParseError::new(format!(
                                "Invalid cast: unknown tuple tag: {:?}",
                                name
                            ))),
                        }?;

                        expr = Expression::FunctionCall {
                            name: fn_name.to_string(),
                            args: args,
                        }
                    } else {
                        return Err(ParseError::new(format!(
                            "Invalid cast: lhs is not a valid tuple tag: {:?}",
                            expr
                        )));
                    }
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    pub fn parse_expression(&mut self, min_precedence: u8) -> Result<Expression, ParseError> {
        let mut atom_lhs = self.parse_atom()?;

        loop {
            let peek = self.tokens.peek();
            if peek.is_none() {
                break;
            }

            let op_info = get_op_info(peek.unwrap());

            if op_info.is_none() {
                break;
            }

            let op_info = op_info.unwrap();

            if op_info.precedence < min_precedence {
                break;
            }

            // Consume the current operator token.
            self.tokens.next();

            let next_min_precedence = match op_info.associativity {
                Associativity::Left => op_info.precedence + 1,
                Associativity::Right => op_info.precedence,
            };

            let atom_rhs = self.parse_expression(next_min_precedence)?;

            // Rewrite assignments into the proper AST node.
            if op_info.name == "__assign" {
                // Check that the LHS is a variable (ie lvalue).
                if let Expression::Variable { name } = atom_lhs {
                    return Ok(Expression::Assignment {
                        name: name.to_string(),
                        value: Box::new(atom_rhs),
                    });
                } else {
                    return Err(ParseError::new(format!(
                        "Invalid assignment: lhs is not a variable: {:?}",
                        atom_lhs
                    )));
                }
            } else {
                atom_lhs = Expression::FunctionCall {
                    name: op_info.name.to_string(),
                    args: vec![atom_lhs, atom_rhs],
                };
            }
        }

        return Ok(atom_lhs);
    }

    fn parse_filter(&mut self) -> Result<Filter, ParseError> {
        self.expect(lexer::Token::Filter)?;

        let name = self.consume_ident()?;

        self.expect(lexer::Token::LParen)?;
        // todo: parse arguments
        self.expect(lexer::Token::RParen)?;

        let mut expressions: Vec<Expression> = Vec::new();

        loop {
            if matches!(self.tokens.peek(), Some(lexer::Token::End)) {
                break;
            }

            let expr = self.parse_expression(1)?;
            expressions.push(expr);

            if matches!(self.tokens.peek(), Some(lexer::Token::Semicolon)) {
                self.tokens.next();
                continue;
            } else {
                break;
            }
        }

        self.expect(lexer::Token::End)?;

        Ok(Filter {
            name: name,
            exprs: expressions,
        })
    }

    fn parse_module(&mut self) -> Result<Module, ParseError> {
        let mut filters: Vec<Filter> = Vec::new();

        while self.tokens.peek().is_some() {
            match self.parse_filter() {
                Ok(filter) => filters.push(filter),
                Err(e) => return Err(e),
            }
        }

        Ok(Module { filters: filters })
    }
}

pub fn parse_module(input: &str) -> Result<Module, ParseError> {
    let mut parser = Parser::new(input);
    parser.parse_module()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_numerical_expression() {
        let input = "1 + 2 * 3 / 4";
        let mut parser = Parser::new(input);

        let ast = parser.parse_expression(1).unwrap();

        let ast_ref = Expression::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                Expression::IntConst { value: 1 },
                Expression::FunctionCall {
                    name: "__div".to_string(),
                    args: vec![
                        Expression::FunctionCall {
                            name: "__mul".to_string(),
                            args: vec![
                                Expression::IntConst { value: 2 },
                                Expression::IntConst { value: 3 },
                            ],
                        },
                        Expression::IntConst { value: 4 },
                    ],
                },
            ],
        };

        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expression_with_variable() {
        let input = "x + 100";
        let mut parser = Parser::new(input);

        let ast = parser.parse_expression(1).unwrap();

        let ast_ref = Expression::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                Expression::Variable {
                    name: "x".to_string(),
                },
                Expression::IntConst { value: 100 },
            ],
        };

        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expr_function_call() {
        let input = "fn(100, x)";
        let mut parser = Parser::new(input);

        let ast = parser.parse_expression(1).unwrap();

        let ast_ref = Expression::FunctionCall {
            name: "fn".to_string(),
            args: vec![
                Expression::IntConst { value: 100 },
                Expression::Variable {
                    name: "x".to_string(),
                },
            ],
        };

        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expr_if() {
        let input = "if x < 100 then 100 else 200 end";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let ast_ref = Expression::If {
            condition: Box::new(Expression::FunctionCall {
                name: "__less".to_string(),
                args: vec![
                    Expression::Variable {
                        name: "x".to_string(),
                    },
                    Expression::IntConst { value: 100 },
                ],
            }),
            then: Box::new(Expression::IntConst { value: 100 }),
            else_: Some(Box::new(Expression::IntConst { value: 200 })),
        };
        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expr_unary() {
        let input = "-x";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let ast_ref = Expression::FunctionCall {
            name: "__neg".to_string(),
            args: vec![Expression::Variable {
                name: "x".to_string(),
            }],
        };
        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expr_assignment() {
        let input = "x = 100";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let ast_ref = Expression::Assignment {
            name: "x".to_string(),
            value: Box::new(Expression::IntConst { value: 100 }),
        };
        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_expr_cast() {
        let input = "rgba:[1,2,3,4]";
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1).unwrap();
        let ast_ref = Expression::FunctionCall {
            name: "rgbaColor".to_string(),
            args: vec![
                Expression::IntConst { value: 1 },
                Expression::IntConst { value: 2 },
                Expression::IntConst { value: 3 },
                Expression::IntConst { value: 4 },
            ],
        };
        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_module() {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";

        let ast = parse_module(input).unwrap();

        let ast_ref = Module {
            filters: vec![Filter {
                name: "red".to_string(),
                exprs: vec![Expression::FunctionCall {
                    name: "rgbColor".to_string(),
                    args: vec![
                        Expression::IntConst { value: 1 },
                        Expression::IntConst { value: 0 },
                        Expression::IntConst { value: 0 },
                    ],
                }],
            }],
        };

        assert_eq!(ast, ast_ref);
    }

    #[test]
    fn test_parse_module_2() {
        let input = "filter red ()
            z = 1;
            rgbColor(z, 0, 0)
        end";

        let ast = parse_module(input).unwrap();

        assert!(ast.filters.len() == 1);
        assert!(ast.filters[0].name == "red");
        assert!(ast.filters[0].exprs.len() == 2);

        let assign_expr = &ast.filters[0].exprs[0];
        let ast_ref = Expression::Assignment {
            name: "z".to_string(),
            value: Box::new(Expression::IntConst { value: 1 }),
        };
        assert_eq!(assign_expr, &ast_ref);
    }
}
