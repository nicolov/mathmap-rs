#![allow(dead_code)]

use crate::SyntaxError;
use crate::lexer::{self, TokenKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TupleTag {
    Nil,   // Default tag for single numbers
    Rgba,  // RGBA color
    Hsva,  // HSVA color
    Ri,    // Complex number
    Xy,    // Cartesian coordinates
    Ra,    // Polar coordinates
    V2,    // 2D vector
    V3,    // 3D vector
    M2x2,  // 2x2 matrix
    M3x3,  // 3x3 matrix
    Quat,  // Non-commutative quaternion
    Cquat, // Commutative quaternion
    Hyper, // Hypercomplex number
}

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

fn get_op_info(op: &lexer::TokenKind) -> Option<OpInfo> {
    match op {
        lexer::TokenKind::Assign => Some(OpInfo {
            precedence: 2,
            associativity: Associativity::Right,
            name: "__assign",
        }),
        lexer::TokenKind::And => Some(OpInfo {
            precedence: 3,
            associativity: Associativity::Left,
            name: "__and",
        }),
        lexer::TokenKind::Or => Some(OpInfo {
            precedence: 3,
            associativity: Associativity::Left,
            name: "__or",
        }),
        lexer::TokenKind::Less => Some(OpInfo {
            precedence: 4,
            associativity: Associativity::Left,
            name: "__less",
        }),
        lexer::TokenKind::LessEqual => Some(OpInfo {
            precedence: 4,
            associativity: Associativity::Left,
            name: "__lessequal",
        }),
        lexer::TokenKind::Plus => Some(OpInfo {
            precedence: 5,
            associativity: Associativity::Left,
            name: "__add",
        }),
        lexer::TokenKind::Minus => Some(OpInfo {
            precedence: 5,
            associativity: Associativity::Left,
            name: "__sub",
        }),
        lexer::TokenKind::Star => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__mul",
        }),
        lexer::TokenKind::Slash => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__div",
        }),
        lexer::TokenKind::Percent => Some(OpInfo {
            precedence: 6,
            associativity: Associativity::Left,
            name: "__mod",
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
    TupleConst {
        tag: TupleTag,
        values: Vec<Expression>,
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
        then: Vec<Expression>,
        else_: Vec<Expression>,
    },
    While {
        condition: Box<Expression>,
        body: Vec<Expression>,
    },
    Assignment {
        name: String,
        value: Box<Expression>,
    },
    Index {
        expr: Box<Expression>,
        index: Box<Expression>,
    },
    Cast {
        tag: TupleTag,
        expr: Box<Expression>,
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

pub struct Parser<'a> {
    tokens: std::iter::Peekable<lexer::Lexer<'a>>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            tokens: lexer::Lexer::new(input).peekable(),
        }
    }

    fn expect(&mut self, expected: lexer::TokenKind) -> Result<(), SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(t) if t.item == expected => Ok(()),
            Some(t) => Err(SyntaxError::new(format!(
                "Expected {:?}, got {:?}",
                expected, t
            ))),
            None => Err(SyntaxError::new("Unexpected end of input")),
        }
    }

    fn expect_done(&mut self) -> Result<(), SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(t) => Err(SyntaxError::new(format!(
                "Expected end of input, got {:?}",
                t
            ))),
            None => Ok(()),
        }
    }

    fn consume_ident(&mut self) -> Result<String, SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(lexer::Token {
                item: TokenKind::Ident(s),
                ..
            }) => Ok(s.to_string()),
            Some(t) => Err(SyntaxError::new(format!(
                "Expected identifier, got {:?}",
                t
            ))),
            None => Err(SyntaxError::new("Unexpected end of input")),
        }
    }

    fn parse_atom(&mut self) -> Result<Expression, SyntaxError> {
        let mut expr = match self.tokens.peek() {
            Some(lexer::Token {
                item: TokenKind::Minus,
                ..
            }) => {
                self.tokens.next();
                let operand = self.parse_expression(100)?;
                let args = vec![operand];
                Ok(Expression::FunctionCall {
                    name: "__neg".to_string(),
                    args,
                })
            }
            Some(lexer::Token {
                item: TokenKind::LParen,
                ..
            }) => {
                self.expect(lexer::TokenKind::LParen)?;
                // Start parsing a subexpression inside parentheses.
                let subexpr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::RParen)?;
                Ok(subexpr)
            }
            Some(lexer::Token {
                item: TokenKind::Ident(s),
                ..
            }) => {
                let name = s.to_string();
                self.tokens.next();
                Ok(Expression::Variable { name: name })
            }
            Some(lexer::Token {
                item: TokenKind::NumberLit(s),
                ..
            }) => {
                if s.contains(".") {
                    match s.parse::<f32>() {
                        Ok(x) => {
                            self.tokens.next(); // consume number
                            Ok(Expression::FloatConst { value: x })
                        }
                        Err(_) => Err(SyntaxError::new(format!("Invalid float literal: {}", s))),
                    }
                } else {
                    match s.parse::<i64>() {
                        Ok(x) => {
                            self.tokens.next(); // consume number
                            Ok(Expression::IntConst { value: x })
                        }
                        Err(_) => Err(SyntaxError::new(format!("Invalid int literal: {}", s))),
                    }
                }
            }
            Some(lexer::Token {
                item: TokenKind::If,
                ..
            }) => {
                self.expect(lexer::TokenKind::If)?;
                let condition_expr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::Then)?;
                let then_expr = self.parse_expr_block()?;

                let else_expr =
                    if self.tokens.peek().map(|t| t.item) == Some(lexer::TokenKind::Else) {
                        self.expect(lexer::TokenKind::Else)?;
                        self.parse_expr_block()?
                    } else {
                        vec![]
                    };

                self.expect(lexer::TokenKind::End)?;

                Ok(Expression::If {
                    condition: Box::new(condition_expr),
                    then: then_expr,
                    else_: else_expr,
                })
            }
            Some(lexer::Token {
                item: TokenKind::While,
                ..
            }) => {
                self.expect(lexer::TokenKind::While)?;
                let condition_expr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::Do)?;
                let body_expr = self.parse_expr_block()?;
                self.expect(lexer::TokenKind::End)?;

                Ok(Expression::While {
                    condition: Box::new(condition_expr),
                    body: body_expr,
                })
            }
            None => Err(SyntaxError::new(
                "Unexpected end of input while parsing expression",
            )),
            _ => Err(SyntaxError::new(format!(
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
            match self.tokens.peek().map(|t| t.item) {
                Some(lexer::TokenKind::LParen) => {
                    if let Expression::Variable { ref name } = expr {
                        self.tokens.next(); // consume '('
                        let mut args = Vec::new();
                        if self.tokens.peek().map(|t| t.item) != Some(lexer::TokenKind::RParen) {
                            loop {
                                args.push(self.parse_expression(1)?);
                                if self.tokens.peek().map(|t| t.item)
                                    == Some(lexer::TokenKind::Comma)
                                {
                                    self.tokens.next(); // consume ','
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(lexer::TokenKind::RParen)?;
                        expr = Expression::FunctionCall {
                            name: name.clone(),
                            args,
                        };
                    } else {
                        return Err(SyntaxError::new(
                            "Only identifiers can be called as functions",
                        ));
                    }
                }
                Some(lexer::TokenKind::LBracket) => {
                    self.expect(lexer::TokenKind::LBracket)?;
                    let index = self.parse_expression(1)?;
                    self.expect(lexer::TokenKind::RBracket)?;
                    expr = Expression::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Some(lexer::TokenKind::Colon) => {
                    self.expect(lexer::TokenKind::Colon)?;
                    // I guess we could make special tokens for the tuple tags since they're a fixed set, but for now
                    // they're parsed as variables.
                    if let Expression::Variable { ref name } = expr {
                        let tag = match name.as_str() {
                            "rgba" => TupleTag::Rgba,
                            "ri" => TupleTag::Ri,
                            "xy" => TupleTag::Xy,
                            "quat" => TupleTag::Quat,
                            _ => panic!("unknown tuple tag {:?}", name),
                        };

                        // Handle either tuple literals rgba:[1,2,3,4] or casts ri:xy.
                        if self.tokens.peek().map(|t| t.item) == Some(lexer::TokenKind::LBracket) {
                            self.expect(lexer::TokenKind::LBracket)?;
                            let mut args = Vec::new();
                            while self.tokens.peek().map(|t| t.item)
                                != Some(lexer::TokenKind::RBracket)
                            {
                                args.push(self.parse_expression(1)?);
                                if self.tokens.peek().map(|t| t.item)
                                    == Some(lexer::TokenKind::Comma)
                                {
                                    self.tokens.next(); // consume ','
                                } else {
                                    break;
                                }
                            }
                            self.expect(lexer::TokenKind::RBracket)?;

                            expr = Expression::TupleConst { tag, values: args }
                        } else {
                            let rhs_expr = self.parse_expression(1)?;
                            expr = Expression::Cast {
                                tag,
                                expr: Box::new(rhs_expr),
                            }
                        }
                    } else {
                        return Err(SyntaxError::new(format!(
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

    pub fn parse_expression(&mut self, min_precedence: u8) -> Result<Expression, SyntaxError> {
        let mut atom_lhs = self.parse_atom()?;

        loop {
            let peek = self.tokens.peek();
            if peek.is_none() {
                break;
            }

            let op_info = get_op_info(&peek.unwrap().item);

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
                    return Err(SyntaxError::new(format!(
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

    // Parse a block of expressions separated by semicolon (for filters, if/while, etc..)
    pub fn parse_expr_block(&mut self) -> Result<Vec<Expression>, SyntaxError> {
        let mut expressions: Vec<Expression> = Vec::new();

        loop {
            if matches!(
                self.tokens.peek().map(|t| t.item),
                Some(lexer::TokenKind::End)
            ) {
                break;
            }

            let expr = self.parse_expression(1)?;
            expressions.push(expr);

            if matches!(
                self.tokens.peek().map(|t| t.item),
                Some(lexer::TokenKind::Semicolon)
            ) {
                self.tokens.next();
                continue;
            } else {
                break;
            }
        }

        Ok(expressions)
    }

    fn parse_filter(&mut self) -> Result<Filter, SyntaxError> {
        self.expect(lexer::TokenKind::Filter)?;

        let name = self.consume_ident()?;

        self.expect(lexer::TokenKind::LParen)?;
        // todo: parse arguments
        self.expect(lexer::TokenKind::RParen)?;

        let expressions = self.parse_expr_block()?;
        self.expect(lexer::TokenKind::End)?;

        Ok(Filter {
            name: name,
            exprs: expressions,
        })
    }

    fn parse_module(&mut self) -> Result<Module, SyntaxError> {
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

pub fn parse_module(input: &str) -> Result<Module, SyntaxError> {
    let mut parser = Parser::new(input);
    parser.parse_module()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_as_expr(input: &str) -> Result<Expression, SyntaxError> {
        let mut parser = Parser::new(input);
        let ast = parser.parse_expression(1)?;
        parser.expect_done()?;
        Ok(ast)
    }

    #[test]
    fn parse_numerical_expression() -> Result<(), SyntaxError> {
        let input = "1 + 2 * 3 / 4";
        let ast = parse_as_expr(input)?;

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
        Ok(())
    }

    #[test]
    fn parse_expression_with_variable() -> Result<(), SyntaxError> {
        let input = "x + 100";
        let ast = parse_as_expr(input)?;

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
        Ok(())
    }

    #[test]
    fn parse_expr_function_call() -> Result<(), SyntaxError> {
        let input = "fn(100, x)";
        let ast = parse_as_expr(input)?;

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
        Ok(())
    }

    #[test]
    fn parse_expr_index() -> Result<(), SyntaxError> {
        let input = "x[1]";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::Index {
            expr: Box::new(Expression::Variable {
                name: "x".to_string(),
            }),
            index: Box::new(Expression::IntConst { value: 1 }),
        };

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_index_2() -> Result<(), SyntaxError> {
        let input = "x + y[1 + z]";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::FunctionCall {
            name: "__add".to_string(),
            args: vec![
                Expression::Variable {
                    name: "x".to_string(),
                },
                Expression::Index {
                    expr: Box::new(Expression::Variable {
                        name: "y".to_string(),
                    }),
                    index: Box::new(Expression::FunctionCall {
                        name: "__add".to_string(),
                        args: vec![
                            Expression::IntConst { value: 1 },
                            Expression::Variable {
                                name: "z".to_string(),
                            },
                        ],
                    }),
                },
            ],
        };

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_index_3() -> Result<(), SyntaxError> {
        let input = "(x + y)[1 + z]";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::Index {
            expr: Box::new(Expression::FunctionCall {
                name: "__add".to_string(),
                args: vec![
                    Expression::Variable {
                        name: "x".to_string(),
                    },
                    Expression::Variable {
                        name: "y".to_string(),
                    },
                ],
            }),
            index: Box::new(Expression::FunctionCall {
                name: "__add".to_string(),
                args: vec![
                    Expression::IntConst { value: 1 },
                    Expression::Variable {
                        name: "z".to_string(),
                    },
                ],
            }),
        };

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_if() -> Result<(), SyntaxError> {
        let input = "if x < 100 then 100 else 200 end";
        let ast = parse_as_expr(input)?;

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
            then: vec![Expression::IntConst { value: 100 }],
            else_: vec![Expression::IntConst { value: 200 }],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_if_multiple_exprs() -> Result<(), SyntaxError> {
        let input = "if x < 100 then y = 10; y else 200 end";
        let ast = parse_as_expr(input)?;

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
            then: vec![
                Expression::Assignment {
                    name: "y".to_string(),
                    value: Box::new(Expression::IntConst { value: 10 }),
                },
                Expression::Variable {
                    name: "y".to_string(),
                },
            ],
            else_: vec![Expression::IntConst { value: 200 }],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_while() -> Result<(), SyntaxError> {
        let input = "while x < 2 do y = 1; z = 2 end";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::While {
            condition: Box::new(Expression::FunctionCall {
                name: "__less".to_string(),
                args: vec![
                    Expression::Variable {
                        name: "x".to_string(),
                    },
                    Expression::IntConst { value: 2 },
                ],
            }),
            body: vec![
                Expression::Assignment {
                    name: "y".to_string(),
                    value: Box::new(Expression::IntConst { value: 1 }),
                },
                Expression::Assignment {
                    name: "z".to_string(),
                    value: Box::new(Expression::IntConst { value: 2 }),
                },
            ],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_unary() -> Result<(), SyntaxError> {
        let input = "-x";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::FunctionCall {
            name: "__neg".to_string(),
            args: vec![Expression::Variable {
                name: "x".to_string(),
            }],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_assignment() -> Result<(), SyntaxError> {
        let input = "x = 100";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::Assignment {
            name: "x".to_string(),
            value: Box::new(Expression::IntConst { value: 100 }),
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_or() -> Result<(), SyntaxError> {
        let input = "x || 100";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::FunctionCall {
            name: "__or".to_string(),
            args: vec![
                Expression::Variable {
                    name: "x".to_string(),
                },
                Expression::IntConst { value: 100 },
            ],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_and() -> Result<(), SyntaxError> {
        let input = "1 && 2";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::FunctionCall {
            name: "__and".to_string(),
            args: vec![
                Expression::IntConst { value: 1 },
                Expression::IntConst { value: 2 },
            ],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_cast() -> Result<(), SyntaxError> {
        let input = "rgba:[1,2,3,4]";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::TupleConst {
            tag: TupleTag::Rgba,
            values: vec![
                Expression::IntConst { value: 1 },
                Expression::IntConst { value: 2 },
                Expression::IntConst { value: 3 },
                Expression::IntConst { value: 4 },
            ],
        };
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_error() -> Result<(), SyntaxError> {
        let input = "\nx + +";
        if let Err(e) = parse_as_expr(input) {
            assert!(e.message.contains("Unexpected token"));
            assert!(e.message.contains("line: 2, column: 5"));
            Ok(())
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn parse_module() -> Result<(), SyntaxError> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        end";
        let ast = super::parse_module(input)?;

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
        Ok(())
    }

    #[test]
    fn parse_module_2() -> Result<(), SyntaxError> {
        let input = "filter red ()
            z = 1;
            rgbColor(z, 0, 0)
        end";
        let ast = super::parse_module(input)?;

        assert!(ast.filters.len() == 1);
        assert!(ast.filters[0].name == "red");
        assert!(ast.filters[0].exprs.len() == 2);

        let assign_expr = &ast.filters[0].exprs[0];
        let ast_ref = Expression::Assignment {
            name: "z".to_string(),
            value: Box::new(Expression::IntConst { value: 1 }),
        };
        assert_eq!(assign_expr, &ast_ref);
        Ok(())
    }
}
