#![allow(dead_code)]

use crate::lexer::{self, Spanned, TokenKind};
use crate::sema::Type;
use crate::{SyntaxError, TypeError};

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
        lexer::TokenKind::Caret => Some(OpInfo {
            precedence: 7,
            associativity: Associativity::Right,
            name: "__pow",
        }),
        _ => None,
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    // From exprtree.h.
    // Types are determined during semantic analysis, not parsing (like clang). The alternatives would have been a completely separate
    // AST tree with typed expressions, or external tables keyed by node id (like rustc).
    IntConst {
        value: i64,
        ty: Type,
    },
    FloatConst {
        value: f32,
        ty: Type,
    },
    TupleConst {
        tag: TupleTag,
        values: Vec<Expression>,
        ty: Type,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
        ty: Type,
    },
    Variable {
        name: String,
        ty: Type,
    },
    If {
        condition: Box<Expression>,
        then: Vec<Expression>,
        else_: Vec<Expression>,
        ty: Type,
    },
    While {
        condition: Box<Expression>,
        body: Vec<Expression>,
        ty: Type,
    },
    Assignment {
        name: String,
        value: Box<Expression>,
        ty: Type,
    },
    Index {
        expr: Box<Expression>,
        index: Box<Expression>,
        ty: Type,
    },
    Cast {
        tag: TupleTag,
        expr: Box<Expression>,
        ty: Type,
    },
}

impl Expression {
    pub fn ty(&self) -> Type {
        match self {
            Self::IntConst { ty, .. } => ty.clone(),
            Self::FloatConst { ty, .. } => ty.clone(),
            Self::TupleConst { ty, .. } => ty.clone(),
            Self::FunctionCall { ty, .. } => ty.clone(),
            Self::Variable { ty, .. } => ty.clone(),
            Self::If { ty, .. } => ty.clone(),
            Self::While { ty, .. } => ty.clone(),
            Self::Assignment { ty, .. } => ty.clone(),
            Self::Index { ty, .. } => ty.clone(),
            Self::Cast { ty, .. } => ty.clone(),
        }
    }

    pub fn int_(value: i64) -> Self {
        Self::IntConst {
            value,
            ty: Type::Int,
        }
    }

    pub fn float_(value: f32) -> Self {
        Self::FloatConst {
            value,
            ty: Type::Tuple(1),
        }
    }

    pub fn tuple_(tag: TupleTag, values: Vec<Self>) -> Self {
        let len = values.len();
        Self::TupleConst {
            tag,
            values,
            ty: Type::Tuple(len),
        }
    }

    pub fn function_call_(name: impl Into<String>, args: Vec<Self>) -> Self {
        Self::FunctionCall {
            name: name.into(),
            args,
            ty: Type::Unknown,
        }
    }

    pub fn variable_(name: impl Into<String>) -> Self {
        Self::Variable {
            name: name.into(),
            ty: Type::Unknown,
        }
    }

    pub fn if_(condition: Self, then: Vec<Self>, else_: Vec<Self>) -> Self {
        Self::If {
            condition: Box::new(condition),
            then,
            else_,
            ty: Type::Unknown,
        }
    }

    pub fn while_(condition: Self, body: Vec<Self>) -> Self {
        Self::While {
            condition: Box::new(condition),
            body,
            // Always returns zero according to the language spec.
            ty: Type::Int,
        }
    }

    pub fn assignment_(name: impl Into<String>, value: Self) -> Self {
        Self::Assignment {
            name: name.into(),
            value: Box::new(value),
            ty: Type::Unknown,
        }
    }

    pub fn index_(expr: Self, index: Self) -> Self {
        Self::Index {
            expr: Box::new(expr),
            index: Box::new(index),
            ty: Type::Unknown,
        }
    }

    pub fn cast_(tag: TupleTag, expr: Self) -> Self {
        Self::Cast {
            tag,
            expr: Box::new(expr),
            ty: Type::Unknown,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Filter {
    pub name: String,
    pub exprs: Vec<Expression>,
}

impl Filter {
    fn ty(&self) -> Type {
        if let Some(last_expr) = self.exprs.last() {
            last_expr.ty()
        } else {
            Type::Unknown
        }
    }
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

    fn peek_token_kind(&mut self) -> Result<Option<lexer::TokenKind<'a>>, SyntaxError> {
        match self.tokens.peek() {
            Some(Ok(t)) => Ok(Some(t.item)),
            Some(Err(e)) => Err(SyntaxError::with_pos(e.message.clone(), e.line, e.column)),
            None => Ok(None),
        }
    }

    fn expect(&mut self, expected: lexer::TokenKind) -> Result<(), SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(Ok(t)) if t.item == expected => Ok(()),
            Some(Ok(t)) => Err(SyntaxError::with_pos(
                format!("Expected {:?}, got {:?}", expected, t.item),
                t.line,
                t.column,
            )),
            Some(Err(e)) => Err(e),
            None => Err(SyntaxError::with_pos("Unexpected end of input", 0, 0)),
        }
    }

    fn expect_done(&mut self) -> Result<(), SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(Ok(t)) => Err(SyntaxError::with_pos(
                format!("Expected end of input, got {:?}", t.item),
                t.line,
                t.column,
            )),
            Some(Err(e)) => Err(e),
            None => Ok(()),
        }
    }

    fn consume_ident(&mut self) -> Result<String, SyntaxError> {
        let t = self.tokens.next();
        match t {
            Some(Ok(lexer::Token {
                item: TokenKind::Ident(s),
                ..
            })) => Ok(s.to_string()),
            Some(Ok(t)) => Err(SyntaxError::with_pos(
                format!("Expected identifier, got {:?}", t.item),
                t.line,
                t.column,
            )),
            Some(Err(e)) => Err(e),
            None => Err(SyntaxError::with_pos("Unexpected end of input", 0, 0)),
        }
    }

    fn parse_atom(&mut self) -> Result<Expression, SyntaxError> {
        let mut expr = match self.peek_token_kind()? {
            Some(TokenKind::Minus) => {
                self.tokens.next();
                let operand = self.parse_expression(100)?;
                let args = vec![operand];
                Ok(Expression::function_call_("__neg".to_string(), args))
            }
            Some(TokenKind::LParen) => {
                self.expect(lexer::TokenKind::LParen)?;
                // Start parsing a subexpression inside parentheses.
                let subexpr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::RParen)?;
                Ok(subexpr)
            }
            Some(TokenKind::Ident(_)) => match self.tokens.next() {
                Some(Ok(lexer::Token {
                    item: TokenKind::Ident(s),
                    ..
                })) => Ok(Expression::variable_(s.to_string())),
                Some(Ok(t)) => Err(SyntaxError::with_pos(
                    format!("Expected identifier, got {:?}", t.item),
                    t.line,
                    t.column,
                )),
                Some(Err(e)) => Err(e),
                None => Err(SyntaxError::with_pos("Unexpected end of input", 0, 0)),
            },
            Some(TokenKind::NumberLit(_)) => match self.tokens.next() {
                Some(Ok(lexer::Token {
                    item: TokenKind::NumberLit(s),
                    line,
                    column,
                })) => {
                    if s.contains('.') {
                        match s.parse::<f32>() {
                            Ok(x) => Ok(Expression::float_(x)),
                            Err(_) => Err(SyntaxError::with_pos(
                                format!("Invalid float literal: {}", s),
                                line,
                                column,
                            )),
                        }
                    } else {
                        match s.parse::<i64>() {
                            Ok(x) => Ok(Expression::int_(x)),
                            Err(_) => Err(SyntaxError::with_pos(
                                format!("Invalid int literal: {}", s),
                                line,
                                column,
                            )),
                        }
                    }
                }
                Some(Ok(t)) => Err(SyntaxError::with_pos(
                    format!("Expected number literal, got {:?}", t.item),
                    t.line,
                    t.column,
                )),
                Some(Err(e)) => Err(e),
                None => Err(SyntaxError::with_pos("Unexpected end of input", 0, 0)),
            },
            Some(TokenKind::If) => {
                self.expect(lexer::TokenKind::If)?;
                let condition_expr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::Then)?;
                let then_expr = self.parse_expr_block()?;

                let else_expr = if self.peek_token_kind()? == Some(lexer::TokenKind::Else) {
                    self.expect(lexer::TokenKind::Else)?;
                    self.parse_expr_block()?
                } else {
                    vec![]
                };

                self.expect(lexer::TokenKind::End)?;

                Ok(Expression::if_(condition_expr, then_expr, else_expr))
            }
            Some(TokenKind::While) => {
                self.expect(lexer::TokenKind::While)?;
                let condition_expr = self.parse_expression(1)?;
                self.expect(lexer::TokenKind::Do)?;
                let body_expr = self.parse_expr_block()?;
                self.expect(lexer::TokenKind::End)?;

                Ok(Expression::while_(condition_expr, body_expr))
            }
            None => Err(SyntaxError::with_pos(
                "Unexpected end of input while parsing expression",
                0,
                0,
            )),
            _ => match self.tokens.peek() {
                Some(Ok(t)) => Err(SyntaxError::with_pos(
                    format!("Unexpected token in expression: {:?}", t.item),
                    t.line,
                    t.column,
                )),
                Some(Err(e)) => Err(SyntaxError::with_pos(e.message.clone(), e.line, e.column)),
                None => Err(SyntaxError::with_pos("Unexpected end of input", 0, 0)),
            },
        }?;

        // We have a choice between handling postfix operators ([], function calls, etc..)
        // here or in parse_expression. Apparently, the most common option is to do it here
        // so we don't have to pollute precedence tables with extra tokens. This works
        // because postfix operators have the highest precedence. Note that
        // https://pdubroy.github.io/200andchange/precedence-climbing/ does it the other way
        // and puts function call handling in parse_expression.
        // Sadly we need to clone() the peeked token here so we can release the (mut) borrow
        // immediately and call other mut methods on the token iterator.
        loop {
            match self.tokens.peek().cloned() {
                Some(Ok(Spanned {
                    item: lexer::TokenKind::LParen,
                    line,
                    column,
                })) => {
                    if let Expression::Variable { ref name, .. } = expr {
                        self.tokens.next(); // consume '('
                        let mut args = Vec::new();
                        if self.peek_token_kind()? != Some(lexer::TokenKind::RParen) {
                            loop {
                                args.push(self.parse_expression(1)?);
                                if self.peek_token_kind()? == Some(lexer::TokenKind::Comma) {
                                    self.tokens.next(); // consume ','
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(lexer::TokenKind::RParen)?;
                        expr = Expression::function_call_(name.clone(), args);
                    } else {
                        return Err(SyntaxError::with_pos(
                            "Only identifiers can be called as functions",
                            line,
                            column,
                        ));
                    }
                }
                Some(Ok(Spanned {
                    item: lexer::TokenKind::LBracket,
                    ..
                })) => {
                    self.expect(lexer::TokenKind::LBracket)?;
                    let index = self.parse_expression(1)?;
                    self.expect(lexer::TokenKind::RBracket)?;
                    expr = Expression::index_(expr, index);
                }
                Some(Ok(Spanned {
                    item: lexer::TokenKind::Colon,
                    line,
                    column,
                })) => {
                    self.expect(lexer::TokenKind::Colon)?;
                    // I guess we could make special tokens for the tuple tags since they're a fixed set, but for now
                    // they're initially parsed as variables and here we turn them into literal/cast AST nodes.
                    if let Expression::Variable { ref name, .. } = expr {
                        let tag = match name.as_str() {
                            "rgba" => TupleTag::Rgba,
                            "ri" => TupleTag::Ri,
                            "xy" => TupleTag::Xy,
                            "quat" => TupleTag::Quat,
                            _ => panic!("unknown tuple tag {:?}", name),
                        };

                        // Handle either tuple literals rgba:[1,2,3,4] or casts ri:xy.
                        if self.peek_token_kind()? == Some(lexer::TokenKind::LBracket) {
                            self.expect(lexer::TokenKind::LBracket)?;
                            let mut args = Vec::new();
                            while self.peek_token_kind()? != Some(lexer::TokenKind::RBracket) {
                                args.push(self.parse_expression(1)?);
                                if self.peek_token_kind()? == Some(lexer::TokenKind::Comma) {
                                    self.tokens.next(); // consume ','
                                } else {
                                    break;
                                }
                            }
                            self.expect(lexer::TokenKind::RBracket)?;

                            expr = Expression::tuple_(tag, args)
                        } else {
                            let rhs_expr = self.parse_expression(1)?;
                            expr = Expression::cast_(tag, rhs_expr)
                        }
                    } else {
                        return Err(SyntaxError::with_pos(
                            format!("Invalid cast: lhs is not a valid tuple tag: {:?}", expr),
                            line,
                            column,
                        ));
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
            let peek_kind = match self.peek_token_kind()? {
                Some(kind) => kind,
                None => break,
            };

            let op_info = match get_op_info(&peek_kind) {
                Some(info) => info,
                None => break,
            };

            if op_info.precedence < min_precedence {
                break;
            }

            // Consume the current operator token.
            let op_token = self.tokens.next().unwrap()?;

            let next_min_precedence = match op_info.associativity {
                Associativity::Left => op_info.precedence + 1,
                Associativity::Right => op_info.precedence,
            };

            let atom_rhs = self.parse_expression(next_min_precedence)?;

            // Rewrite assignments into the proper AST node.
            if op_info.name == "__assign" {
                // Check that the LHS is a variable (ie lvalue).
                if let Expression::Variable { name, .. } = atom_lhs {
                    return Ok(Expression::assignment_(name, atom_rhs));
                } else {
                    return Err(SyntaxError::with_pos(
                        format!("Invalid assignment: lhs is not a variable: {:?}", atom_lhs),
                        op_token.line,
                        op_token.column,
                    ));
                }
            } else {
                atom_lhs =
                    Expression::function_call_(op_info.name.to_string(), vec![atom_lhs, atom_rhs]);
            }
        }

        return Ok(atom_lhs);
    }

    // Parse a block of expressions separated by semicolon (for filters, if/while, etc..)
    pub fn parse_expr_block(&mut self) -> Result<Vec<Expression>, SyntaxError> {
        let mut expressions: Vec<Expression> = Vec::new();

        loop {
            if matches!(self.peek_token_kind()?, Some(lexer::TokenKind::End)) {
                break;
            }

            let expr = self.parse_expression(1)?;
            expressions.push(expr);

            if matches!(self.peek_token_kind()?, Some(lexer::TokenKind::Semicolon)) {
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

    use super::Expression as E;

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

        let ast_ref = E::function_call_(
            "__add",
            vec![
                E::int_(1),
                E::function_call_(
                    "__div",
                    vec![
                        E::function_call_("__mul", vec![E::int_(2), E::int_(3)]),
                        E::int_(4),
                    ],
                ),
            ],
        );

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expression_with_variable() -> Result<(), SyntaxError> {
        let input = "x + 100";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_("__add", vec![E::variable_("x"), E::int_(100)]);

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_function_call() -> Result<(), SyntaxError> {
        let input = "fn(100, x)";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_("fn", vec![E::int_(100), E::variable_("x")]);

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_index() -> Result<(), SyntaxError> {
        let input = "x[1]";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::index_(E::variable_("x"), E::int_(1));

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_index_2() -> Result<(), SyntaxError> {
        let input = "x + y[1 + z]";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_(
            "__add",
            vec![
                E::variable_("x"),
                E::index_(
                    E::variable_("y"),
                    E::function_call_("__add", vec![E::int_(1), E::variable_("z")]),
                ),
            ],
        );

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_index_3() -> Result<(), SyntaxError> {
        let input = "(x + y)[1 + z]";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::index_(
            E::function_call_("__add", vec![E::variable_("x"), E::variable_("y")]),
            E::function_call_("__add", vec![E::int_(1), E::variable_("z")]),
        );

        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_if() -> Result<(), SyntaxError> {
        let input = "if x < 100 then 100 else 200 end";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::if_(
            E::function_call_("__less", vec![E::variable_("x"), E::int_(100)]),
            vec![E::int_(100)],
            vec![E::int_(200)],
        );
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_if_multiple_exprs() -> Result<(), SyntaxError> {
        let input = "if x < 100 then y = 10; y else 200 end";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::if_(
            E::function_call_("__less", vec![E::variable_("x"), E::int_(100)]),
            vec![E::assignment_("y", E::int_(10)), E::variable_("y")],
            vec![E::int_(200)],
        );
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_while() -> Result<(), SyntaxError> {
        let input = "while x < 2 do y = 1; z = 2 end";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::while_(
            E::function_call_("__less", vec![E::variable_("x"), E::int_(2)]),
            vec![
                E::assignment_("y", E::int_(1)),
                E::assignment_("z", E::int_(2)),
            ],
        );
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_unary() -> Result<(), SyntaxError> {
        let input = "-x";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_("__neg", vec![E::variable_("x")]);
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_assignment() -> Result<(), SyntaxError> {
        let input = "x = 100";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::assignment_("x", E::int_(100));
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_assignment_error() -> Result<(), SyntaxError> {
        let input = "1 = 2";
        if let Err(e) = parse_as_expr(input) {
            assert!(
                e.message
                    .contains("Invalid assignment: lhs is not a variable")
            );
            assert_eq!(e.line, 1);
            assert_eq!(e.column, 3);
            Ok(())
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn parse_expr_or() -> Result<(), SyntaxError> {
        let input = "x || 100";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_("__or", vec![E::variable_("x"), E::int_(100)]);
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_and() -> Result<(), SyntaxError> {
        let input = "1 && 2";
        let ast = parse_as_expr(input)?;

        let ast_ref = E::function_call_("__and", vec![E::int_(1), E::int_(2)]);
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_cast() -> Result<(), SyntaxError> {
        let input = "rgba:[1,2,3,4]";
        let ast = parse_as_expr(input)?;

        let ast_ref = Expression::tuple_(
            TupleTag::Rgba,
            vec![E::int_(1), E::int_(2), E::int_(3), E::int_(4)],
        );
        assert_eq!(ast, ast_ref);
        Ok(())
    }

    #[test]
    fn parse_expr_error() -> Result<(), SyntaxError> {
        let input = "
x + +";
        if let Err(e) = parse_as_expr(input) {
            assert!(e.message.contains("Unexpected token"));
            assert_eq!(e.line, 2);
            assert_eq!(e.column, 5);
            Ok(())
        } else {
            panic!("expected the parser to fail");
        }
    }

    #[test]
    fn parse_fn_call_error() -> Result<(), SyntaxError> {
        let input = "1(1, 2, 3)";
        if let Err(e) = parse_as_expr(input) {
            dbg!(&e);
            assert!(
                e.message
                    .contains("Only identifiers can be called as functions")
            );
            assert_eq!(e.line, 1);
            assert_eq!(e.column, 2);
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
                exprs: vec![E::function_call_(
                    "rgbColor",
                    vec![E::int_(1), E::int_(0), E::int_(0)],
                )],
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
        let ast_ref = E::assignment_("z", E::int_(1));
        assert_eq!(assign_expr, &ast_ref);
        Ok(())
    }

    #[test]
    fn parse_module_error_eof() -> Result<(), SyntaxError> {
        let input = "filter red ()
            rgbColor(1, 0, 0)
        ";
        if let Err(e) = super::parse_module(input) {
            assert!(e.message.contains("Unexpected end of input"));
            assert_eq!(e.line, 0);
            assert_eq!(e.column, 0);
            Ok(())
        } else {
            panic!("expected the parser to fail");
        }
    }
}
