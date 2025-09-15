/*
Lexer based on Crafting Interpreters:
https://craftinginterpreters.com/scanning-on-demand.html#a-token-at-a-time
*/

#![allow(dead_code)]

use core::panic;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token<'a> {
    Ident(&'a str),
    StringLit(&'a str),
    NumberLit(&'a str),

    Range,
    Filter,

    FloatType,
    IntType,
    BoolType,
    ColorType,
    GradientType,
    CurveType,
    ImageType,

    If,
    Then,
    Else,
    End,
    While,
    Do,
    For,

    Or,
    And,
    Xor,
    Equal,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    NotEqual,

    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,

    Colon,
    Convert,
    Bang,
    Assign,

    Semicolon,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
}

pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
    start_pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input,
            pos: 0,
            start_pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn peek_next(&self) -> Option<char> {
        if let Some(p) = self.peek() {
            self.input[self.pos + p.len_utf8()..].chars().next()
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        let x = self.peek();
        if let Some(c) = x {
            self.pos += c.len_utf8();
        }
        x
    }

    fn match_next(&mut self, expected: char) -> bool {
        if let Some(next) = self.peek() {
            if next == expected {
                self.pos += expected.len_utf8();
                return true;
            }
        }
        false
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c == '#' {
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.pos += c.len_utf8();
                }
            } else if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn lex_number(&mut self) -> Option<Token<'a>> {
        // Integral part.
        while let Some(c) = self.peek() {
            if !c.is_digit(10) {
                break;
            }
            self.pos += c.len_utf8();
        }

        // Fractional part.
        if self.peek() == Some('.') {
            self.advance();

            while let Some(c) = self.peek() {
                if !c.is_digit(10) {
                    break;
                }
                self.pos += c.len_utf8();
            }
        }

        Some(Token::NumberLit(&self.input[self.start_pos..self.pos]))
    }

    fn lex_identifier(&mut self) -> Option<Token<'a>> {
        while let Some(c) = self.peek() {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.pos += c.len_utf8();
        }

        let text = &self.input[self.start_pos..self.pos];

        match text {
            "filter" => Some(Token::Filter),

            "if" => Some(Token::If),
            "then" => Some(Token::Then),
            "else" => Some(Token::Else),
            "end" => Some(Token::End),
            "while" => Some(Token::While),
            "do" => Some(Token::Do),
            "for" => Some(Token::For),
            "xor" => Some(Token::Xor),
            _ => Some(Token::Ident(&self.input[self.start_pos..self.pos])),
        }
    }

    pub fn next_token(&mut self) -> Option<Token<'a>> {
        self.skip_whitespace();

        self.start_pos = self.pos;

        if let Some(c) = self.advance() {
            match c {
                '(' => Some(Token::LParen),
                ')' => Some(Token::RParen),
                '[' => Some(Token::LBracket),
                ']' => Some(Token::RBracket),
                ';' => Some(Token::Semicolon),
                ',' => Some(Token::Comma),
                ':' => Some(Token::Colon),
                '+' => Some(Token::Plus),
                '-' => Some(Token::Minus),
                '*' => Some(Token::Star),
                '/' => Some(Token::Slash),
                '%' => Some(Token::Percent),
                '^' => Some(Token::Caret),

                '!' => {
                    if self.match_next('=') {
                        Some(Token::NotEqual)
                    } else {
                        Some(Token::Bang)
                    }
                }

                '=' => {
                    if self.match_next('=') {
                        Some(Token::Equal)
                    } else {
                        Some(Token::Assign)
                    }
                }

                '<' => {
                    if self.match_next('=') {
                        Some(Token::LessEqual)
                    } else {
                        Some(Token::Less)
                    }
                }

                '>' => {
                    if self.match_next('=') {
                        Some(Token::GreaterEqual)
                    } else {
                        Some(Token::Greater)
                    }
                }

                '|' => {
                    if self.match_next('|') {
                        Some(Token::Or)
                    } else {
                        panic!("unexpected | without ||");
                    }
                }

                '&' => {
                    if self.match_next('&') {
                        Some(Token::And)
                    } else {
                        panic!("unexpected & without &&");
                    }
                }

                x if x.is_digit(10) => self.lex_number(),

                x if x.is_alphabetic() => self.lex_identifier(),

                _ => {
                    panic!("lexer not implemented {:?}", c);
                }
            }
        } else {
            None
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize<'a>(input: &'a str) -> Vec<Token<'a>> {
        let x = Lexer::new(input);
        x.collect()
    }

    #[test]
    fn peek() {
        let s = "abc";
        let mut lexer = Lexer::new(s);

        assert!(lexer.peek() == Some('a'));
        assert!(lexer.peek_next() == Some('b'));
        lexer.advance();
        assert!(lexer.peek() == Some('b'));
        assert!(lexer.peek_next() == Some('c'));
        lexer.advance();
        assert!(lexer.peek() == Some('c'));
        assert!(lexer.peek_next() == None);
    }

    macro_rules! check_lexer {
    ($input:expr, [ $($token:expr),* $(,)? ]) => {
        assert_eq!(
            tokenize($input),
            vec![ $($token),* ]
        )
        };
    }

    #[test]
    fn simple() {
        check_lexer!("  (  )  ", [Token::LParen, Token::RParen]);
    }

    #[test]
    fn two_chars() {
        check_lexer!(
            "= == < <= > >= ! !=",
            [
                Token::Assign,
                Token::Equal,
                Token::Less,
                Token::LessEqual,
                Token::Greater,
                Token::GreaterEqual,
                Token::Bang,
                Token::NotEqual,
            ]
        );
    }

    #[test]
    fn integer_literal() {
        check_lexer!("1234", [Token::NumberLit("1234")]);
    }

    #[test]
    fn float_literal() {
        check_lexer!("1234.5678", [Token::NumberLit("1234.5678")]);
    }

    #[test]
    fn identifier() {
        check_lexer!("filter hello", [Token::Filter, Token::Ident("hello")]);
    }

    #[test]
    fn filter() {
        check_lexer!(
            "filter red ()
                rgbColor(1, 0, 0)
            end",
            [
                Token::Filter,
                Token::Ident("red"),
                Token::LParen,
                Token::RParen,
                Token::Ident("rgbColor"),
                Token::LParen,
                Token::NumberLit("1"),
                Token::Comma,
                Token::NumberLit("0"),
                Token::Comma,
                Token::NumberLit("0"),
                Token::RParen,
                Token::End,
            ]
        );
    }

    #[test]
    fn or() {
        check_lexer!(
            "(a + b) || c",
            [
                Token::LParen,
                Token::Ident("a"),
                Token::Plus,
                Token::Ident("b"),
                Token::RParen,
                Token::Or,
                Token::Ident("c"),
            ]
        );
    }

    #[test]
    fn comment_start_of_line() {
        check_lexer!(
            "# a comment
            filter red ()",
            [
                Token::Filter,
                Token::Ident("red"),
                Token::LParen,
                Token::RParen
            ]
        )
    }

    #[test]
    fn comment_rest_of_line() {
        check_lexer!(
            "filter red () # comment",
            [
                Token::Filter,
                Token::Ident("red"),
                Token::LParen,
                Token::RParen
            ]
        )
    }
}
