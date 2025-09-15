/*
Lexer based on Crafting Interpreters:
https://craftinginterpreters.com/scanning-on-demand.html#a-token-at-a-time
*/

#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub item: T,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind<'a> {
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

pub type Token<'a> = Spanned<TokenKind<'a>>;

pub struct Lexer<'a> {
    input: &'a str,
    pos: usize,
    start_pos: usize,
    line: usize,
    column: usize,
    start_column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input,
            pos: 0,
            start_pos: 0,
            line: 1,
            column: 1,
            start_column: 1,
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
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        x
    }

    fn match_next(&mut self, expected: char) -> bool {
        if let Some(next) = self.peek() {
            if next == expected {
                self.pos += expected.len_utf8();
                self.column += 1;
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
                    self.advance();
                }
            } else if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn spanned(&self, item: TokenKind<'a>) -> Token<'a> {
        Token {
            item,
            line: self.line,
            column: self.start_column,
        }
    }

    fn lex_number(&mut self) -> Token<'a> {
        // Integral part.
        while let Some(c) = self.peek() {
            if !c.is_digit(10) {
                break;
            }
            self.advance();
        }

        // Fractional part.
        if self.peek() == Some('.') {
            self.advance();

            while let Some(c) = self.peek() {
                if !c.is_digit(10) {
                    break;
                }
                self.advance();
            }
        }

        self.spanned(TokenKind::NumberLit(&self.input[self.start_pos..self.pos]))
    }

    fn lex_identifier(&mut self) -> Token<'a> {
        while let Some(c) = self.peek() {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.advance();
        }

        let text = &self.input[self.start_pos..self.pos];

        let kind = match text {
            "filter" => TokenKind::Filter,

            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "end" => TokenKind::End,
            "while" => TokenKind::While,
            "do" => TokenKind::Do,
            "for" => TokenKind::For,
            "xor" => TokenKind::Xor,
            _ => TokenKind::Ident(&self.input[self.start_pos..self.pos]),
        };
        self.spanned(kind)
    }

    pub fn next_token(&mut self) -> Result<Option<Token<'a>>, crate::SyntaxError> {
        self.skip_whitespace();

        self.start_pos = self.pos;
        self.start_column = self.column;

        if let Some(c) = self.advance() {
            let kind = match c {
                '(' => TokenKind::LParen,
                ')' => TokenKind::RParen,
                '[' => TokenKind::LBracket,
                ']' => TokenKind::RBracket,
                ';' => TokenKind::Semicolon,
                ',' => TokenKind::Comma,
                ':' => TokenKind::Colon,
                '+' => TokenKind::Plus,
                '-' => TokenKind::Minus,
                '*' => TokenKind::Star,
                '/' => TokenKind::Slash,
                '%' => TokenKind::Percent,
                '^' => TokenKind::Caret,

                '!' => {
                    if self.match_next('=') {
                        TokenKind::NotEqual
                    } else {
                        TokenKind::Bang
                    }
                }

                '=' => {
                    if self.match_next('=') {
                        TokenKind::Equal
                    } else {
                        TokenKind::Assign
                    }
                }

                '<' => {
                    if self.match_next('=') {
                        TokenKind::LessEqual
                    } else {
                        TokenKind::Less
                    }
                }

                '>' => {
                    if self.match_next('=') {
                        TokenKind::GreaterEqual
                    } else {
                        TokenKind::Greater
                    }
                }

                '|' => {
                    if self.match_next('|') {
                        TokenKind::Or
                    } else {
                        return Err(crate::SyntaxError::with_pos(
                            "Unexpected '|' without '||'",
                            self.line,
                            self.start_column,
                        ));
                    }
                }

                '&' => {
                    if self.match_next('&') {
                        TokenKind::And
                    } else {
                        return Err(crate::SyntaxError::with_pos(
                            "Unexpected '&' without '&&'",
                            self.line,
                            self.start_column,
                        ));
                    }
                }

                x if x.is_digit(10) => return Ok(Some(self.lex_number())),

                x if x.is_alphabetic() => return Ok(Some(self.lex_identifier())),

                _ => {
                    return Err(crate::SyntaxError::with_pos(
                        format!("Unexpected character {:?}", c),
                        self.line,
                        self.start_column,
                    ));
                }
            };
            Ok(Some(self.spanned(kind)))
        } else {
            Ok(None)
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token<'a>, crate::SyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Ok(Some(t)) => Some(Ok(t)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize<'a>(input: &'a str) -> Vec<TokenKind<'a>> {
        let x = Lexer::new(input);
        x.map(|s| s.unwrap().item).collect()
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

    #[test]
    fn test_span() {
        let s = "a\nb\n c";
        let lexer = Lexer::new(s);
        let tokens: Vec<_> = lexer.map(|r| r.unwrap()).collect();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].line, 1);
        assert_eq!(tokens[0].column, 1);
        assert_eq!(tokens[1].line, 2);
        assert_eq!(tokens[1].column, 1);
        assert_eq!(tokens[2].line, 3);
        assert_eq!(tokens[2].column, 2);
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
        check_lexer!("  (  )  ", [TokenKind::LParen, TokenKind::RParen]);
    }

    #[test]
    fn two_chars() {
        check_lexer!(
            "= == < <= > >= ! !=",
            [
                TokenKind::Assign,
                TokenKind::Equal,
                TokenKind::Less,
                TokenKind::LessEqual,
                TokenKind::Greater,
                TokenKind::GreaterEqual,
                TokenKind::Bang,
                TokenKind::NotEqual,
            ]
        );
    }

    #[test]
    fn integer_literal() {
        check_lexer!("1234", [TokenKind::NumberLit("1234")]);
    }

    #[test]
    fn float_literal() {
        check_lexer!("1234.5678", [TokenKind::NumberLit("1234.5678")]);
    }

    #[test]
    fn identifier() {
        check_lexer!(
            "filter hello",
            [TokenKind::Filter, TokenKind::Ident("hello")]
        );
    }

    #[test]
    fn filter() {
        check_lexer!(
            "filter red ()
                rgbColor(1, 0, 0)
            end",
            [
                TokenKind::Filter,
                TokenKind::Ident("red"),
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::Ident("rgbColor"),
                TokenKind::LParen,
                TokenKind::NumberLit("1"),
                TokenKind::Comma,
                TokenKind::NumberLit("0"),
                TokenKind::Comma,
                TokenKind::NumberLit("0"),
                TokenKind::RParen,
                TokenKind::End,
            ]
        );
    }

    #[test]
    fn or() {
        check_lexer!(
            "(a + b) || c",
            [
                TokenKind::LParen,
                TokenKind::Ident("a"),
                TokenKind::Plus,
                TokenKind::Ident("b"),
                TokenKind::RParen,
                TokenKind::Or,
                TokenKind::Ident("c"),
            ]
        );
    }

    #[test]
    fn comment_start_of_line() {
        check_lexer!(
            "# a comment
            filter red ()",
            [
                TokenKind::Filter,
                TokenKind::Ident("red"),
                TokenKind::LParen,
                TokenKind::RParen
            ]
        )
    }

    #[test]
    fn comment_rest_of_line() {
        check_lexer!(
            "filter red () # comment",
            [
                TokenKind::Filter,
                TokenKind::Ident("red"),
                TokenKind::LParen,
                TokenKind::RParen
            ]
        )
    }
}
