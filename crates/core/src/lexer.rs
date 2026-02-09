use std::iter::Peekable;
use std::str::Chars;

use crate::token::{Kind as TokenKind, Token, lookup_identifier};

#[derive(Debug, Clone)]
pub struct Lexer<'src> {
  peekable: Peekable<Chars<'src>>,
}

impl<'src> Lexer<'src> {
  #[must_use]
  pub fn new(source: &'src str) -> Self {
    Self {
      peekable: source.chars().peekable(),
    }
  }

  fn next_token(&mut self) -> Option<Token> {
    self.skip_whitespace_and_comments();
    let ch = self.peek_char()?;
    match ch {
      '=' => self.eat(TokenKind::Assign),
      ',' => self.eat(TokenKind::Comma),
      ';' => self.eat(TokenKind::Semicolon),
      '(' => self.eat(TokenKind::LParen),
      ')' => self.eat(TokenKind::RParen),
      '[' => self.eat(TokenKind::LBracket),
      ']' => self.eat(TokenKind::RBracket),
      '0'..='9' => self.eat_number(),
      c if c.is_alphabetic() || c == '_' => self.eat_word(),
      _ => self.eat(TokenKind::Unknown),
    }
  }

  fn eat(&mut self, kind: TokenKind) -> Option<Token> {
    let c = self.peekable.next()?;
    Some(Token::new(kind, c))
  }

  fn eat_number(&mut self) -> Option<Token> {
    let mut buffer = String::new();
    while self.peek_char().is_some_and(|c| c.is_ascii_digit()) {
      buffer.push(self.peekable.next()?);
    }
    Some(Token::new(TokenKind::Int, buffer))
  }

  fn eat_word(&mut self) -> Option<Token> {
    let mut buffer = String::new();
    while self
      .peek_char()
      .is_some_and(|c| c.is_alphanumeric() || c == '_')
    {
      buffer.push(self.peekable.next()?);
    }
    let kind = lookup_identifier(buffer.as_str());
    Some(Token::new(kind, buffer))
  }

  fn peek_char(&mut self) -> Option<char> {
    self.peekable.peek().copied()
  }

  fn skip_whitespace_and_comments(&mut self) {
    loop {
      // skip whitespace
      while self.peek_char().is_some_and(char::is_whitespace) {
        self.peekable.next();
      }
      // skip // line comments
      if self.peek_char() == Some('/') {
        let mut clone = self.peekable.clone();
        clone.next();
        if clone.peek() == Some(&'/') {
          // consume until end of line
          while self.peek_char().is_some_and(|c| c != '\n') {
            self.peekable.next();
          }
          continue; // check for more whitespace/comments
        }
      }
      break;
    }
  }
}

impl Iterator for Lexer<'_> {
  type Item = Token;

  fn next(&mut self) -> Option<Self::Item> {
    self.next_token()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rstest::*;

  mod lexer {
    use super::*;

    mod fixtures {
      use super::*;

      #[fixture]
      pub fn basic_def() -> (&'static str, Vec<Token>) {
        (
          "def pred = Pr[const(0, 0), id(1,2)];",
          vec![
            Token::new(TokenKind::Def, "def"),
            Token::new(TokenKind::Ident, "pred"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Pr, "Pr"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Const, "const"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "0"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "0"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn composition() -> (&'static str, Vec<Token>) {
        (
          "def add = Pr[id(1,1), Cn[s, id(3,3)]];",
          vec![
            Token::new(TokenKind::Def, "def"),
            Token::new(TokenKind::Ident, "add"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Pr, "Pr"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Cn, "Cn"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Succ, "s"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "3"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "3"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn eval_stmt() -> (&'static str, Vec<Token>) {
        (
          "eval add(3, 2);",
          vec![
            Token::new(TokenKind::Eval, "eval"),
            Token::new(TokenKind::Ident, "add"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "3"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn minimization() -> (&'static str, Vec<Token>) {
        (
          "def isqrt = Mn[Cn[monus, id(1,2), Cn[mult, id(2,2), id(2,2)]]];",
          vec![
            Token::new(TokenKind::Def, "def"),
            Token::new(TokenKind::Ident, "isqrt"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Mn, "Mn"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Cn, "Cn"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Ident, "monus"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Cn, "Cn"),
            Token::new(TokenKind::LBracket, "["),
            Token::new(TokenKind::Ident, "mult"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Id, "id"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "2"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::RBracket, "]"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }

      #[fixture]
      pub fn comments() -> (&'static str, Vec<Token>) {
        (
          "// this is a comment\ndef f = const(1, 0);",
          vec![
            Token::new(TokenKind::Def, "def"),
            Token::new(TokenKind::Ident, "f"),
            Token::new(TokenKind::Assign, "="),
            Token::new(TokenKind::Const, "const"),
            Token::new(TokenKind::LParen, "("),
            Token::new(TokenKind::Int, "1"),
            Token::new(TokenKind::Comma, ","),
            Token::new(TokenKind::Int, "0"),
            Token::new(TokenKind::RParen, ")"),
            Token::new(TokenKind::Semicolon, ";"),
          ],
        )
      }
    }

    #[rstest]
    #[case::basic_def(fixtures::basic_def())]
    #[case::composition(fixtures::composition())]
    #[case::eval_stmt(fixtures::eval_stmt())]
    #[case::minimization(fixtures::minimization())]
    #[case::comments(fixtures::comments())]
    fn next_token(#[case] input: (&'static str, Vec<Token>)) {
      let (input, expected_tokens) = input;
      let mut lexer = Lexer::new(input);

      for expected in expected_tokens {
        let actual = lexer.next_token().unwrap();
        assert_eq!(actual.kind, expected.kind, "TokenKind mismatch");
        assert_eq!(actual.literal, expected.literal, "Token literal mismatch");
      }
    }
  }
}
