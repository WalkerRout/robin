use std::collections::VecDeque;
use std::str::Chars;

use crate::token::{Kind as TokenKind, Token, lookup_identifier};

const LOOKAHEAD: usize = 2;

fn is_digit(c: char) -> bool {
  c.is_ascii_digit()
}

fn is_word_start(c: char) -> bool {
  c.is_alphabetic() || c == '_'
}

fn is_word_continue(c: char) -> bool {
  c.is_alphanumeric() || c == '_'
}

#[derive(Debug, Clone)]
pub struct Lexer<'src> {
  chars: Chars<'src>,
  buf: VecDeque<char>,
}

impl<'src> Lexer<'src> {
  #[must_use]
  pub fn new(source: &'src str) -> Self {
    let mut chars = source.chars();
    let buf = (&mut chars).take(LOOKAHEAD).collect();
    Self { chars, buf }
  }

  pub fn next_token(&mut self) -> Option<Token> {
    skip_whitespace_and_comments(self);
    let ch = self.peek()?;
    match ch {
      '=' => eat(self, TokenKind::Assign),
      ',' => eat(self, TokenKind::Comma),
      ';' => eat(self, TokenKind::Semicolon),
      '(' => eat(self, TokenKind::LParen),
      ')' => eat(self, TokenKind::RParen),
      '[' => eat(self, TokenKind::LBracket),
      ']' => eat(self, TokenKind::RBracket),
      c if is_digit(c) => eat_while(self, is_digit, TokenKind::Int),
      c if is_word_start(c) => eat_word(self),
      _ => eat(self, TokenKind::Unknown),
    }
  }

  fn peek(&self) -> Option<char> {
    self.buf.front().copied()
  }

  fn peek2(&self) -> Option<char> {
    self.buf.get(1).copied()
  }

  fn advance(&mut self) -> Option<char> {
    let c = self.buf.pop_front()?;
    if let Some(next) = self.chars.next() {
      self.buf.push_back(next);
    }
    Some(c)
  }
}

fn eat(lexer: &mut Lexer, kind: TokenKind) -> Option<Token> {
  let c = lexer.advance()?;
  Some(Token::new(kind, c))
}

fn eat_while(lexer: &mut Lexer, pred: impl Fn(char) -> bool, kind: TokenKind) -> Option<Token> {
  let literal = take_while(lexer, pred);
  (!literal.is_empty()).then(|| Token::new(kind, literal))
}

fn eat_word(lexer: &mut Lexer) -> Option<Token> {
  let literal = take_while(lexer, is_word_continue);
  let kind = lookup_identifier(&literal);
  Some(Token::new(kind, literal))
}

fn take_while(lexer: &mut Lexer, pred: impl Fn(char) -> bool) -> String {
  let mut buffer = String::new();
  while lexer.peek().is_some_and(&pred) {
    buffer.push(lexer.advance().unwrap());
  }
  buffer
}

fn skip_while(lexer: &mut Lexer, pred: impl Fn(char) -> bool) {
  while lexer.peek().is_some_and(&pred) {
    lexer.advance();
  }
}

fn skip_whitespace_and_comments(lexer: &mut Lexer) {
  loop {
    skip_while(lexer, char::is_whitespace);
    if lexer.peek() == Some('/') && lexer.peek2() == Some('/') {
      skip_while(lexer, |c| c != '\n');
      continue;
    }
    break;
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

  mod lexer {
    use super::*;

    fn assert_tokens(input: &str, expected: Vec<Token>) {
      let mut lexer = Lexer::new(input);
      for expected_token in expected {
        let actual = lexer.next_token().unwrap();
        assert_eq!(actual.kind, expected_token.kind, "token kind mismatch");
        assert_eq!(
          actual.literal, expected_token.literal,
          "token literal mismatch"
        );
      }
    }

    #[test]
    fn next_token() {
      assert_tokens(
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
      );
    }

    #[test]
    fn next_token_eval() {
      assert_tokens(
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
      );
    }

    #[test]
    fn next_token_comments() {
      assert_tokens(
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
      );
    }
  }
}
