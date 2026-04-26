use std::collections::VecDeque;
use std::mem;

use crate::ast::{Decl, Def, Eval, Expr, Program};
use crate::token::{Kind as TokenKind, Token};

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
  #[error("ran out of input while parsing")]
  UnexpectedEof,

  #[error("unexpected token - {0:?}")]
  UnexpectedToken(Token),

  #[error("expected {expected}, got {actual:?}")]
  Expected { expected: String, actual: Token },

  #[error("invalid integer literal - {0:?}")]
  InvalidInteger(Token),
}

#[derive(Debug, Clone)]
pub struct Parser<I> {
  tokens: I,
  buf: VecDeque<Token>,
  errors: Vec<ParserError>,
}

const LOOKAHEAD: usize = 1;

impl<I> Parser<I>
where
  I: Iterator<Item = Token>,
{
  pub fn new(mut tokens: I) -> Self {
    use std::borrow::BorrowMut;
    let buf = tokens.borrow_mut().take(LOOKAHEAD).collect();
    Self {
      tokens,
      buf,
      errors: Vec::new(),
    }
  }

  pub fn parse(mut self) -> Result<Program, Vec<ParserError>> {
    let mut decls = Vec::new();
    while self.peek().is_some() && !self.peek_is(TokenKind::Eof) {
      let decl = eat_decl(&mut self);
      match self.record(decl) {
        Some(d) => decls.push(d),
        None => synchronize(&mut self),
      }
    }

    if self.errors.is_empty() {
      Ok(Program { decls })
    } else {
      Err(mem::take(&mut self.errors))
    }
  }

  fn peek(&self) -> Option<&Token> {
    self.buf.front()
  }

  fn peek_is(&self, kind: TokenKind) -> bool {
    matches!(self.peek(), Some(t) if t.kind == kind)
  }

  fn advance(&mut self) -> Option<Token> {
    let t = self.buf.pop_front()?;
    if let Some(next) = self.tokens.next() {
      self.buf.push_back(next);
    }
    Some(t)
  }

  fn record<T>(&mut self, result: Result<T, ParserError>) -> Option<T> {
    match result {
      Ok(t) => Some(t),
      Err(e) => {
        self.errors.push(e);
        None
      }
    }
  }
}

fn advance_eof<I>(parser: &mut Parser<I>) -> Result<Token, ParserError>
where
  I: Iterator<Item = Token>,
{
  parser.advance().ok_or(ParserError::UnexpectedEof)
}

fn peek_eof<I>(parser: &mut Parser<I>) -> Result<&Token, ParserError>
where
  I: Iterator<Item = Token>,
{
  parser.peek().ok_or(ParserError::UnexpectedEof)
}

fn eat<I>(parser: &mut Parser<I>, expected: TokenKind) -> Result<Token, ParserError>
where
  I: Iterator<Item = Token>,
{
  let token = advance_eof(parser)?;
  if token.kind == expected {
    Ok(token)
  } else {
    Err(ParserError::Expected {
      expected: expected.to_string(),
      actual: token,
    })
  }
}

fn eat_int<I>(parser: &mut Parser<I>) -> Result<u64, ParserError>
where
  I: Iterator<Item = Token>,
{
  let token = eat(parser, TokenKind::Int)?;
  token
    .literal
    .parse::<u64>()
    .map_err(|_| ParserError::InvalidInteger(token))
}

fn munch_comma_separated<T, I>(
  parser: &mut Parser<I>,
  terminator: TokenKind,
  mut eat_element: impl FnMut(&mut Parser<I>) -> Result<T, ParserError>,
) -> Result<Vec<T>, ParserError>
where
  I: Iterator<Item = Token>,
{
  let mut elements = Vec::new();
  if parser.peek_is(terminator) {
    return Ok(elements);
  }
  loop {
    elements.push(eat_element(parser)?);
    match parser.peek() {
      Some(Token { kind, .. }) if *kind == terminator => break,
      Some(Token {
        kind: TokenKind::Comma,
        ..
      }) => {
        let _ = eat(parser, TokenKind::Comma)?;
        if parser.peek_is(terminator) {
          break;
        }
      }
      _ => return Err(ParserError::UnexpectedToken(peek_eof(parser)?.clone())),
    }
  }
  Ok(elements)
}

fn synchronize<I>(parser: &mut Parser<I>)
where
  I: Iterator<Item = Token>,
{
  while let Some(token) = parser.peek() {
    if token.kind == TokenKind::Semicolon {
      let _ = parser.advance();
      return;
    }
    let _ = parser.advance();
  }
}

fn eat_decl<I>(parser: &mut Parser<I>) -> Result<Decl, ParserError>
where
  I: Iterator<Item = Token>,
{
  match peek_eof(parser)?.kind {
    TokenKind::Def => eat_def(parser).map(Into::into),
    TokenKind::Eval => eat_eval(parser).map(Into::into),
    _ => Err(ParserError::UnexpectedToken(peek_eof(parser)?.clone())),
  }
}

fn eat_def<I>(parser: &mut Parser<I>) -> Result<Def, ParserError>
where
  I: Iterator<Item = Token>,
{
  let _ = eat(parser, TokenKind::Def)?;
  let name_token = eat(parser, TokenKind::Ident)?;
  let _ = eat(parser, TokenKind::Assign)?;
  let body = eat_expr(parser)?;
  let _ = eat(parser, TokenKind::Semicolon)?;
  Ok(Def {
    name: name_token.literal,
    body,
  })
}

fn eat_eval<I>(parser: &mut Parser<I>) -> Result<Eval, ParserError>
where
  I: Iterator<Item = Token>,
{
  let _ = eat(parser, TokenKind::Eval)?;
  let func = eat_expr(parser)?;
  let _ = eat(parser, TokenKind::LParen)?;
  let args = munch_comma_separated(parser, TokenKind::RParen, |p| eat_int(p))?;
  let _ = eat(parser, TokenKind::RParen)?;
  let _ = eat(parser, TokenKind::Semicolon)?;
  Ok(Eval { func, args })
}

fn eat_expr<I>(parser: &mut Parser<I>) -> Result<Expr, ParserError>
where
  I: Iterator<Item = Token>,
{
  let token = peek_eof(parser)?;
  match token.kind {
    TokenKind::Const => {
      parser.advance();
      let _ = eat(parser, TokenKind::LParen)?;
      let arity = eat_int(parser)? as usize;
      let _ = eat(parser, TokenKind::Comma)?;
      let value = eat_int(parser)?;
      let _ = eat(parser, TokenKind::RParen)?;
      Ok(Expr::Const { arity, value })
    }
    TokenKind::Succ => {
      parser.advance();
      Ok(Expr::Succ)
    }
    TokenKind::Id => {
      parser.advance();
      let _ = eat(parser, TokenKind::LParen)?;
      let k = eat_int(parser)? as usize;
      let _ = eat(parser, TokenKind::Comma)?;
      let n = eat_int(parser)? as usize;
      let _ = eat(parser, TokenKind::RParen)?;
      Ok(Expr::Id { k, n })
    }
    TokenKind::Ident => {
      let token = advance_eof(parser)?;
      Ok(Expr::Ref(token.literal))
    }
    TokenKind::Cn => {
      parser.advance();
      let _ = eat(parser, TokenKind::LBracket)?;
      let f = eat_expr(parser)?;
      let _ = eat(parser, TokenKind::Comma)?;
      let gs = munch_comma_separated(parser, TokenKind::RBracket, |p| eat_expr(p))?;
      let _ = eat(parser, TokenKind::RBracket)?;
      Ok(Expr::Cn { f: Box::new(f), gs })
    }
    TokenKind::Pr => {
      parser.advance();
      let _ = eat(parser, TokenKind::LBracket)?;
      let base = eat_expr(parser)?;
      let _ = eat(parser, TokenKind::Comma)?;
      let step = eat_expr(parser)?;
      let _ = eat(parser, TokenKind::RBracket)?;
      Ok(Expr::Pr {
        base: Box::new(base),
        step: Box::new(step),
      })
    }
    TokenKind::Mn => {
      parser.advance();
      let _ = eat(parser, TokenKind::LBracket)?;
      let f = eat_expr(parser)?;
      let _ = eat(parser, TokenKind::RBracket)?;
      Ok(Expr::Mn { f: Box::new(f) })
    }
    _ => Err(ParserError::UnexpectedToken(peek_eof(parser)?.clone())),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lexer::Lexer;

  mod parser {
    use super::*;

    fn parse(input: &str) -> Result<Program, Vec<ParserError>> {
      let lexer = Lexer::new(input);
      let parser = Parser::new(lexer);
      parser.parse()
    }

    #[test]
    fn parse_def() {
      let result = parse("def pred = Pr[const(0, 0), id(1,2)];").unwrap();
      assert_eq!(
        result,
        Program {
          decls: vec![Decl::Def(Def {
            name: "pred".into(),
            body: Expr::Pr {
              base: Box::new(Expr::Const { arity: 0, value: 0 }),
              step: Box::new(Expr::Id { k: 1, n: 2 }),
            },
          })],
        }
      );
    }

    #[test]
    fn parse_eval() {
      let result = parse("eval add(3, 2);").unwrap();
      assert_eq!(
        result,
        Program {
          decls: vec![Decl::Eval(Eval {
            func: Expr::Ref("add".into()),
            args: vec![3, 2],
          })],
        }
      );
    }

    #[test]
    fn parse_minimization() {
      let result =
        parse("def isqrt = Mn[Cn[monus, id(1,2), Cn[mult, id(2,2), id(2,2)]]];").unwrap();
      assert_eq!(
        result,
        Program {
          decls: vec![Decl::Def(Def {
            name: "isqrt".into(),
            body: Expr::Mn {
              f: Box::new(Expr::Cn {
                f: Box::new(Expr::Ref("monus".into())),
                gs: vec![
                  Expr::Id { k: 1, n: 2 },
                  Expr::Cn {
                    f: Box::new(Expr::Ref("mult".into())),
                    gs: vec![Expr::Id { k: 2, n: 2 }, Expr::Id { k: 2, n: 2 }],
                  },
                ],
              }),
            },
          })],
        }
      );
    }

    #[test]
    fn parse_error_missing_name() {
      assert!(parse("def = const(1, 0);").is_err());
    }

    #[test]
    fn parse_error_missing_eq() {
      assert!(parse("def x const(1, 0);").is_err());
    }

    #[test]
    fn parse_error_unexpected_token() {
      assert!(parse("blah").is_err());
    }
  }
}
