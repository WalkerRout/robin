use std::iter::Peekable;
use std::mem;

use crate::ast::{Decl, Def, Eval, Expr, Program};
use crate::token::{Kind as TokenKind, Token};

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
  #[error("ran out of input while parsing")]
  UnexpectedEof,

  #[error("unexpected token: {0:?}")]
  UnexpectedToken(Token),

  #[error("expected {expected}, got {actual:?}")]
  Expected { expected: String, actual: Token },

  #[error("invalid integer literal: {0:?}")]
  InvalidInteger(Token),
}

#[derive(Debug, Clone)]
pub struct Parser<I>
where
  I: Iterator<Item = Token>,
{
  tokens: Peekable<I>,
  errors: Vec<ParserError>,
}

impl<I> Parser<I>
where
  I: Iterator<Item = Token>,
{
  pub fn new(tokens: I) -> Self {
    Self {
      tokens: tokens.peekable(),
      errors: Vec::new(),
    }
  }

  /// Parses a stream of tokens into a `Program` AST node
  pub fn parse(mut self) -> Result<Program, Vec<ParserError>> {
    match self.parse_program() {
      Some(program) => Ok(program),
      None => Err(mem::take(&mut self.errors)),
    }
  }

  fn next(&mut self) -> Option<Token> {
    self.tokens.next()
  }

  fn next_eof(&mut self) -> Result<Token, ParserError> {
    self.next().ok_or(ParserError::UnexpectedEof)
  }

  fn peek(&mut self) -> Option<&Token> {
    self.tokens.peek()
  }

  fn peek_eof(&mut self) -> Result<&Token, ParserError> {
    self.peek().ok_or(ParserError::UnexpectedEof)
  }

  fn peek_is(&mut self, kind: TokenKind) -> bool {
    matches!(self.peek(), Some(t) if t.kind == kind)
  }

  fn eat(&mut self, expected: TokenKind) -> Result<Token, ParserError> {
    let token = self.next_eof()?;
    if token.kind == expected {
      Ok(token)
    } else {
      Err(ParserError::Expected {
        expected: expected.to_string(),
        actual: token,
      })
    }
  }

  fn eat_int(&mut self) -> Result<u64, ParserError> {
    let token = self.eat(TokenKind::Int)?;
    token
      .literal
      .parse::<u64>()
      .map_err(|_| ParserError::InvalidInteger(token))
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

  fn synchronize(&mut self) {
    while let Some(token) = self.peek() {
      if token.kind == TokenKind::Semicolon {
        self.next();
        return;
      }
      self.next();
    }
  }

  fn parse_comma_separated<T>(
    &mut self,
    terminator: TokenKind,
    mut parse_element: impl FnMut(&mut Self) -> Result<T, ParserError>,
  ) -> Result<Vec<T>, ParserError> {
    let mut elements = Vec::new();
    if self.peek_is(terminator) {
      return Ok(elements);
    }
    loop {
      elements.push(parse_element(self)?);
      match self.peek() {
        Some(Token { kind, .. }) if *kind == terminator => break,
        Some(Token {
          kind: TokenKind::Comma,
          ..
        }) => {
          let _ = self.eat(TokenKind::Comma)?;
          if self.peek_is(terminator) {
            break;
          }
        }
        _ => return Err(ParserError::UnexpectedToken(self.peek_eof()?.clone())),
      }
    }
    Ok(elements)
  }

  fn parse_program(&mut self) -> Option<Program> {
    let mut decls = Vec::new();

    while self.peek().is_some() && !self.peek_is(TokenKind::Eof) {
      let decl = self.parse_decl();
      match self.record(decl) {
        Some(d) => decls.push(d),
        None => self.synchronize(),
      }
    }

    if self.errors.is_empty() {
      Some(Program { decls })
    } else {
      None
    }
  }

  fn parse_decl(&mut self) -> Result<Decl, ParserError> {
    match self.peek_eof()?.kind {
      TokenKind::Def => self.parse_def().map(Into::into),
      TokenKind::Eval => self.parse_eval().map(Into::into),
      _ => Err(ParserError::UnexpectedToken(self.peek_eof()?.clone())),
    }
  }

  fn parse_def(&mut self) -> Result<Def, ParserError> {
    let _ = self.eat(TokenKind::Def)?;
    let name_token = self.eat(TokenKind::Ident)?;
    let _ = self.eat(TokenKind::Assign)?;
    let body = self.parse_expr()?;
    let _ = self.eat(TokenKind::Semicolon)?;
    Ok(Def {
      name: name_token.literal,
      body,
    })
  }

  fn parse_eval(&mut self) -> Result<Eval, ParserError> {
    let _ = self.eat(TokenKind::Eval)?;
    let func = self.parse_expr()?;
    let _ = self.eat(TokenKind::LParen)?;
    let args = self.parse_comma_separated(TokenKind::RParen, |parser| parser.eat_int())?;
    let _ = self.eat(TokenKind::RParen)?;
    let _ = self.eat(TokenKind::Semicolon)?;
    Ok(Eval { func, args })
  }

  fn parse_expr(&mut self) -> Result<Expr, ParserError> {
    let token = self.peek_eof()?;
    match token.kind {
      TokenKind::Const => {
        self.next();
        let _ = self.eat(TokenKind::LParen)?;
        let arity = self.eat_int()? as usize;
        let _ = self.eat(TokenKind::Comma)?;
        let value = self.eat_int()?;
        let _ = self.eat(TokenKind::RParen)?;
        Ok(Expr::Const { arity, value })
      }
      TokenKind::Succ => {
        self.next();
        Ok(Expr::Succ)
      }
      TokenKind::Id => {
        self.next();
        let _ = self.eat(TokenKind::LParen)?;
        let k = self.eat_int()? as usize;
        let _ = self.eat(TokenKind::Comma)?;
        let n = self.eat_int()? as usize;
        let _ = self.eat(TokenKind::RParen)?;
        Ok(Expr::Id { k, n })
      }
      TokenKind::Ident => {
        let token = self.next_eof()?;
        Ok(Expr::Ref(token.literal))
      }
      TokenKind::Cn => {
        self.next();
        let _ = self.eat(TokenKind::LBracket)?;
        let f = self.parse_expr()?;
        let _ = self.eat(TokenKind::Comma)?;
        let gs = self.parse_comma_separated(TokenKind::RBracket, |parser| parser.parse_expr())?;
        let _ = self.eat(TokenKind::RBracket)?;
        Ok(Expr::Cn { f: Box::new(f), gs })
      }
      TokenKind::Pr => {
        self.next();
        let _ = self.eat(TokenKind::LBracket)?;
        let base = self.parse_expr()?;
        let _ = self.eat(TokenKind::Comma)?;
        let step = self.parse_expr()?;
        let _ = self.eat(TokenKind::RBracket)?;
        Ok(Expr::Pr {
          base: Box::new(base),
          step: Box::new(step),
        })
      }
      TokenKind::Mn => {
        self.next();
        let _ = self.eat(TokenKind::LBracket)?;
        let f = self.parse_expr()?;
        let _ = self.eat(TokenKind::RBracket)?;
        Ok(Expr::Mn { f: Box::new(f) })
      }
      _ => Err(ParserError::UnexpectedToken(self.peek_eof()?.clone())),
    }
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
