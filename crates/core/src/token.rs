use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
  Unknown = 0,
  Eof,

  // Identifiers + literals
  Ident,
  Int,

  // Symbols
  Assign,    // =
  Comma,     // ,
  Semicolon, // ;
  LParen,    // (
  RParen,    // )
  LBracket,  // [
  RBracket,  // ]

  // Combinator keywords
  Cn, // composition
  Pr, // primitive recursion
  Mn, // minimization

  // Basic function keywords
  Const, // const (constant function)
  Succ,  // s     (successor function)
  Id,    // id    (projection function)

  // Declaration keywords
  Def,  // def
  Eval, // eval
}

impl fmt::Display for Kind {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let s = match self {
      Self::Unknown => "<unknown>",
      Self::Eof => "<eof>",

      Self::Ident => "<ident>",
      Self::Int => "<int>",

      Self::Assign => "=",
      Self::Comma => ",",
      Self::Semicolon => ";",
      Self::LParen => "(",
      Self::RParen => ")",
      Self::LBracket => "[",
      Self::RBracket => "]",

      Self::Cn => "Cn",
      Self::Pr => "Pr",
      Self::Mn => "Mn",

      Self::Const => "const",
      Self::Succ => "s",
      Self::Id => "id",

      Self::Def => "def",
      Self::Eval => "eval",
    };
    write!(f, "{s}")
  }
}

static KEYWORDS: OnceLock<HashMap<&'static str, Kind>> = OnceLock::new();

fn init_keywords() -> HashMap<&'static str, Kind> {
  let mut m = HashMap::new();
  m.insert("Cn", Kind::Cn);
  m.insert("Pr", Kind::Pr);
  m.insert("Mn", Kind::Mn);
  m.insert("const", Kind::Const);
  m.insert("s", Kind::Succ);
  m.insert("id", Kind::Id);
  m.insert("def", Kind::Def);
  m.insert("eval", Kind::Eval);
  m
}

pub fn lookup_identifier(s: &str) -> Kind {
  let map = KEYWORDS.get_or_init(init_keywords);
  *map.get(s).unwrap_or(&Kind::Ident)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
  pub kind: Kind,
  pub literal: String,
}

impl Token {
  pub fn new(kind: Kind, literal: impl Into<String>) -> Self {
    Self {
      kind,
      literal: literal.into(),
    }
  }
}
