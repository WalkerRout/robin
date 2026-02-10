use std::fmt;

/// Catamorphism/abstract-fold over parsed AST
pub trait Visitor<T> {
  type Error;

  fn visit_program(&mut self, program: &Program) -> Result<T, Self::Error>;
  fn visit_decl(&mut self, decl: &Decl) -> Result<T, Self::Error>;
  fn visit_def(&mut self, def: &Def) -> Result<T, Self::Error>;
  fn visit_eval(&mut self, eval: &Eval) -> Result<T, Self::Error>;
  fn visit_expr(&mut self, expr: &Expr) -> Result<T, Self::Error>;
}

pub trait Visitable<T> {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
  pub decls: Vec<Decl>,
}

impl<T> Visitable<T> for Program {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_program(self)
  }
}

impl fmt::Display for Program {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for decl in &self.decls {
      writeln!(f, "{decl}")?;
    }
    Ok(())
  }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
  Def(Def),
  Eval(Eval),
}

impl<T> Visitable<T> for Decl {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_decl(self)
  }
}

impl From<Def> for Decl {
  fn from(def: Def) -> Self {
    Self::Def(def)
  }
}

impl From<Eval> for Decl {
  fn from(eval: Eval) -> Self {
    Self::Eval(eval)
  }
}

impl fmt::Display for Decl {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Def(def) => write!(f, "{def}"),
      Self::Eval(eval) => write!(f, "{eval}"),
    }
  }
}

/// `def <name> = <expr>;`
#[derive(Debug, Clone, PartialEq)]
pub struct Def {
  pub name: String,
  pub body: Expr,
}

impl<T> Visitable<T> for Def {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_def(self)
  }
}

impl fmt::Display for Def {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "def {} = {};", self.name, self.body)
  }
}

/// `eval <expr>(<arg>, ...);`
#[derive(Debug, Clone, PartialEq)]
pub struct Eval {
  pub func: Expr,
  pub args: Vec<u64>,
}

impl<T> Visitable<T> for Eval {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_eval(self)
  }
}

impl fmt::Display for Eval {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "eval {}(", self.func)?;
    for (i, arg) in self.args.iter().enumerate() {
      if i > 0 {
        write!(f, ", ")?;
      }
      write!(f, "{arg}")?;
    }
    write!(f, ");")
  }
}

// combinator trees

/// An expression in the recursive functions language
///
/// Each variant corresponds to a construct from recursion theory
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
  /// constant function - const(k, n)(x_1, ..., x_k) = n
  Const { arity: usize, value: u64 },
  /// successor function - s(x) = x'
  Succ,
  /// k'th projection from n arguments - id(k, n)(x_0,...,x_{n-1}) = x_k
  Id { k: usize, n: usize },
  /// function reference - refers to a function in the environment
  Ref(String),
  /// composition - Cn[f, g_0, ..., g_{m-1}](x_0, ..., x_{n-1}) = f(g_0(...), ..., g_{m-1}(...))
  Cn { f: Box<Expr>, gs: Vec<Expr> },
  /// primitive recursion - Pr[base, step]
  Pr { base: Box<Expr>, step: Box<Expr> },
  /// unbounded minimization (μ-operator) - Mn[f]
  Mn { f: Box<Expr> },
}

impl<T> Visitable<T> for Expr {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_expr(self)
  }
}

impl fmt::Display for Expr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Const { arity, value } => write!(f, "const({arity},{value})"),
      Self::Succ => write!(f, "s"),
      Self::Id { k, n } => write!(f, "id({k},{n})"),
      Self::Ref(name) => write!(f, "{name}"),
      Self::Cn { f: func, gs } => {
        write!(f, "Cn[{func}")?;
        for g in gs {
          write!(f, ", {g}")?;
        }
        write!(f, "]")
      }
      Self::Pr { base, step } => write!(f, "Pr[{base}, {step}]"),
      Self::Mn { f: func } => write!(f, "Mn[{func}]"),
    }
  }
}
