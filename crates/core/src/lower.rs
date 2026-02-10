use std::collections::HashMap;

use crate::ast::{Decl, Def, Eval, Expr, Program, Visitable, Visitor};
use crate::ir::{Cmp, EvalIR, FuncIR, Node, ProgramIR, shift_loop_depth, shift_search_depth};
use crate::pass::Pass;

const MU_STEP_LIMIT: i64 = 100_000;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Lowerror {
  #[error("undefined function - {0}")]
  Undefined(String),

  // can we use latex in doc strings? thatd be neat...
  // - could use https://github.com/victe/rust-latex-doc-minimal-example but its a bit more work...
  #[error("id({k},{n}) - k must satisfy 1 <= k <= n")]
  InvalidProjection { k: usize, n: usize },

  #[error("arity mismatch - {name} expects {expected} arg(s), got {got}")]
  ArityMismatch {
    name: String,
    expected: usize,
    got: usize,
  },

  #[error("Cn - all g_i must have the same arity")]
  CnArityMismatch,

  #[error("Pr step arity - expected {expected}, got {got}")]
  PrStepArity { expected: usize, got: usize },

  #[error("Mn - inner function must have arity >= 1, got {0}")]
  MnArityTooSmall(usize),
}

/// AST -> IR lowering visitor
///
/// implements Visitor<Node>, visit_expr is the actual catamorphism (Expr -> Node),
/// visit_def/visit_eval handle structural wrapping and accumulate into ProgramIR...
pub struct Lower {
  env: HashMap<String, usize>,
  def_bodies: HashMap<String, Expr>,
  funcs: Vec<FuncIR>,
  evals: Vec<EvalIR>,
}

impl Lower {
  pub fn new() -> Self {
    Self {
      env: HashMap::new(),
      def_bodies: HashMap::new(),
      funcs: Vec::new(),
      evals: Vec::new(),
    }
  }

  /// Extract the accumulated IR
  pub fn finish(self) -> ProgramIR {
    ProgramIR {
      funcs: self.funcs,
      evals: self.evals,
    }
  }

  // paramorphism, folding Expr into Node while threading symbolic args top down
  // through the recursion
  fn lower_expr(&self, expr: &Expr, args: &[Node]) -> Result<Node, Lowerror> {
    // peepholes match structure before folding
    if let Expr::Pr { base, step } = expr {
      if let Some(node) = try_peephole_pred(base, step, args) {
        return Ok(node);
      }
      if let Some(node) = try_peephole_add(base, step, args) {
        return Ok(node);
      }
      if let Some(node) = try_peephole_monus(&self.def_bodies, base, step, args) {
        return Ok(node);
      }
      if let Some(node) = try_peephole_sg(&self.def_bodies, base, step, args) {
        return Ok(node);
      }
      if let Some(node) = try_peephole_sgbar(&self.def_bodies, base, step, args) {
        return Ok(node);
      }
      if let Some(node) = try_peephole_mult(&self.def_bodies, base, step, args) {
        return Ok(node);
      }
    }

    match expr {
      Expr::Const { value, .. } => Ok(Node::Iconst(*value as i64)),

      Expr::Succ => Ok(Node::Iadd(
        Box::new(args[0].clone()),
        Box::new(Node::Iconst(1)),
      )),

      Expr::Id { k, .. } => Ok(args[*k - 1].clone()),

      Expr::Ref(name) => {
        // inline small functions
        if let Some(body) = self.def_bodies.get(name)
          && expr_node_count(body) <= 6
        {
          return self.lower_expr(body, args);
        }
        Ok(Node::Call {
          name: name.clone(),
          args: args.to_vec(),
        })
      }

      Expr::Cn { f, gs } => {
        let mut inner = Vec::with_capacity(gs.len());
        for g in gs {
          inner.push(self.lower_expr(g, args)?);
        }
        self.lower_expr(f, &inner)
      }

      Expr::Pr { base, step } => self.lower_pr(base, step, args),
      Expr::Mn { f } => self.lower_mn(f, args),
    }
  }

  fn lower_pr(&self, base: &Expr, step: &Expr, args: &[Node]) -> Result<Node, Lowerror> {
    let base_arity = infer_arity(&self.env, base)?;
    let (xs, y) = if base_arity == 0 {
      (&args[..0], &args[0])
    } else {
      let n = args.len();
      (&args[..n - 1], &args[n - 1])
    };

    let base_args: &[Node] = if base_arity == 0 { &[] } else { xs };
    let init = self.lower_expr(base, base_args)?;

    let mut step_args: Vec<Node> = xs.iter().map(|x| shift_loop_depth(x, 1)).collect();
    step_args.push(Node::Counter(0));
    step_args.push(Node::Acc(0));

    let body = self.lower_expr(step, &step_args)?;

    Ok(Node::Loop {
      bound: Box::new(y.clone()),
      init: Box::new(init),
      body: Box::new(body),
    })
  }

  fn lower_mn(&self, f: &Expr, args: &[Node]) -> Result<Node, Lowerror> {
    let mut search_args: Vec<Node> = args.iter().map(|a| shift_search_depth(a, 1)).collect();
    search_args.push(Node::Probe(0));

    let body = self.lower_expr(f, &search_args)?;

    Ok(Node::Search {
      body: Box::new(body),
      limit: MU_STEP_LIMIT,
    })
  }
}

impl Pass<Program> for Lower {
  type Output = ProgramIR;
  type Error = Lowerror;

  fn run(mut self, program: &Program) -> Result<ProgramIR, Lowerror> {
    let _ = program.fold(&mut self)?;
    Ok(self.finish())
  }
}

impl Visitor<Node> for Lower {
  type Error = Lowerror;

  fn visit_program(&mut self, program: &Program) -> Result<Node, Lowerror> {
    for decl in &program.decls {
      decl.fold(self)?;
    }
    // program result is structural... real output via Pass::run
    Ok(Node::Iconst(0))
  }

  fn visit_decl(&mut self, decl: &Decl) -> Result<Node, Lowerror> {
    match decl {
      Decl::Def(def) => def.fold(self),
      Decl::Eval(eval) => eval.fold(self),
    }
  }

  fn visit_def(&mut self, def: &Def) -> Result<Node, Lowerror> {
    let arity = infer_arity(&self.env, &def.body)?;
    let arg_nodes: Vec<Node> = (0..arity).map(Node::Arg).collect();
    let body = self.lower_expr(&def.body, &arg_nodes)?;

    self.funcs.push(FuncIR {
      name: def.name.clone(),
      arity,
      body: body.clone(),
    });
    self.env.insert(def.name.clone(), arity);
    self.def_bodies.insert(def.name.clone(), def.body.clone());

    Ok(body)
  }

  fn visit_eval(&mut self, eval: &Eval) -> Result<Node, Lowerror> {
    let arity = infer_arity(&self.env, &eval.func)?;
    if arity != eval.args.len() {
      return Err(Lowerror::ArityMismatch {
        name: format!("{}", eval.func),
        expected: arity,
        got: eval.args.len(),
      });
    }

    let arg_nodes: Vec<Node> = eval.args.iter().map(|a| Node::Iconst(*a as i64)).collect();
    let body = self.lower_expr(&eval.func, &arg_nodes)?;

    self.evals.push(EvalIR { body: body.clone() });

    Ok(body)
  }

  fn visit_expr(&mut self, expr: &Expr) -> Result<Node, Lowerror> {
    // standalone expr fold with no args context
    self.lower_expr(expr, &[])
  }
}

impl Default for Lower {
  fn default() -> Self {
    Self::new()
  }
}

// peephole optimizations we can perform when folding Expr -> Node

fn try_peephole_pred(base: &Expr, step: &Expr, args: &[Node]) -> Option<Node> {
  if matches!(base, Expr::Const { arity: 0, value: 0 }) && matches!(step, Expr::Id { k: 1, n: 2 }) {
    let y = args[0].clone();
    let zero = Node::Iconst(0);
    let one = Node::Iconst(1);
    Some(Node::Select {
      cond: Box::new(Node::Icmp(
        Cmp::Eq,
        Box::new(y.clone()),
        Box::new(zero.clone()),
      )),
      then_val: Box::new(zero),
      else_val: Box::new(Node::Isub(Box::new(y), Box::new(one))),
    })
  } else {
    None
  }
}

fn try_peephole_add(base: &Expr, step: &Expr, args: &[Node]) -> Option<Node> {
  if !matches!(base, Expr::Id { k: 1, n: 1 }) {
    return None;
  }
  if let Expr::Cn { f, gs } = step
    && matches!(f.as_ref(), Expr::Succ)
    && gs.len() == 1
    && matches!(gs[0], Expr::Id { k: 3, n: 3 })
  {
    return Some(Node::Iadd(
      Box::new(args[0].clone()),
      Box::new(args[1].clone()),
    ));
  }
  None
}

fn try_peephole_monus(
  def_bodies: &HashMap<String, Expr>,
  base: &Expr,
  step: &Expr,
  args: &[Node],
) -> Option<Node> {
  if !matches!(base, Expr::Id { k: 1, n: 1 }) {
    return None;
  }
  if let Expr::Cn { f, gs } = step
    && is_pred_function(f, def_bodies)
    && gs.len() == 1
    && matches!(gs[0], Expr::Id { k: 3, n: 3 })
  {
    let x = args[0].clone();
    let y = args[1].clone();
    let zero = Node::Iconst(0);
    return Some(Node::Select {
      cond: Box::new(Node::Icmp(
        Cmp::Ugt,
        Box::new(x.clone()),
        Box::new(y.clone()),
      )),
      then_val: Box::new(Node::Isub(Box::new(x), Box::new(y))),
      else_val: Box::new(zero),
    });
  }
  None
}

fn try_peephole_sg(
  def_bodies: &HashMap<String, Expr>,
  base: &Expr,
  step: &Expr,
  args: &[Node],
) -> Option<Node> {
  if !matches!(base, Expr::Const { arity: 0, value: 0 }) {
    return None;
  }
  if let Expr::Cn { f, gs } = step
    && matches!(f.as_ref(), Expr::Succ)
    && gs.len() == 1
    && is_const_value(&gs[0], 0, def_bodies)
  {
    let y = args[0].clone();
    let zero = Node::Iconst(0);
    let one = Node::Iconst(1);
    return Some(Node::Select {
      cond: Box::new(Node::Icmp(Cmp::Ne, Box::new(y), Box::new(zero.clone()))),
      then_val: Box::new(one),
      else_val: Box::new(zero),
    });
  }
  None
}

fn try_peephole_sgbar(
  def_bodies: &HashMap<String, Expr>,
  base: &Expr,
  step: &Expr,
  args: &[Node],
) -> Option<Node> {
  if matches!(base, Expr::Const { arity: 0, value: 1 }) && is_const_value(step, 0, def_bodies) {
    let y = args[0].clone();
    let zero = Node::Iconst(0);
    let one = Node::Iconst(1);
    Some(Node::Select {
      cond: Box::new(Node::Icmp(Cmp::Eq, Box::new(y), Box::new(zero.clone()))),
      then_val: Box::new(one),
      else_val: Box::new(zero),
    })
  } else {
    None
  }
}

fn try_peephole_mult(
  def_bodies: &HashMap<String, Expr>,
  base: &Expr,
  step: &Expr,
  args: &[Node],
) -> Option<Node> {
  if !is_const_value(base, 0, def_bodies) {
    return None;
  }
  if let Expr::Cn { f, gs } = step
    && is_add_function(f, def_bodies)
    && gs.len() == 2
    && matches!(gs[0], Expr::Id { k: 3, n: 3 })
    && matches!(gs[1], Expr::Id { k: 1, n: 3 })
  {
    return Some(Node::Imul(
      Box::new(args[0].clone()),
      Box::new(args[1].clone()),
    ));
  }
  None
}

pub fn infer_arity(env: &HashMap<String, usize>, expr: &Expr) -> Result<usize, Lowerror> {
  match expr {
    Expr::Const { arity, .. } => Ok(*arity),
    Expr::Succ => Ok(1),
    Expr::Id { k, n } => {
      if *k < 1 || *k > *n {
        return Err(Lowerror::InvalidProjection { k: *k, n: *n });
      }
      Ok(*n)
    }
    Expr::Ref(name) => env
      .get(name)
      .copied()
      .ok_or_else(|| Lowerror::Undefined(name.clone())),
    Expr::Cn { f, gs } => {
      let m = infer_arity(env, f)?;
      if gs.len() != m {
        return Err(Lowerror::ArityMismatch {
          name: "Cn outer function".to_string(),
          expected: m,
          got: gs.len(),
        });
      }
      if gs.is_empty() {
        return Ok(0);
      }
      let n = infer_arity(env, &gs[0])?;
      for g in gs.iter().skip(1) {
        if infer_arity(env, g)? != n {
          return Err(Lowerror::CnArityMismatch);
        }
      }
      Ok(n)
    }
    Expr::Pr { base, step } => {
      let ba = infer_arity(env, base)?;
      let sa = infer_arity(env, step)?;
      let expected = if ba == 0 { 2 } else { ba + 2 };
      if sa != expected {
        return Err(Lowerror::PrStepArity { expected, got: sa });
      }
      Ok(if ba == 0 { 1 } else { ba + 1 })
    }
    Expr::Mn { f } => {
      let inner = infer_arity(env, f)?;
      if inner < 1 {
        return Err(Lowerror::MnArityTooSmall(inner));
      }
      Ok(inner - 1)
    }
  }
}

fn expr_node_count(expr: &Expr) -> usize {
  match expr {
    Expr::Const { .. } | Expr::Succ | Expr::Id { .. } | Expr::Ref(_) => 1,
    Expr::Cn { f, gs } => 1 + expr_node_count(f) + gs.iter().map(expr_node_count).sum::<usize>(),
    Expr::Pr { base, step } => 1 + expr_node_count(base) + expr_node_count(step),
    Expr::Mn { f } => 1 + expr_node_count(f),
  }
}

fn is_const_value(expr: &Expr, val: u64, def_bodies: &HashMap<String, Expr>) -> bool {
  match expr {
    Expr::Const { value, .. } => *value == val,
    Expr::Ref(name) => def_bodies
      .get(name)
      .is_some_and(|body| is_const_value(body, val, def_bodies)),
    Expr::Cn { f, gs } => {
      if is_const_value(f, val, def_bodies) {
        return true;
      }
      if val > 0 && matches!(f.as_ref(), Expr::Succ) && gs.len() == 1 {
        return is_const_value(&gs[0], val - 1, def_bodies);
      }
      false
    }
    _ => false,
  }
}

fn is_pred_function(expr: &Expr, def_bodies: &HashMap<String, Expr>) -> bool {
  let resolved = match expr {
    Expr::Ref(name) => match def_bodies.get(name) {
      Some(body) => body,
      None => return false,
    },
    other => other,
  };
  matches!(
    resolved,
    Expr::Pr { base, step }
    if matches!(base.as_ref(), Expr::Const { arity: 0, value: 0 })
      && matches!(step.as_ref(), Expr::Id { k: 1, n: 2 })
  )
}

fn is_add_function(expr: &Expr, def_bodies: &HashMap<String, Expr>) -> bool {
  let resolved = match expr {
    Expr::Ref(name) => match def_bodies.get(name) {
      Some(body) => body,
      None => return false,
    },
    other => other,
  };
  if let Expr::Pr { base, step } = resolved
    && matches!(base.as_ref(), Expr::Id { k: 1, n: 1 })
    && let Expr::Cn { f, gs } = step.as_ref()
  {
    return matches!(f.as_ref(), Expr::Succ) && matches!(gs.first(), Some(Expr::Id { k: 3, n: 3 }));
  }
  false
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ast::Visitable;
  use crate::lexer::Lexer;
  use crate::parser::Parser;
  use crate::pass::Acceptor;

  fn parse_src(input: &str) -> Program {
    let lexer = Lexer::new(input);
    let parser = Parser::new(lexer);
    parser.parse().expect("parse failed")
  }

  fn lower_src(input: &str) -> ProgramIR {
    let program = parse_src(input);
    program.accept(Lower::new()).expect("lowering failed")
  }

  mod lower {
    use super::*;

    #[test]
    fn lowers_const() {
      let ir = lower_src("def f = const(1, 0);\neval f(42);");
      assert_eq!(ir.funcs.len(), 1);
      assert_eq!(ir.funcs[0].body, Node::Iconst(0));
      assert_eq!(ir.evals.len(), 1);
      assert_eq!(ir.evals[0].body, Node::Iconst(0));
    }

    #[test]
    fn lowers_succ() {
      let ir = lower_src("def f = s;\neval f(5);");
      assert_eq!(
        ir.funcs[0].body,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)))
      );
    }

    #[test]
    fn lowers_pred_peephole() {
      let ir = lower_src("def pred = Pr[const(0, 0), id(1,2)];\neval pred(5);");
      match &ir.funcs[0].body {
        Node::Select { .. } => {}
        other => panic!("expected Select for pred peephole, got: {other}"),
      }
    }

    #[test]
    fn lowers_add_peephole() {
      let ir = lower_src("def add = Pr[id(1,1), Cn[s, id(3,3)]];\neval add(3, 2);");
      assert_eq!(
        ir.funcs[0].body,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Arg(1)))
      );
    }

    #[test]
    fn lowers_mult_peephole() {
      let ir = lower_src(
        "def z = const(1, 0);\n\
         def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
         def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
         eval mult(3, 4);",
      );
      assert_eq!(
        ir.funcs[2].body,
        Node::Imul(Box::new(Node::Arg(0)), Box::new(Node::Arg(1)))
      );
    }

    #[test]
    fn lowers_generic_pr_to_loop() {
      let ir = lower_src(
        "def z = const(1, 0);\n\
         def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
         def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
         def exp = Pr[const(1,1), Cn[mult, id(1,3), id(3,3)]];\n\
         eval exp(2, 10);",
      );
      match &ir.funcs[3].body {
        Node::Loop { .. } => {}
        other => panic!("expected Loop for exp, got: {other}"),
      }
    }

    #[test]
    fn lowers_mn_to_search() {
      let ir = lower_src(
        "def z = const(1, 0);\n\
         def pred = Pr[const(0, 0), id(1,2)];\n\
         def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
         def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
         def monus = Pr[id(1,1), Cn[pred, id(3,3)]];\n\
         def isqrt = Mn[Cn[monus, id(1,2), Cn[mult, id(2,2), id(2,2)]]];\n\
         eval isqrt(9);",
      );
      match &ir.funcs[5].body {
        Node::Search { .. } => {}
        other => panic!("expected Search for isqrt, got: {other}"),
      }
    }

    #[test]
    fn eval_inline_combinator() {
      let ir = lower_src("eval Cn[s, s](5);");
      assert_eq!(ir.funcs.len(), 0);
      assert_eq!(ir.evals.len(), 1);
      assert_eq!(
        ir.evals[0].body,
        Node::Iadd(
          Box::new(Node::Iadd(
            Box::new(Node::Iconst(5)),
            Box::new(Node::Iconst(1)),
          )),
          Box::new(Node::Iconst(1)),
        )
      );
    }

    #[test]
    fn lowers_sg_peephole() {
      let ir = lower_src(
        "def sg = Pr[const(0, 0), Cn[s, const(2, 0)]];\n\
         eval sg(0);",
      );
      match &ir.funcs[0].body {
        Node::Select { .. } => {}
        other => panic!("expected Select for sg peephole, got: {other}"),
      }
    }

    #[test]
    fn lowers_sgbar_peephole() {
      let ir = lower_src(
        "def sgbar = Pr[const(0, 1), const(2, 0)];\n\
         eval sgbar(0);",
      );
      match &ir.funcs[0].body {
        Node::Select { .. } => {}
        other => panic!("expected Select for sgbar peephole, got: {other}"),
      }
    }

    #[test]
    fn arity_mismatch_in_eval() {
      let program = parse_src("def f = Pr[id(1,1), Cn[s, id(3,3)]];\neval f(1, 2, 3);");
      let err = program.accept(Lower::new()).unwrap_err();
      match err {
        Lowerror::ArityMismatch {
          expected: 2,
          got: 3,
          ..
        } => {}
        other => panic!("expected ArityMismatch, got: {other}"),
      }
    }

    mod acceptor {
      use super::*;

      #[test]
      fn accept() {
        let program = parse_src("def f = s;\neval f(5);");
        let ir = program.accept(Lower::new()).unwrap();
        assert_eq!(
          ir.funcs[0].body,
          Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)))
        );
        assert_eq!(ir.evals.len(), 1);
      }
    }

    mod visitor {
      use super::*;

      #[test]
      fn fold() {
        let program = parse_src("def f = s;");
        let mut lower = Lower::new();
        let result = program.fold(&mut lower).unwrap();
        assert_eq!(result, Node::Iconst(0));
        let ir = lower.finish();
        assert_eq!(
          ir.funcs[0].body,
          Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)))
        );
      }
    }
  }
}
