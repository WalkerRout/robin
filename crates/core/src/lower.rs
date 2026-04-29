use std::collections::HashMap;

use crate::ast::{Decl, Def, Eval, Expr, Program};
use crate::ir::{Cmp, EvalIR, FuncIR, Node, ProgramIR, shift_loop_depth, shift_search_depth};

const MU_STEP_LIMIT: i64 = 100_000;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
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

/// ast -> ir lowering
///
/// paramorphism: `Program` -> `ProgramIR`, folding `Expr` trees into `Node` trees
/// while threading symbolic args top-down through the recursion
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

  /// consume the lowering pass and produce a `ProgramIR`
  pub fn lower(mut self, program: Program) -> Result<ProgramIR, LowerError> {
    for decl in program.decls {
      match decl {
        Decl::Def(def) => lower_def(&mut self, def)?,
        Decl::Eval(eval) => lower_eval(&mut self, eval)?,
      }
    }
    Ok(ProgramIR {
      funcs: self.funcs,
      evals: self.evals,
    })
  }
}

impl Default for Lower {
  fn default() -> Self {
    Self::new()
  }
}

fn lower_def(lower: &mut Lower, def: Def) -> Result<(), LowerError> {
  let arity = infer_arity(&lower.env, &def.body)?;
  let arg_nodes: Vec<Node> = (0..arity).map(Node::Arg).collect();
  let body = lower_expr(lower, &def.body, &arg_nodes)?;

  lower.funcs.push(FuncIR {
    name: def.name.clone(),
    arity,
    body,
  });
  lower.env.insert(def.name.clone(), arity);
  lower.def_bodies.insert(def.name, def.body);

  Ok(())
}

fn lower_eval(lower: &mut Lower, eval: Eval) -> Result<(), LowerError> {
  let arity = infer_arity(&lower.env, &eval.func)?;
  if arity != eval.args.len() {
    return Err(LowerError::ArityMismatch {
      name: format!("{}", eval.func),
      expected: arity,
      got: eval.args.len(),
    });
  }

  let arg_nodes: Vec<Node> = eval.args.iter().map(|a| Node::Iconst(*a as i64)).collect();
  let body = lower_expr(lower, &eval.func, &arg_nodes)?;

  lower.evals.push(EvalIR { body });

  Ok(())
}

fn lower_expr(lower: &Lower, expr: &Expr, args: &[Node]) -> Result<Node, LowerError> {
  // peepholes match structure before folding
  if let Expr::Pr { base, step } = expr {
    if let Some(node) = try_peephole_pred(base, step, args) {
      return Ok(node);
    }
    if let Some(node) = try_peephole_add(base, step, args) {
      return Ok(node);
    }
    if let Some(node) = try_peephole_monus(&lower.def_bodies, base, step, args) {
      return Ok(node);
    }
    if let Some(node) = try_peephole_sg(&lower.def_bodies, base, step, args) {
      return Ok(node);
    }
    if let Some(node) = try_peephole_sgbar(&lower.def_bodies, base, step, args) {
      return Ok(node);
    }
    if let Some(node) = try_peephole_mult(&lower.def_bodies, base, step, args) {
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
      if let Some(body) = lower.def_bodies.get(name)
        && expr_node_count(body) <= 6
      {
        return lower_expr(lower, body, args);
      }
      Ok(Node::Call {
        name: name.clone(),
        args: args.to_vec(),
      })
    }

    Expr::Cn { f, gs } => {
      let mut inner = Vec::with_capacity(gs.len());
      for g in gs {
        inner.push(lower_expr(lower, g, args)?);
      }
      lower_expr(lower, f, &inner)
    }

    Expr::Pr { base, step } => lower_pr(lower, base, step, args),
    Expr::Mn { f } => lower_mn(lower, f, args),
  }
}

fn lower_pr(lower: &Lower, base: &Expr, step: &Expr, args: &[Node]) -> Result<Node, LowerError> {
  let base_arity = infer_arity(&lower.env, base)?;
  let (xs, y) = if base_arity == 0 {
    (&args[..0], &args[0])
  } else {
    let n = args.len();
    (&args[..n - 1], &args[n - 1])
  };

  let base_args: &[Node] = if base_arity == 0 { &[] } else { xs };
  let init = lower_expr(lower, base, base_args)?;

  let mut step_args: Vec<Node> = xs.iter().map(|x| shift_loop_depth(x, 1)).collect();
  step_args.push(Node::Counter(0));
  step_args.push(Node::Acc(0));

  let body = lower_expr(lower, step, &step_args)?;

  Ok(Node::Loop {
    bound: Box::new(y.clone()),
    init: Box::new(init),
    body: Box::new(body),
  })
}

fn lower_mn(lower: &Lower, f: &Expr, args: &[Node]) -> Result<Node, LowerError> {
  let mut search_args: Vec<Node> = args.iter().map(|a| shift_search_depth(a, 1)).collect();
  search_args.push(Node::Probe(0));

  let body = lower_expr(lower, f, &search_args)?;

  Ok(Node::Search {
    body: Box::new(body),
    limit: MU_STEP_LIMIT,
  })
}

// peephole optimizations we can perform when folding `Expr` -> `Node`

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

pub fn infer_arity(env: &HashMap<String, usize>, expr: &Expr) -> Result<usize, LowerError> {
  match expr {
    Expr::Const { arity, .. } => Ok(*arity),
    Expr::Succ => Ok(1),
    Expr::Id { k, n } => {
      if *k < 1 || *k > *n {
        return Err(LowerError::InvalidProjection { k: *k, n: *n });
      }
      Ok(*n)
    }
    Expr::Ref(name) => env
      .get(name)
      .copied()
      .ok_or_else(|| LowerError::Undefined(name.clone())),
    Expr::Cn { f, gs } => {
      let m = infer_arity(env, f)?;
      if gs.len() != m {
        return Err(LowerError::ArityMismatch {
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
          return Err(LowerError::CnArityMismatch);
        }
      }
      Ok(n)
    }
    Expr::Pr { base, step } => {
      let ba = infer_arity(env, base)?;
      let sa = infer_arity(env, step)?;
      let expected = if ba == 0 { 2 } else { ba + 2 };
      if sa != expected {
        return Err(LowerError::PrStepArity { expected, got: sa });
      }
      Ok(if ba == 0 { 1 } else { ba + 1 })
    }
    Expr::Mn { f } => {
      let inner = infer_arity(env, f)?;
      if inner < 1 {
        return Err(LowerError::MnArityTooSmall(inner));
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
  use crate::lexer::Lexer;
  use crate::parser::Parser;

  fn parse_src(input: &str) -> Program {
    let lexer = Lexer::new(input);
    let parser = Parser::new(lexer);
    parser.parse().expect("parse failed")
  }

  fn lower_src(input: &str) -> ProgramIR {
    let program = parse_src(input);
    Lower::new().lower(program).expect("lowering failed")
  }

  mod lower {
    use super::*;

    #[test]
    fn lower_const() {
      let ir = lower_src("def f = const(1, 0);\neval f(42);");
      assert_eq!(ir.funcs.len(), 1);
      assert_eq!(ir.funcs[0].body, Node::Iconst(0));
      assert_eq!(ir.evals.len(), 1);
      assert_eq!(ir.evals[0].body, Node::Iconst(0));
    }

    #[test]
    fn lower_succ() {
      let ir = lower_src("def f = s;\neval f(5);");
      assert_eq!(
        ir.funcs[0].body,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)))
      );
    }

    #[test]
    fn lower_pred_peephole() {
      let ir = lower_src("def pred = Pr[const(0, 0), id(1,2)];\neval pred(5);");
      match &ir.funcs[0].body {
        Node::Select { .. } => {}
        other => panic!("expected Select for pred peephole, got: {other}"),
      }
    }

    #[test]
    fn lower_add_peephole() {
      let ir = lower_src("def add = Pr[id(1,1), Cn[s, id(3,3)]];\neval add(3, 2);");
      assert_eq!(
        ir.funcs[0].body,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Arg(1)))
      );
    }

    #[test]
    fn lower_mult_peephole() {
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
    fn lower_pr_to_loop() {
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
    fn lower_mn_to_search() {
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
    fn lower_inline_combinator_eval() {
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
    fn lower_sg_peephole() {
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
    fn lower_sgbar_peephole() {
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
    fn lower_arity_mismatch() {
      let program = parse_src("def f = Pr[id(1,1), Cn[s, id(3,3)]];\neval f(1, 2, 3);");
      let err = Lower::new().lower(program).unwrap_err();
      match err {
        LowerError::ArityMismatch {
          expected: 2,
          got: 3,
          ..
        } => {}
        other => panic!("expected ArityMismatch, got: {other}"),
      }
    }

    #[test]
    fn lower_consumes_self() {
      let program = parse_src("def f = s;\neval f(5);");
      let ir = Lower::new().lower(program).unwrap();
      assert_eq!(
        ir.funcs[0].body,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)))
      );
      assert_eq!(ir.evals.len(), 1);
    }
  }
}
