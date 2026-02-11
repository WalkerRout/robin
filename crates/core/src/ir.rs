use std::fmt;

/// Catamorphism/abstract-fold over middle end IR
pub trait Visitor<T> {
  type Error;

  fn visit_program(&mut self, p: &ProgramIR) -> Result<T, Self::Error>;
  fn visit_func(&mut self, f: &FuncIR) -> Result<T, Self::Error>;
  fn visit_eval(&mut self, e: &EvalIR) -> Result<T, Self::Error>;
  fn visit_node(&mut self, n: &Node) -> Result<T, Self::Error>;
}

pub trait Visitable<T> {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>;
}

/// comparison operators for the IR
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cmp {
  Eq,
  Ne,
  Ult,
  Ugt,
  Uge,
}

impl fmt::Display for Cmp {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Cmp::Eq => write!(f, "eq"),
      Cmp::Ne => write!(f, "ne"),
      Cmp::Ult => write!(f, "ult"),
      Cmp::Ugt => write!(f, "ugt"),
      Cmp::Uge => write!(f, "uge"),
    }
  }
}

/// node in the middle-end intermediate representation
///
/// todo: bump allocators/arenas
///
/// arity inference, ref resolution, peephole optimization, and inlining are
/// all applied during lowering to this crappy ir
///
/// loop variables (Counter, Acc, Probe) use de bruijn depth indices to
/// handle nesting properly, depth of 0 means innermost enclosing loop/search,
/// depth of 1 means next outer layer, etc etc
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
  /// integer constant
  Iconst(i64),

  /// function parameter by index (0-based, always available via dominance)
  Arg(usize),

  /// integer addition
  Iadd(Box<Node>, Box<Node>),

  /// integer subtraction
  Isub(Box<Node>, Box<Node>),

  /// integer multiplication
  Imul(Box<Node>, Box<Node>),

  /// comparison (produces a boolean-like value for Select)
  Icmp(Cmp, Box<Node>, Box<Node>),

  /// conditional select, if cond then then_val else else_val
  Select {
    cond: Box<Node>,
    then_val: Box<Node>,
    else_val: Box<Node>,
  },

  /// call a named function
  Call { name: String, args: Vec<Node> },

  /// bounded for-loop (from primitive recursion)
  ///
  /// iterates `bound` times starting with `init` as the accumulator...
  /// `bound` and `init` are evaluated in the enclosing scope, but `body`
  /// is evaluated in a new scope where Counter(0) and Acc(0) are bound...
  Loop {
    bound: Box<Node>,
    init: Box<Node>,
    body: Box<Node>,
  },

  /// loop counter at de bruijn depth (0 is innermost Loop)
  Counter(usize),

  /// loop accumulator at de bruijn depth (0 is innermost Loop)
  Acc(usize),

  /// unbounded search (from mu-minimization)
  ///
  /// searches for the least y s.t. f is 0
  /// - `body` is evaluated in a new scope where Probe(0) is bound
  Search { body: Box<Node>, limit: i64 },

  /// search variable at de bruijn depth (again, 0 innermost Search)
  Probe(usize),
}

impl<T> Visitable<T> for Node {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_node(self)
  }
}

impl fmt::Display for Node {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Node::Iconst(v) => write!(f, "{v}"),
      Node::Arg(i) => write!(f, "arg({i})"),
      Node::Iadd(a, b) => write!(f, "iadd({a}, {b})"),
      Node::Isub(a, b) => write!(f, "isub({a}, {b})"),
      Node::Imul(a, b) => write!(f, "imul({a}, {b})"),
      Node::Icmp(cmp, a, b) => write!(f, "icmp.{cmp}({a}, {b})"),
      Node::Select {
        cond,
        then_val,
        else_val,
      } => write!(f, "select({cond}, {then_val}, {else_val})"),
      Node::Call { name, args } => {
        let args_str: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        write!(f, "call {name}({})", args_str.join(", "))
      }
      Node::Loop { bound, init, body } => {
        write!(f, "loop(bound={bound}, init={init}, {body})")
      }
      Node::Counter(d) => write!(f, "counter({d})"),
      Node::Acc(d) => write!(f, "acc({d})"),
      Node::Search { body, limit } => {
        write!(f, "search(limit={limit}, {body})")
      }
      Node::Probe(d) => write!(f, "probe({d})"),
    }
  }
}

/// A compiled `def` statement
#[derive(Debug)]
pub struct FuncIR {
  pub name: String,
  pub arity: usize,
  pub body: Node,
}

impl<T> Visitable<T> for FuncIR {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_func(self)
  }
}

/// A fully applied `eval` statement
#[derive(Debug)]
pub struct EvalIR {
  pub body: Node,
}

impl<T> Visitable<T> for EvalIR {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_eval(self)
  }
}

/// Lowered program consists of all function definitions and eval statements
#[derive(Debug)]
pub struct ProgramIR {
  pub funcs: Vec<FuncIR>,
  pub evals: Vec<EvalIR>,
}

impl<T> Visitable<T> for ProgramIR {
  fn fold<V>(&self, visitor: &mut V) -> Result<T, V::Error>
  where
    V: Visitor<T>,
  {
    visitor.visit_program(self)
  }
}

// increment all Counter/Acc depths by some amount, useful for pushing nodes down through nested Loops
// - doesnt touch Probe or Arg since they live in separate scopes
pub fn shift_loop_depth(node: &Node, delta: usize) -> Node {
  match node {
    Node::Counter(d) => Node::Counter(d + delta),
    Node::Acc(d) => Node::Acc(d + delta),

    // leaves dont change
    Node::Iconst(v) => Node::Iconst(*v),
    Node::Arg(i) => Node::Arg(*i),
    Node::Probe(d) => Node::Probe(*d),

    // recursive cases
    Node::Iadd(a, b) => Node::Iadd(
      Box::new(shift_loop_depth(a, delta)),
      Box::new(shift_loop_depth(b, delta)),
    ),
    Node::Isub(a, b) => Node::Isub(
      Box::new(shift_loop_depth(a, delta)),
      Box::new(shift_loop_depth(b, delta)),
    ),
    Node::Imul(a, b) => Node::Imul(
      Box::new(shift_loop_depth(a, delta)),
      Box::new(shift_loop_depth(b, delta)),
    ),
    Node::Icmp(cmp, a, b) => Node::Icmp(
      cmp.clone(),
      Box::new(shift_loop_depth(a, delta)),
      Box::new(shift_loop_depth(b, delta)),
    ),
    Node::Select {
      cond,
      then_val,
      else_val,
    } => Node::Select {
      cond: Box::new(shift_loop_depth(cond, delta)),
      then_val: Box::new(shift_loop_depth(then_val, delta)),
      else_val: Box::new(shift_loop_depth(else_val, delta)),
    },
    Node::Call { name, args } => Node::Call {
      name: name.clone(),
      args: args.iter().map(|a| shift_loop_depth(a, delta)).collect(),
    },
    Node::Loop { bound, init, body } => Node::Loop {
      bound: Box::new(shift_loop_depth(bound, delta)),
      init: Box::new(shift_loop_depth(init, delta)),
      body: Box::new(shift_loop_depth(body, delta)),
    },
    Node::Search { body, limit } => Node::Search {
      body: Box::new(shift_loop_depth(body, delta)),
      limit: *limit,
    },
  }
}

// increment all Probe depths by some amount, useful for pushing nodes down through nested Search
// - againt, doesnt touch Counter/Acc or Arg, they live in separate scopes
pub fn shift_search_depth(node: &Node, delta: usize) -> Node {
  match node {
    Node::Probe(d) => Node::Probe(d + delta),

    // leaves, no change
    Node::Iconst(v) => Node::Iconst(*v),
    Node::Arg(i) => Node::Arg(*i),
    Node::Counter(d) => Node::Counter(*d),
    Node::Acc(d) => Node::Acc(*d),

    // recursive cases
    Node::Iadd(a, b) => Node::Iadd(
      Box::new(shift_search_depth(a, delta)),
      Box::new(shift_search_depth(b, delta)),
    ),
    Node::Isub(a, b) => Node::Isub(
      Box::new(shift_search_depth(a, delta)),
      Box::new(shift_search_depth(b, delta)),
    ),
    Node::Imul(a, b) => Node::Imul(
      Box::new(shift_search_depth(a, delta)),
      Box::new(shift_search_depth(b, delta)),
    ),
    Node::Icmp(cmp, a, b) => Node::Icmp(
      cmp.clone(),
      Box::new(shift_search_depth(a, delta)),
      Box::new(shift_search_depth(b, delta)),
    ),
    Node::Select {
      cond,
      then_val,
      else_val,
    } => Node::Select {
      cond: Box::new(shift_search_depth(cond, delta)),
      then_val: Box::new(shift_search_depth(then_val, delta)),
      else_val: Box::new(shift_search_depth(else_val, delta)),
    },
    Node::Call { name, args } => Node::Call {
      name: name.clone(),
      args: args.iter().map(|a| shift_search_depth(a, delta)).collect(),
    },
    Node::Loop { bound, init, body } => Node::Loop {
      bound: Box::new(shift_search_depth(bound, delta)),
      init: Box::new(shift_search_depth(init, delta)),
      body: Box::new(shift_search_depth(body, delta)),
    },
    Node::Search { body, limit } => Node::Search {
      body: Box::new(shift_search_depth(body, delta)),
      limit: *limit,
    },
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  mod node {
    use super::*;

    #[test]
    fn shift_loop_depth_increments_counter_and_acc() {
      let node = Node::Iadd(Box::new(Node::Counter(0)), Box::new(Node::Acc(0)));
      let shifted = shift_loop_depth(&node, 1);
      assert_eq!(
        shifted,
        Node::Iadd(Box::new(Node::Counter(1)), Box::new(Node::Acc(1)))
      );
    }

    #[test]
    fn shift_loop_depth_does_not_touch_probe() {
      let node = Node::Iadd(Box::new(Node::Counter(0)), Box::new(Node::Probe(0)));
      let shifted = shift_loop_depth(&node, 1);
      assert_eq!(
        shifted,
        Node::Iadd(Box::new(Node::Counter(1)), Box::new(Node::Probe(0)))
      );
    }

    #[test]
    fn shift_loop_depth_does_not_touch_arg() {
      let node = Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Counter(2)));
      let shifted = shift_loop_depth(&node, 1);
      assert_eq!(
        shifted,
        Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Counter(3)))
      );
    }

    #[test]
    fn shift_loop_depth_recurses_into_loop_body() {
      let node = Node::Loop {
        bound: Box::new(Node::Arg(0)),
        init: Box::new(Node::Counter(0)),
        body: Box::new(Node::Iadd(
          Box::new(Node::Counter(0)),
          Box::new(Node::Acc(1)),
        )),
      };
      let shifted = shift_loop_depth(&node, 1);
      assert_eq!(
        shifted,
        Node::Loop {
          bound: Box::new(Node::Arg(0)),
          init: Box::new(Node::Counter(1)),
          body: Box::new(Node::Iadd(
            Box::new(Node::Counter(1)),
            Box::new(Node::Acc(2))
          )),
        }
      );
    }

    #[test]
    fn shift_search_depth_increments_probe() {
      let node = Node::Iadd(Box::new(Node::Probe(0)), Box::new(Node::Counter(0)));
      let shifted = shift_search_depth(&node, 1);
      assert_eq!(
        shifted,
        Node::Iadd(Box::new(Node::Probe(1)), Box::new(Node::Counter(0)))
      );
    }

    mod display {
      use super::*;

      #[test]
      fn simple_nodes() {
        assert_eq!(Node::Iconst(42).to_string(), "42");
        assert_eq!(Node::Arg(0).to_string(), "arg(0)");
        assert_eq!(Node::Counter(0).to_string(), "counter(0)");
        assert_eq!(Node::Acc(1).to_string(), "acc(1)");
        assert_eq!(Node::Probe(0).to_string(), "probe(0)");
      }

      #[test]
      fn compound_node() {
        let node = Node::Iadd(Box::new(Node::Arg(0)), Box::new(Node::Iconst(1)));
        assert_eq!(node.to_string(), "iadd(arg(0), 1)");
      }
    }
  }
}
