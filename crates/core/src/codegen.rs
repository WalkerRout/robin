use std::collections::HashMap;
use std::io;
use std::path::Path;

use cranelift_codegen::Context as ClifContext;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, BlockArg, Function, InstBuilder, TrapCode, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::{Decl, Def, Eval, Expr, Program, Visitable, Visitor};

const MU_STEP_LIMIT: i64 = 100_000;

type Env = HashMap<String, (FuncId, usize)>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
  #[error("undefined function - {0}")]
  Undefined(String),

  #[error("id({k},{n}) - k must satisfy 1 ≤ k ≤ n")]
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

  #[error("Mn - inner function must have arity ≥ 1, got {0}")]
  MnArityTooSmall(usize),
}

// todo: jit compilation (woof)
// - we are real close to module trait api here...

// we need borrows for module and env while some function builder is alive
struct EmitCtx<'a> {
  module: &'a mut ObjectModule,
  env: &'a Env,
  def_bodies: &'a HashMap<String, Expr>,
  printf_id: FuncId,
  exit_id: FuncId,
  fmt_div_id: DataId,
  ptr_ty: types::Type,
}

impl EmitCtx<'_> {
  // evaluates expr applied to args, returning resulting SSA value
  fn emit_apply(
    &mut self,
    builder: &mut FunctionBuilder,
    expr: &Expr,
    args: &[Value],
  ) -> Result<Value, EmitError> {
    // peepholes (see identifying helpers a bit further below)

    // pred = Pr[const(0,0), id(1,2)] -> select(y==0, 0, y-1)
    if let Expr::Pr { base, step } = expr {
      if matches!(base.as_ref(), Expr::Const { arity: 0, value: 0 })
        && matches!(step.as_ref(), Expr::Id { k: 1, n: 2 })
      {
        let y = args[0];
        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let sub = builder.ins().isub(y, one);
        let is_zero = builder.ins().icmp(IntCC::Equal, y, zero);
        return Ok(builder.ins().select(is_zero, zero, sub));
      }
    }

    // add = Pr[id(1,1), Cn[s, id(3,3)]] -> iadd
    if let Expr::Pr { base, step } = expr {
      if matches!(base.as_ref(), Expr::Id { k: 1, n: 1 }) {
        if let Expr::Cn { f, gs } = step.as_ref() {
          if matches!(f.as_ref(), Expr::Succ)
            && gs.len() == 1
            && matches!(gs[0], Expr::Id { k: 3, n: 3 })
          {
            return Ok(builder.ins().iadd(args[0], args[1]));
          }
        }
      }
    }

    // monus = Pr[id(1,1), Cn[pred, id(3,3)]] -> select(x>y, x-y, 0)
    if let Expr::Pr { base, step } = expr {
      if matches!(base.as_ref(), Expr::Id { k: 1, n: 1 }) {
        if let Expr::Cn { f, gs } = step.as_ref() {
          if is_pred_function(f, self.def_bodies)
            && gs.len() == 1
            && matches!(gs[0], Expr::Id { k: 3, n: 3 })
          {
            let x = args[0];
            let y = args[1];
            let zero = builder.ins().iconst(types::I64, 0);
            let sub = builder.ins().isub(x, y);
            let gt = builder.ins().icmp(IntCC::UnsignedGreaterThan, x, y);
            return Ok(builder.ins().select(gt, sub, zero));
          }
        }
      }
    }

    // sg = Pr[const(0,0), Cn[s, const(_,0)]] -> select(y!=0, 1, 0)
    if let Expr::Pr { base, step } = expr {
      if matches!(base.as_ref(), Expr::Const { arity: 0, value: 0 }) {
        if let Expr::Cn { f, gs } = step.as_ref() {
          if matches!(f.as_ref(), Expr::Succ)
            && gs.len() == 1
            && is_const_value(&gs[0], 0, self.def_bodies)
          {
            let y = args[0];
            let zero = builder.ins().iconst(types::I64, 0);
            let one = builder.ins().iconst(types::I64, 1);
            let nz = builder.ins().icmp(IntCC::NotEqual, y, zero);
            return Ok(builder.ins().select(nz, one, zero));
          }
        }
      }
    }

    // sgbar = Pr[const(0,1), const(_,0)] -> select(y==0, 1, 0)
    if let Expr::Pr { base, step } = expr {
      if matches!(base.as_ref(), Expr::Const { arity: 0, value: 1 })
        && is_const_value(step, 0, self.def_bodies)
      {
        let y = args[0];
        let zero = builder.ins().iconst(types::I64, 0);
        let one = builder.ins().iconst(types::I64, 1);
        let is_zero = builder.ins().icmp(IntCC::Equal, y, zero);
        return Ok(builder.ins().select(is_zero, one, zero));
      }
    }

    // mult = Pr[zero_fn, Cn[add_fn, id(3,3), id(1,3)]] -> imul
    if let Expr::Pr { base, step } = expr {
      if is_const_value(base, 0, self.def_bodies) {
        if let Expr::Cn { f, gs } = step.as_ref() {
          if is_add_function(f, self.def_bodies)
            && gs.len() == 2
            && matches!(gs[0], Expr::Id { k: 3, n: 3 })
            && matches!(gs[1], Expr::Id { k: 1, n: 3 })
          {
            return Ok(builder.ins().imul(args[0], args[1]));
          }
        }
      }
    }

    match expr {
      Expr::Const { value, .. } => Ok(builder.ins().iconst(types::I64, *value as i64)),

      Expr::Succ => {
        let one = builder.ins().iconst(types::I64, 1);
        Ok(builder.ins().iadd(args[0], one))
      }

      Expr::Id { k, .. } => Ok(args[*k - 1]),

      Expr::Ref(name) => {
        // inline small leaf functions to avoid call overhead
        if let Some(body) = self.def_bodies.get(name) {
          if expr_node_count(body) <= 6 {
            return self.emit_apply(builder, body, args);
          }
        }
        let (func_id, _) = self
          .env
          .get(name)
          .ok_or_else(|| EmitError::Undefined(name.clone()))?;
        let func_ref = self.module.declare_func_in_func(*func_id, builder.func);
        let call = builder.ins().call(func_ref, args);
        Ok(builder.inst_results(call)[0])
      }

      Expr::Cn { f, gs } => {
        let mut inner = Vec::with_capacity(gs.len());
        for g in gs {
          inner.push(self.emit_apply(builder, g, args)?);
        }
        self.emit_apply(builder, f, &inner)
      }

      Expr::Pr { base, step } => self.emit_pr(builder, base, step, args),
      Expr::Mn { f } => self.emit_mn(builder, f, args),
    }
  }

  // primitive recursion is just a for loop, yay termination!
  //
  // Pr[base, step](xs..., y):
  //   acc = base(xs...)
  //   for i in 0..y { acc = step(xs..., i, acc) }
  //   return acc
  fn emit_pr(
    &mut self,
    builder: &mut FunctionBuilder,
    base: &Expr,
    step: &Expr,
    args: &[Value],
  ) -> Result<Value, EmitError> {
    let base_arity = infer_arity(self.env, base)?;
    let (xs, y) = if base_arity == 0 {
      (&args[..0], args[0])
    } else {
      let n = args.len();
      (&args[..n - 1], args[n - 1])
    };

    let base_args: &[Value] = if base_arity == 0 { &[] } else { xs };
    let base_val = self.emit_apply(builder, base, base_args)?;

    // construct loop blocks
    let header = builder.create_block();
    let body = builder.create_block();
    let exit = builder.create_block();

    // header (i: i64, acc: i64)
    builder.append_block_param(header, types::I64);
    builder.append_block_param(header, types::I64);

    let zero = builder.ins().iconst(types::I64, 0);
    builder
      .ins()
      .jump(header, &[BlockArg::Value(zero), BlockArg::Value(base_val)]);

    // header
    builder.switch_to_block(header);
    let i_val = builder.block_params(header)[0];
    let acc_val = builder.block_params(header)[1];
    let cmp = builder.ins().icmp(IntCC::UnsignedLessThan, i_val, y);
    builder.ins().brif(cmp, body, &[], exit, &[]);

    // body
    builder.switch_to_block(body);
    let mut step_args: Vec<Value> = xs.to_vec();
    step_args.push(i_val);
    step_args.push(acc_val);
    let step_val = self.emit_apply(builder, step, &step_args)?;
    let one = builder.ins().iconst(types::I64, 1);
    let i_next = builder.ins().iadd(i_val, one);
    builder.ins().jump(
      header,
      &[BlockArg::Value(i_next), BlockArg::Value(step_val)],
    );

    // exit
    builder.switch_to_block(exit);

    Ok(acc_val)
  }

  // unbounded minimisation (μ-operator)...
  //
  // Mn[f](xs...) = least y such that f(xs..., y) == 0
  fn emit_mn(
    &mut self,
    builder: &mut FunctionBuilder,
    f: &Expr,
    args: &[Value],
  ) -> Result<Value, EmitError> {
    let guard = builder.create_block();
    let eval_bb = builder.create_block();
    let next = builder.create_block();
    let found = builder.create_block();
    let div = builder.create_block();

    // guard(y: i64)
    builder.append_block_param(guard, types::I64);

    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(guard, &[BlockArg::Value(zero)]);

    // guard for step limit before evaluating f
    builder.switch_to_block(guard);
    let y_val = builder.block_params(guard)[0];
    let limit = builder.ins().iconst(types::I64, MU_STEP_LIMIT);
    let over = builder
      .ins()
      .icmp(IntCC::UnsignedGreaterThanOrEqual, y_val, limit);
    builder.ins().brif(over, div, &[], eval_bb, &[]);

    // eval, compute f(args..., y)
    builder.switch_to_block(eval_bb);
    let mut full_args: Vec<Value> = args.to_vec();
    full_args.push(y_val);
    let f_val = self.emit_apply(builder, f, &full_args)?;
    let zero2 = builder.ins().iconst(types::I64, 0);
    let is_zero = builder.ins().icmp(IntCC::Equal, f_val, zero2);
    builder.ins().brif(is_zero, found, &[], next, &[]);

    // increment y
    builder.switch_to_block(next);
    let one = builder.ins().iconst(types::I64, 1);
    let y_next = builder.ins().iadd(y_val, one);
    builder.ins().jump(guard, &[BlockArg::Value(y_next)]);

    // we diverged, print message, exit 1
    builder.switch_to_block(div);
    self.emit_diverged(builder);

    // found! return least y
    builder.switch_to_block(found);

    Ok(y_val)
  }

  // emit "diverged" handler
  // - prints a message, calls exit 1, traps
  fn emit_diverged(&mut self, builder: &mut FunctionBuilder) {
    let printf_ref = self
      .module
      .declare_func_in_func(self.printf_id, builder.func);
    let gv = self
      .module
      .declare_data_in_func(self.fmt_div_id, builder.func);
    let fmt_ptr = builder.ins().global_value(self.ptr_ty, gv);
    // printf expects (ptr, i64), so we just pass dummy second arg (fmt string has no specifiers)
    let dummy = builder.ins().iconst(types::I64, 0);
    builder.ins().call(printf_ref, &[fmt_ptr, dummy]);
    // exit and trap to avoid returning to caller
    let exit_ref = self.module.declare_func_in_func(self.exit_id, builder.func);
    let exit_code = builder.ins().iconst(types::I32, 1);
    builder.ins().call(exit_ref, &[exit_code]);
    builder.ins().trap(TrapCode::unwrap_user(1));
  }
}

// arity inference for all expressions
fn infer_arity(env: &Env, expr: &Expr) -> Result<usize, EmitError> {
  match expr {
    Expr::Const { arity, .. } => Ok(*arity),
    Expr::Succ => Ok(1),
    Expr::Id { k, n } => {
      if *k < 1 || *k > *n {
        return Err(EmitError::InvalidProjection { k: *k, n: *n });
      }
      Ok(*n)
    }
    Expr::Ref(name) => env
      .get(name)
      .map(|(_, arity)| *arity)
      .ok_or_else(|| EmitError::Undefined(name.clone())),
    Expr::Cn { f, gs } => {
      let m = infer_arity(env, f)?;
      if gs.len() != m {
        return Err(EmitError::ArityMismatch {
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
          return Err(EmitError::CnArityMismatch);
        }
      }
      Ok(n)
    }
    Expr::Pr { base, step } => {
      let ba = infer_arity(env, base)?;
      let sa = infer_arity(env, step)?;
      let expected = if ba == 0 { 2 } else { ba + 2 };
      if sa != expected {
        return Err(EmitError::PrStepArity { expected, got: sa });
      }
      Ok(if ba == 0 { 1 } else { ba + 1 })
    }
    Expr::Mn { f } => {
      let inner = infer_arity(env, f)?;
      if inner < 1 {
        return Err(EmitError::MnArityTooSmall(inner));
      }
      Ok(inner - 1)
    }
  }
}

// peephole helpers
// - we can resolve Refs through def_bodies to detect known patterns

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
      .map_or(false, |body| is_const_value(body, val, def_bodies)),
    Expr::Cn { f, gs } => {
      // Cn[const_fn, ...], if const_fn always returns val, so does the Cn
      if is_const_value(f, val, def_bodies) {
        return true;
      }
      // Cn[s, g], if g always returns val-1, then s(g(...)) = val
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
  if let Expr::Pr { base, step } = resolved {
    if matches!(base.as_ref(), Expr::Id { k: 1, n: 1 }) {
      if let Expr::Cn { f, gs } = step.as_ref() {
        return matches!(f.as_ref(), Expr::Succ)
          && matches!(gs.get(0), Some(Expr::Id { k: 3, n: 3 }));
      }
    }
  }
  false
}

#[derive(thiserror::Error, Debug)]
pub enum CodegenError {
  #[error("code emission failed - {0}")]
  Emit(#[from] EmitError),

  #[error("settings misconfigured - {0}")]
  Settings(#[from] cranelift_codegen::settings::SetError),

  #[error("isa issue - {0}")]
  Isa(#[from] cranelift_codegen::CodegenError),

  #[error("module error - {0}")]
  Module(#[from] cranelift_module::ModuleError),

  #[error("object error - {0}")]
  Object(#[from] cranelift_object::object::write::Error),

  #[error("io error - {0}")]
  Io(#[from] io::Error),

  #[error("unsupported host - {0}")]
  UnsupportedHost(String),
}

/// Code generator that uses Cranelift to compile robin ASTs to object files
pub struct Codegen {
  module: ObjectModule,
  builder_ctx: FunctionBuilderContext,
  env: Env,
  /// Stored definition bodies for inlining and peephole resolution
  def_bodies: HashMap<String, Expr>,
  printf_id: FuncId,
  exit_id: FuncId,
  fmt_id: DataId,
  fmt_div_id: DataId,
  ptr_ty: types::Type,
  /// Eval statements are collected during the tree walk and compiled into `main` at the end...
  pending_evals: Vec<(Expr, Vec<u64>)>,
}

impl Codegen {
  /// Create a new Cranelift codegen targeting the host machine.
  ///
  /// # Errors
  ///
  /// Returns an error string if target lookup or module creation fails.
  pub fn new(opt_level: &str) -> Result<Self, CodegenError> {
    let mut settings_builder = settings::builder();
    settings_builder.set("opt_level", opt_level)?;
    settings_builder.set("preserve_frame_pointers", "false")?;

    // windows needs is_pic=false for coff object files...
    if cfg!(windows) {
      settings_builder.set("is_pic", "false")?;
    }

    let flags = settings::Flags::new(settings_builder);
    let isa = cranelift_native::builder()
      .map_err(|e| CodegenError::UnsupportedHost(e.to_string()))?
      .finish(flags)?;

    let ptr_ty = isa.pointer_type();

    let builder = ObjectBuilder::new(isa, "robin", cranelift_module::default_libcall_names())?;
    let mut module = ObjectModule::new(builder);

    // c ffi
    // - todo: implement our own target specific syscall shims

    // printf: we declare a fixed (ptr, i64) -> i32 signature...
    // - we dont have access to varargs so we use a workaround
    let mut printf_sig = module.make_signature();
    printf_sig.params.push(AbiParam::new(ptr_ty));
    printf_sig.params.push(AbiParam::new(types::I64));
    printf_sig.returns.push(AbiParam::new(types::I32));
    let printf_id = module.declare_function("printf", Linkage::Import, &printf_sig)?;

    // exit: (i32) -> void
    let mut exit_sig = module.make_signature();
    exit_sig.params.push(AbiParam::new(types::I32));
    let exit_id = module.declare_function("exit", Linkage::Import, &exit_sig)?;

    // format strings embedded in .fmt and .fmt.div sections...
    let fmt_id = Self::define_data(&mut module, ".fmt", b"%llu\n\0")?;
    let fmt_div_id = Self::define_data(
      &mut module,
      ".fmt.div",
      b"diverged: mu-minimization did not converge\n\0",
    )?;

    Ok(Self {
      module,
      builder_ctx: FunctionBuilderContext::new(),
      env: HashMap::new(),
      def_bodies: HashMap::new(),
      printf_id,
      exit_id,
      fmt_id,
      fmt_div_id,
      ptr_ty,
      pending_evals: Vec::new(),
    })
  }

  fn define_data(
    module: &mut ObjectModule,
    name: &str,
    bytes: &[u8],
  ) -> Result<DataId, CodegenError> {
    let data_id = module.declare_data(name, Linkage::Local, false, false)?;
    let mut desc = DataDescription::new();
    desc.define(bytes.to_vec().into_boxed_slice());
    module.define_data(data_id, &desc)?;
    Ok(data_id)
  }

  fn infer_arity(&self, expr: &Expr) -> Result<usize, CodegenError> {
    Ok(infer_arity(&self.env, expr)?)
  }

  fn build_def(&mut self, name: &str, arity: usize, body: &Expr) -> Result<(), CodegenError> {
    // declare a new function in the module
    let mut sig = self.module.make_signature();
    for _ in 0..arity {
      sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));

    let func_id = self.module.declare_function(name, Linkage::Export, &sig)?;

    // build the function body
    let mut func = Function::new();
    func.signature = sig;

    {
      let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_ctx);
      let entry = builder.create_block();
      builder.append_block_params_for_function_params(entry);
      builder.switch_to_block(entry);

      let arg_values: Vec<Value> = (0..arity).map(|i| builder.block_params(entry)[i]).collect();

      let result = {
        let mut ctx = EmitCtx {
          module: &mut self.module,
          env: &self.env,
          def_bodies: &self.def_bodies,
          printf_id: self.printf_id,
          exit_id: self.exit_id,
          fmt_div_id: self.fmt_div_id,
          ptr_ty: self.ptr_ty,
        };
        ctx.emit_apply(&mut builder, body, &arg_values)?
      };

      builder.ins().return_(&[result]);
      builder.seal_all_blocks();
      builder.finalize();
    }

    let mut clif_ctx = ClifContext::for_function(func);
    self.module.define_function(func_id, &mut clif_ctx)?;

    self.env.insert(name.to_string(), (func_id, arity));

    Ok(())
  }

  // build main from collected evals...
  fn build_main(&mut self) -> Result<(), CodegenError> {
    let mut sig = self.module.make_signature();
    sig.returns.push(AbiParam::new(types::I32));

    let main_id = self
      .module
      .declare_function("main", Linkage::Export, &sig)?;

    let mut func = Function::new();
    func.signature = sig;

    let evals = std::mem::take(&mut self.pending_evals);

    {
      let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_ctx);
      let entry = builder.create_block();
      builder.switch_to_block(entry);

      for (expr, args) in &evals {
        let arg_values: Vec<Value> = args
          .iter()
          .map(|a| builder.ins().iconst(types::I64, *a as i64))
          .collect();

        let result = {
          let mut ctx = EmitCtx {
            module: &mut self.module,
            env: &self.env,
            def_bodies: &self.def_bodies,
            printf_id: self.printf_id,
            exit_id: self.exit_id,
            fmt_div_id: self.fmt_div_id,
            ptr_ty: self.ptr_ty,
          };
          ctx.emit_apply(&mut builder, expr, &arg_values)?
        };

        // printf("%llu\n", result)
        let printf_ref = self
          .module
          .declare_func_in_func(self.printf_id, builder.func);
        let gv = self.module.declare_data_in_func(self.fmt_id, builder.func);
        let fmt_ptr = builder.ins().global_value(self.ptr_ty, gv);
        builder.ins().call(printf_ref, &[fmt_ptr, result]);
      }

      let zero = builder.ins().iconst(types::I32, 0);
      builder.ins().return_(&[zero]);

      builder.seal_all_blocks();
      builder.finalize();
    }

    let mut clif_ctx = ClifContext::for_function(func);
    self.module.define_function(main_id, &mut clif_ctx)?;

    Ok(())
  }

  /// Consume the codegen and write the object file to `path`
  pub fn write_object_file(self, path: &Path) -> Result<(), CodegenError> {
    use std::fs;
    let product = self.module.finish();
    let bytes = product.emit()?;
    Ok(fs::write(path, bytes)?)
  }

  /// Consume the codegen and return the raw object bytes
  pub fn finish_to_bytes(self) -> Result<Vec<u8>, CodegenError> {
    let product = self.module.finish();
    Ok(product.emit()?)
  }
}

impl Visitor<()> for Codegen {
  type Error = CodegenError;

  fn visit_program(&mut self, program: &Program) -> Result<(), Self::Error> {
    // process all declarations (defs get compiled, evals get queued)
    for decl in &program.decls {
      decl.accept(self)?;
    }
    // build main from queued statements
    self.build_main()
  }

  fn visit_decl(&mut self, decl: &Decl) -> Result<(), Self::Error> {
    match decl {
      Decl::Def(def) => def.accept(self),
      Decl::Eval(eval) => eval.accept(self),
    }
  }

  fn visit_def(&mut self, def: &Def) -> Result<(), Self::Error> {
    let arity = self.infer_arity(&def.body)?;
    self.build_def(&def.name, arity, &def.body)?;
    self.def_bodies.insert(def.name.clone(), def.body.clone());
    Ok(())
  }

  fn visit_eval(&mut self, eval: &Eval) -> Result<(), Self::Error> {
    // we eagerly validate arity
    let arity = self.infer_arity(&eval.func)?;
    if arity != eval.args.len() {
      return Err(
        EmitError::ArityMismatch {
          name: format!("{}", eval.func),
          expected: arity,
          got: eval.args.len(),
        }
        .into(),
      );
    }

    // queue into main -> will be compiled in build_main
    self
      .pending_evals
      .push((eval.func.clone(), eval.args.clone()));

    Ok(())
  }

  fn visit_expr(&mut self, _expr: &Expr) -> Result<(), Self::Error> {
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lexer::Lexer;
  use crate::parser::Parser;

  // shitty pipeline fixture to test proper compilation
  fn compile(input: &str) -> Vec<u8> {
    let lexer = Lexer::new(input);
    let parser = Parser::new(lexer);
    let program = parser.parse().expect("parse failed");
    let mut codegen = Codegen::new("speed").expect("codegen init failed");
    program.accept(&mut codegen).expect("codegen failed");
    codegen.finish_to_bytes().expect("emit failed")
  }

  fn compile_ok(input: &str) {
    let bytes = compile(input);
    assert!(!bytes.is_empty(), "object file should not be empty");
  }

  #[test]
  fn emits_const() {
    compile_ok("def f = const(1, 0);\neval f(42);");
  }

  #[test]
  fn emits_succ() {
    compile_ok("def f = s;\neval f(5);");
  }

  #[test]
  fn emits_composition() {
    compile_ok("def f = Cn[s, s];\neval f(5);");
  }

  #[test]
  fn emits_pred() {
    compile_ok("def pred = Pr[const(0, 0), id(1,2)];\neval pred(5);");
  }

  #[test]
  fn emits_add() {
    compile_ok("def add = Pr[id(1,1), Cn[s, id(3,3)]];\neval add(3, 2);");
  }

  #[test]
  fn emits_monus() {
    compile_ok(
      "def pred = Pr[const(0, 0), id(1,2)];\n\
       def monus = Pr[id(1,1), Cn[pred, id(3,3)]];\n\
       eval monus(10, 3);",
    );
  }

  #[test]
  fn emits_pr_loop_generic() {
    compile_ok(
      "def z = const(1, 0);\n\
       def pred = Pr[const(0, 0), id(1,2)];\n\
       def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
       def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
       def exp = Pr[const(1,1), Cn[mult, id(1,3), id(3,3)]];\n\
       eval exp(2, 10);",
    );
  }

  #[test]
  fn emits_mn_search() {
    compile_ok(
      "def z = const(1, 0);\n\
       def pred = Pr[const(0, 0), id(1,2)];\n\
       def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
       def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
       def monus = Pr[id(1,1), Cn[pred, id(3,3)]];\n\
       def isqrt = Mn[Cn[monus, id(1,2), Cn[mult, id(2,2), id(2,2)]]];\n\
       eval isqrt(9);",
    );
  }

  #[test]
  fn module_has_main() {
    compile_ok("def f = const(1, 0);");
  }

  #[test]
  fn eval_inline_combinator() {
    compile_ok("eval Cn[s, s](5);");
  }

  #[test]
  fn emits_sg() {
    compile_ok(
      "def sg = Pr[const(0, 0), Cn[s, const(2, 0)]];\n\
       eval sg(0);\n\
       eval sg(5);",
    );
  }

  #[test]
  fn emits_sgbar() {
    compile_ok(
      "def sgbar = Pr[const(0, 1), const(2, 0)];\n\
       eval sgbar(0);\n\
       eval sgbar(3);",
    );
  }

  #[test]
  fn emits_mult_peephole() {
    compile_ok(
      "def z = const(1, 0);\n\
       def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
       def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
       eval mult(3, 4);",
    );
  }

  #[test]
  fn emits_fact_with_inlining() {
    compile_ok(
      "def z = const(1, 0);\n\
       def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
       def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
       def fact = Pr[const(0, 1), Cn[mult, Cn[s, id(1,2)], id(2,2)]];\n\
       eval fact(5);",
    );
  }
}
