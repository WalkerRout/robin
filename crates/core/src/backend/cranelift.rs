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

#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};

use crate::ir::{Cmp, EvalIR, FuncIR, Node, ProgramIR, Visitable, Visitor};
use crate::pass::Pass;

#[derive(thiserror::Error, Debug)]
pub enum CodegenError {
  #[error("settings misconfigured - {0}")]
  Settings(#[from] cranelift_codegen::settings::SetError),

  #[error("isa issue - {0}")]
  Isa(#[from] cranelift_codegen::CodegenError),

  // we box this variant cause its massive...
  #[error("module error - {0}")]
  Module(Box<cranelift_module::ModuleError>),

  #[error("object error - {0}")]
  Object(#[from] cranelift_object::object::write::Error),

  #[error("io error - {0}")]
  Io(#[from] io::Error),

  #[error("unsupported host - {0}")]
  UnsupportedHost(String),
}

// impl from ourselves since we box the module error...
impl From<cranelift_module::ModuleError> for CodegenError {
  fn from(e: cranelift_module::ModuleError) -> Self {
    CodegenError::Module(Box::new(e))
  }
}

// borrow split context for emission
// - borrows module and func_ids while some FunctionBuilder lives
struct EmitCtx<'a, M> {
  module: &'a mut M,
  func_ids: &'a HashMap<String, FuncId>,
  printf_id: FuncId,
  exit_id: FuncId,
  fmt_div_id: DataId,
  ptr_ty: types::Type,
}

impl<M> EmitCtx<'_, M>
where
  M: Module,
{
  // walk tree and emit cranelift IR, returning resulting SSA value
  fn emit_node(
    &mut self,
    builder: &mut FunctionBuilder,
    node: &Node,
    func_args: &[Value],
    loop_stack: &[(Value, Value)],
    search_stack: &[Value],
  ) -> Result<Value, CodegenError> {
    match node {
      Node::Iconst(v) => Ok(builder.ins().iconst(types::I64, *v)),
      Node::Arg(i) => Ok(func_args[*i]),
      Node::Counter(depth) => {
        let idx = loop_stack.len() - 1 - depth;
        Ok(loop_stack[idx].0)
      }
      Node::Acc(depth) => {
        let idx = loop_stack.len() - 1 - depth;
        Ok(loop_stack[idx].1)
      }
      Node::Probe(depth) => {
        let idx = search_stack.len() - 1 - depth;
        Ok(search_stack[idx])
      }
      Node::Iadd(a, b) => {
        let va = self.emit_node(builder, a, func_args, loop_stack, search_stack)?;
        let vb = self.emit_node(builder, b, func_args, loop_stack, search_stack)?;
        Ok(builder.ins().iadd(va, vb))
      }
      Node::Isub(a, b) => {
        let va = self.emit_node(builder, a, func_args, loop_stack, search_stack)?;
        let vb = self.emit_node(builder, b, func_args, loop_stack, search_stack)?;
        Ok(builder.ins().isub(va, vb))
      }
      Node::Imul(a, b) => {
        let va = self.emit_node(builder, a, func_args, loop_stack, search_stack)?;
        let vb = self.emit_node(builder, b, func_args, loop_stack, search_stack)?;
        Ok(builder.ins().imul(va, vb))
      }
      Node::Icmp(cmp, a, b) => {
        let va = self.emit_node(builder, a, func_args, loop_stack, search_stack)?;
        let vb = self.emit_node(builder, b, func_args, loop_stack, search_stack)?;
        let cc = match cmp {
          Cmp::Eq => IntCC::Equal,
          Cmp::Ne => IntCC::NotEqual,
          Cmp::Ult => IntCC::UnsignedLessThan,
          Cmp::Ugt => IntCC::UnsignedGreaterThan,
          Cmp::Uge => IntCC::UnsignedGreaterThanOrEqual,
        };
        let result = builder.ins().icmp(cc, va, vb);
        Ok(result)
      }
      Node::Select {
        cond,
        then_val,
        else_val,
      } => {
        let vc = self.emit_node(builder, cond, func_args, loop_stack, search_stack)?;
        let vt = self.emit_node(builder, then_val, func_args, loop_stack, search_stack)?;
        let ve = self.emit_node(builder, else_val, func_args, loop_stack, search_stack)?;
        Ok(builder.ins().select(vc, vt, ve))
      }
      Node::Call { name, args } => {
        let values: Vec<Value> = args
          .iter()
          .map(|a| self.emit_node(builder, a, func_args, loop_stack, search_stack))
          .collect::<Result<_, _>>()?;
        let func_id = self.func_ids[name];
        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, &values);
        Ok(builder.inst_results(call)[0])
      }
      Node::Loop { bound, init, body } => self.emit_loop(
        builder,
        bound,
        init,
        body,
        func_args,
        loop_stack,
        search_stack,
      ),
      Node::Search { body, limit } => {
        self.emit_search(builder, body, *limit, func_args, loop_stack, search_stack)
      }
    }
  }

  // primitive recursion, only for loops...
  //
  // acc = new
  // for i in 0..bound { acc = body(i, acc) }
  // return acc
  //
  // (sorry clippy)
  #[allow(clippy::too_many_arguments)]
  fn emit_loop(
    &mut self,
    builder: &mut FunctionBuilder,
    bound: &Node,
    new: &Node,
    body: &Node,
    func_args: &[Value],
    loop_stack: &[(Value, Value)],
    search_stack: &[Value],
  ) -> Result<Value, CodegenError> {
    let bound_val = self.emit_node(builder, bound, func_args, loop_stack, search_stack)?;
    let new_val = self.emit_node(builder, new, func_args, loop_stack, search_stack)?;

    let header = builder.create_block();
    let loop_body = builder.create_block();
    let exit = builder.create_block();

    // loops need (counter, acc)
    builder.append_block_param(header, types::I64);
    builder.append_block_param(header, types::I64);

    let zero = builder.ins().iconst(types::I64, 0);
    builder
      .ins()
      .jump(header, &[BlockArg::Value(zero), BlockArg::Value(new_val)]);

    // header
    // - check counter < bound
    builder.switch_to_block(header);
    let i_val = builder.block_params(header)[0];
    let acc_val = builder.block_params(header)[1];
    let cmp = builder
      .ins()
      .icmp(IntCC::UnsignedLessThan, i_val, bound_val);
    builder.ins().brif(cmp, loop_body, &[], exit, &[]);

    // body
    // - evaluate with this loop's variables pushed onto the stack
    builder.switch_to_block(loop_body);
    let mut new_loop_stack = loop_stack.to_vec();
    new_loop_stack.push((i_val, acc_val));
    let step_val = self.emit_node(builder, body, func_args, &new_loop_stack, search_stack)?;
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

  // unbounded search is bounded in practice since we need to converge eventually...
  //
  // y = 0
  // while y < limit:
  //   if body(y) == 0: return y
  //   y += 1
  // diverge!
  fn emit_search(
    &mut self,
    builder: &mut FunctionBuilder,
    body: &Node,
    limit: i64,
    func_args: &[Value],
    loop_stack: &[(Value, Value)],
    search_stack: &[Value],
  ) -> Result<Value, CodegenError> {
    let guard = builder.create_block();
    let eval_bb = builder.create_block();
    let next = builder.create_block();
    let found = builder.create_block();
    let div = builder.create_block();

    // guard(y: i64)
    builder.append_block_param(guard, types::I64);

    let zero = builder.ins().iconst(types::I64, 0);
    builder.ins().jump(guard, &[BlockArg::Value(zero)]);

    // guard, check step limit
    builder.switch_to_block(guard);
    let y_val = builder.block_params(guard)[0];
    let limit_val = builder.ins().iconst(types::I64, limit);
    let over = builder
      .ins()
      .icmp(IntCC::UnsignedGreaterThanOrEqual, y_val, limit_val);
    builder.ins().brif(over, div, &[], eval_bb, &[]);

    // eval
    // - compute body with this search's probe pushed onto the stack
    builder.switch_to_block(eval_bb);
    let mut new_search_stack = search_stack.to_vec();
    new_search_stack.push(y_val);
    let f_val = self.emit_node(builder, body, func_args, loop_stack, &new_search_stack)?;
    let zero2 = builder.ins().iconst(types::I64, 0);
    let is_zero = builder.ins().icmp(IntCC::Equal, f_val, zero2);
    builder.ins().brif(is_zero, found, &[], next, &[]);

    // increment y
    builder.switch_to_block(next);
    let one = builder.ins().iconst(types::I64, 1);
    let y_next = builder.ins().iadd(y_val, one);
    builder.ins().jump(guard, &[BlockArg::Value(y_next)]);

    // diverged! print our message, exit 1, trap
    builder.switch_to_block(div);
    self.emit_diverged(builder);

    // found target, return least y
    builder.switch_to_block(found);

    Ok(y_val)
  }

  // emit backend internal "diverged" handler
  fn emit_diverged(&mut self, builder: &mut FunctionBuilder) {
    let printf_ref = self
      .module
      .declare_func_in_func(self.printf_id, builder.func);
    let gv = self
      .module
      .declare_data_in_func(self.fmt_div_id, builder.func);
    let fmt_ptr = builder.ins().global_value(self.ptr_ty, gv);
    let dummy = builder.ins().iconst(types::I64, 0);
    builder.ins().call(printf_ref, &[fmt_ptr, dummy]);
    let exit_ref = self.module.declare_func_in_func(self.exit_id, builder.func);
    let exit_code = builder.ins().iconst(types::I32, 1);
    builder.ins().call(exit_ref, &[exit_code]);
    builder.ins().trap(TrapCode::unwrap_user(1));
  }
}

// cranelift codegen, generic over the module backend
#[allow(dead_code)]
pub struct Codegen<M> {
  module: M,
  builder_ctx: FunctionBuilderContext,
  printf_id: FuncId,
  exit_id: FuncId,
  fmt_id: DataId,
  fmt_div_id: DataId,
  ptr_ty: types::Type,
  func_ids: HashMap<String, FuncId>,
  eval_ids: Vec<FuncId>,
}

impl<M> Codegen<M>
where
  M: Module,
{
  // declares printf, exit, and format string data objects
  fn new(mut module: M) -> Result<Self, CodegenError> {
    let ptr_ty = module.isa().pointer_type();

    // c ffi
    let mut printf_sig = module.make_signature();
    printf_sig.params.push(AbiParam::new(ptr_ty));
    printf_sig.params.push(AbiParam::new(types::I64));
    printf_sig.returns.push(AbiParam::new(types::I32));
    let printf_id = module.declare_function("printf", Linkage::Import, &printf_sig)?;

    let mut exit_sig = module.make_signature();
    exit_sig.params.push(AbiParam::new(types::I32));
    let exit_id = module.declare_function("exit", Linkage::Import, &exit_sig)?;

    let fmt_id = Self::define_data(&mut module, ".fmt", b"%llu\n\0")?;
    let fmt_div_id = Self::define_data(
      &mut module,
      ".fmt.div",
      b"diverged: mu-minimization did not converge\n\0",
    )?;

    Ok(Self {
      module,
      builder_ctx: FunctionBuilderContext::new(),
      printf_id,
      exit_id,
      fmt_id,
      fmt_div_id,
      ptr_ty,
      func_ids: HashMap::new(),
      eval_ids: Vec::new(),
    })
  }

  fn define_data(module: &mut M, name: &str, bytes: &[u8]) -> Result<DataId, CodegenError> {
    let data_id = module.declare_data(name, Linkage::Local, false, false)?;
    let mut desc = DataDescription::new();
    desc.define(bytes.to_vec().into_boxed_slice());
    module.define_data(data_id, &desc)?;
    Ok(data_id)
  }

  fn build_func(&mut self, func_ir: &FuncIR) -> Result<(), CodegenError> {
    let func_id = self.func_ids[&func_ir.name];

    let mut sig = self.module.make_signature();
    for _ in 0..func_ir.arity {
      sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));

    let mut func = Function::new();
    func.signature = sig;

    {
      let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_ctx);
      let entry = builder.create_block();
      builder.append_block_params_for_function_params(entry);
      builder.switch_to_block(entry);

      let arg_values: Vec<Value> = (0..func_ir.arity)
        .map(|i| builder.block_params(entry)[i])
        .collect();

      let result = {
        let mut ctx = EmitCtx {
          module: &mut self.module,
          func_ids: &self.func_ids,
          printf_id: self.printf_id,
          exit_id: self.exit_id,
          fmt_div_id: self.fmt_div_id,
          ptr_ty: self.ptr_ty,
        };
        ctx.emit_node(&mut builder, &func_ir.body, &arg_values, &[], &[])?
      };

      builder.ins().return_(&[result]);
      builder.seal_all_blocks();
      builder.finalize();
    }

    let mut clif_ctx = ClifContext::for_function(func);
    self.module.define_function(func_id, &mut clif_ctx)?;

    Ok(())
  }

  fn build_main(&mut self, evals: &[EvalIR]) -> Result<(), CodegenError> {
    let mut sig = self.module.make_signature();
    sig.returns.push(AbiParam::new(types::I32));

    let main_id = self
      .module
      .declare_function("main", Linkage::Export, &sig)?;

    let mut func = Function::new();
    func.signature = sig;

    {
      let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_ctx);
      let entry = builder.create_block();
      builder.switch_to_block(entry);

      for eval in evals {
        // eval bodies fully applied (args already Iconst), so func_args is empty
        let result = {
          let mut ctx = EmitCtx {
            module: &mut self.module,
            func_ids: &self.func_ids,
            printf_id: self.printf_id,
            exit_id: self.exit_id,
            fmt_div_id: self.fmt_div_id,
            ptr_ty: self.ptr_ty,
          };
          ctx.emit_node(&mut builder, &eval.body, &[], &[], &[])?
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
}

impl<M> Visitor<()> for Codegen<M>
where
  M: Module,
{
  type Error = CodegenError;

  fn visit_program(&mut self, program: &ProgramIR) -> Result<(), CodegenError> {
    // pass 1 - declare all functions (forward declarations)
    for func_ir in &program.funcs {
      let mut sig = self.module.make_signature();
      for _ in 0..func_ir.arity {
        sig.params.push(AbiParam::new(types::I64));
      }
      sig.returns.push(AbiParam::new(types::I64));
      let func_id = self
        .module
        .declare_function(&func_ir.name, Linkage::Export, &sig)?;
      self.func_ids.insert(func_ir.name.clone(), func_id);
    }

    // pass 2 - define function bodies
    for func_ir in &program.funcs {
      func_ir.fold(self)?;
    }

    Ok(())
  }

  fn visit_func(&mut self, func_ir: &FuncIR) -> Result<(), CodegenError> {
    self.build_func(func_ir)
  }

  fn visit_eval(&mut self, _eval: &EvalIR) -> Result<(), CodegenError> {
    Ok(())
  }

  fn visit_node(&mut self, _node: &Node) -> Result<(), CodegenError> {
    Ok(())
  }
}

// aot backend, compiles to object files
pub type AotBackend = Codegen<ObjectModule>;

impl Codegen<ObjectModule> {
  // Create a new aot backend, targeting host machine
  pub fn new_aot(opt_level: &str) -> Result<Self, CodegenError> {
    let mut settings_builder = settings::builder();
    settings_builder.set("opt_level", opt_level)?;
    settings_builder.set("preserve_frame_pointers", "false")?;

    if cfg!(windows) {
      settings_builder.set("is_pic", "false")?;
    }

    let flags = settings::Flags::new(settings_builder);
    let isa = cranelift_native::builder()
      .map_err(|e| CodegenError::UnsupportedHost(e.to_string()))?
      .finish(flags)?;

    let builder = ObjectBuilder::new(isa, "robin", cranelift_module::default_libcall_names())?;
    let module = ObjectModule::new(builder);

    Self::new(module)
  }

  /// Consume the backend and write the object file to `path`
  pub fn write_object_file(self, path: &Path) -> Result<(), CodegenError> {
    use std::fs;
    let product = self.module.finish();
    let bytes = product.emit()?;
    Ok(fs::write(path, bytes)?)
  }

  /// Consume backend and return raw object bytes
  pub fn into_bytes(self) -> Result<Vec<u8>, CodegenError> {
    let product = self.module.finish();
    Ok(product.emit()?)
  }
}

impl Pass<ProgramIR> for Codegen<ObjectModule> {
  type Output = Codegen<ObjectModule>;
  type Error = CodegenError;

  fn run(mut self, ir: &ProgramIR) -> Result<Codegen<ObjectModule>, CodegenError> {
    ir.fold(&mut self)?;
    self.build_main(&ir.evals)?;
    Ok(self)
  }
}

#[cfg(feature = "jit")]
// jit backend, compiles to memory and runs directly in process
pub type JitBackend = Codegen<JITModule>;

#[cfg(feature = "jit")]
impl Codegen<JITModule> {
  /// Create a new jit backend, targeting host machine
  pub fn new_jit(opt_level: &str) -> Result<Self, CodegenError> {
    let mut settings_builder = settings::builder();
    settings_builder.set("opt_level", opt_level)?;
    settings_builder.set("preserve_frame_pointers", "false")?;

    let flags = settings::Flags::new(settings_builder);
    let isa = cranelift_native::builder()
      .map_err(|e| CodegenError::UnsupportedHost(e.to_string()))?
      .finish(flags)?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // register libc symbols so jit can resolve printf/exit from host process
    unsafe extern "C" {
      safe fn printf(fmt: *const i8, ...) -> i32;
      safe fn exit(code: i32);
    }
    jit_builder.symbol("printf", printf as *const u8);
    jit_builder.symbol("exit", exit as *const u8);

    let module = JITModule::new(jit_builder);
    Self::new(module)
  }

  // compile each eval as standalone no argument function returning and i64
  fn build_evals(&mut self, evals: &[EvalIR]) -> Result<(), CodegenError> {
    for (i, eval) in evals.iter().enumerate() {
      let name = format!("__robin_eval_{i}");

      let mut sig = self.module.make_signature();
      sig.returns.push(AbiParam::new(types::I64));

      let func_id = self.module.declare_function(&name, Linkage::Export, &sig)?;

      let mut func = Function::new();
      func.signature = sig;

      {
        let mut builder = FunctionBuilder::new(&mut func, &mut self.builder_ctx);
        let entry = builder.create_block();
        builder.switch_to_block(entry);

        let result = {
          let mut ctx = EmitCtx {
            module: &mut self.module,
            func_ids: &self.func_ids,
            printf_id: self.printf_id,
            exit_id: self.exit_id,
            fmt_div_id: self.fmt_div_id,
            ptr_ty: self.ptr_ty,
          };
          ctx.emit_node(&mut builder, &eval.body, &[], &[], &[])?
        };

        builder.ins().return_(&[result]);
        builder.seal_all_blocks();
        builder.finalize();
      }

      let mut clif_ctx = ClifContext::for_function(func);
      self.module.define_function(func_id, &mut clif_ctx)?;
      self.eval_ids.push(func_id);
    }

    Ok(())
  }

  // finalize all definitions and execute each eval function, returning results
  pub fn run_evals(&mut self) -> Result<Vec<i64>, CodegenError> {
    self.module.finalize_definitions()?;

    let mut results = Vec::with_capacity(self.eval_ids.len());
    for &func_id in &self.eval_ids {
      use std::mem;
      let code_ptr = self.module.get_finalized_function(func_id);
      // safety: we just compiled this function with signature () -> i64
      let f: unsafe extern "C" fn() -> i64 = unsafe { mem::transmute(code_ptr) };
      results.push(unsafe { f() });
    }
    Ok(results)
  }
}

#[cfg(feature = "jit")]
impl Pass<ProgramIR> for Codegen<JITModule> {
  type Output = Codegen<JITModule>;
  type Error = CodegenError;

  fn run(mut self, ir: &ProgramIR) -> Result<Codegen<JITModule>, CodegenError> {
    ir.fold(&mut self)?;
    self.build_evals(&ir.evals)?;
    Ok(self)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::lexer::Lexer;
  use crate::lower::Lower;
  use crate::parser::Parser;
  use crate::pass::Acceptor;

  mod aot_backend {
    use super::*;

    fn compile(input: &str) -> Vec<u8> {
      let lexer = Lexer::new(input);
      let parser = Parser::new(lexer);
      let program = parser.parse().expect("parse failed");
      let ir = program.accept(Lower::new()).expect("lowering failed");
      let backend = ir
        .accept(AotBackend::new_aot("speed").expect("backend new failed"))
        .expect("compile failed");
      backend.into_bytes().expect("emit failed")
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

  #[cfg(feature = "jit")]
  mod jit_backend {
    use super::*;

    fn jit_eval(input: &str) -> Vec<i64> {
      let lexer = Lexer::new(input);
      let parser = Parser::new(lexer);
      let program = parser.parse().expect("parse failed");
      let ir = program.accept(Lower::new()).expect("lowering failed");
      let mut backend = ir
        .accept(JitBackend::new_jit("speed").expect("jit new failed"))
        .expect("jit compile failed");
      backend.run_evals().expect("jit run failed")
    }

    #[test]
    fn jit_const() {
      assert_eq!(jit_eval("def f = const(1, 0);\neval f(42);"), vec![0]);
    }

    #[test]
    fn jit_succ() {
      assert_eq!(jit_eval("def f = s;\neval f(5);"), vec![6]);
    }

    #[test]
    fn jit_add() {
      assert_eq!(
        jit_eval("def add = Pr[id(1,1), Cn[s, id(3,3)]];\neval add(3, 2);"),
        vec![5]
      );
    }

    #[test]
    fn jit_mult() {
      assert_eq!(
        jit_eval(
          "def z = const(1, 0);\n\
           def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
           def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
           eval mult(3, 4);"
        ),
        vec![12]
      );
    }

    #[test]
    fn jit_multiple_evals() {
      assert_eq!(
        jit_eval("def f = s;\neval f(0);\neval f(5);\neval f(99);"),
        vec![1, 6, 100]
      );
    }

    #[test]
    fn jit_search() {
      let results = jit_eval(
        "def z = const(1, 0);\n\
         def pred = Pr[const(0, 0), id(1,2)];\n\
         def add = Pr[id(1,1), Cn[s, id(3,3)]];\n\
         def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];\n\
         def monus = Pr[id(1,1), Cn[pred, id(3,3)]];\n\
         def isqrt = Mn[Cn[monus, id(1,2), Cn[mult, id(2,2), id(2,2)]]];\n\
         eval isqrt(9);",
      );
      assert_eq!(results, vec![3]);
    }
  }
}
