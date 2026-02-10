use std::fmt;
use std::fs;
use std::path::Path;
use std::process::Command;

use argh::FromArgs;

use lib_robin_core::backend::cranelift::AotBackend;
use lib_robin_core::lexer::Lexer;
use lib_robin_core::lower::Lower;
use lib_robin_core::parser::Parser;
use lib_robin_core::pass::Acceptor;

#[derive(Debug, Clone, PartialEq, Eq)]
enum EmitType {
  Obj,
  Exe,
  #[cfg(feature = "jit")]
  Jit,
}

impl fmt::Display for EmitType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      EmitType::Obj => write!(f, "obj"),
      EmitType::Exe => write!(f, "exe"),
      #[cfg(feature = "jit")]
      EmitType::Jit => write!(f, "jit"),
    }
  }
}

impl argh::FromArgValue for EmitType {
  fn from_arg_value(value: &str) -> Result<Self, String> {
    match value {
      "obj" => Ok(EmitType::Obj),
      "exe" => Ok(EmitType::Exe),
      #[cfg(feature = "jit")]
      "jit" => Ok(EmitType::Jit),
      other => {
        #[allow(unused_mut)]
        #[allow(clippy::useless_vec)]
        let mut valid = vec!["obj", "exe"];
        #[cfg(feature = "jit")]
        valid.push("jit");
        Err(format!(
          "unknown emit type '{other}', expected one of '{}'",
          valid.join(", ")
        ))
      }
    }
  }
}

/// robin - a compiler for the class of recursive functions
#[derive(FromArgs)]
struct Args {
  /// input source file (.rec)
  #[argh(positional)]
  input: String,

  /// output file path (default: <stem>.o or <stem>.exe)
  #[argh(option, short = 'o')]
  output: Option<String>,

  /// emit type: obj, exe, or jit (default: exe)
  #[argh(option, default = "EmitType::Exe")]
  emit: EmitType,

  /// optimization: 0=none, 1=speed, 2=speed_and_size (default: 1)
  #[argh(option, default = "1")]
  opt_level: u8,
}

fn main() -> Result<(), anyhow::Error> {
  let args: Args = argh::from_env();

  let opt_str = match args.opt_level {
    0 => "none",
    1 => "speed",
    2 => "speed_and_size",
    n => anyhow::bail!("invalid optimization level {n}, expected 0, 1, or 2"),
  };

  let source = fs::read_to_string(&args.input)?;

  // parse
  let lexer = Lexer::new(&source);
  let parser = Parser::new(lexer);
  let program = parser.parse().map_err(|errors| {
    let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    // list all errors, if any
    anyhow::anyhow!("parse errors:\n  {}", msgs.join("\n  "))
  })?;

  // lower ast into mid ir
  let ir = program.accept(Lower::new())?;

  // jit compile and run in process, print results from rust
  #[cfg(feature = "jit")]
  if args.emit == EmitType::Jit {
    use lib_robin_core::backend::cranelift::JitBackend;

    let mut backend = ir.accept(JitBackend::new_jit(opt_str)?)?;
    let results = backend.run_evals()?;
    for result in results {
      println!("{result}");
    }
    // return early to avoid other codepath
    return Ok(());
  }

  // aot compile ir to native object code
  let backend = ir.accept(AotBackend::new_aot(opt_str)?)?;

  // determine output paths
  let input_path = Path::new(&args.input);
  let stem = input_path
    .file_stem()
    .and_then(|s| s.to_str())
    .unwrap_or("output");

  let obj_path = match (&args.emit, &args.output) {
    (EmitType::Obj, Some(out)) => out.clone(),
    (EmitType::Obj, None) => format!("{stem}.o"),
    (EmitType::Exe, _) => format!("{stem}.tmp.o"),
    #[cfg(feature = "jit")]
    (EmitType::Jit, _) => unreachable!(),
  };

  // spit out the .o file
  backend.write_object_file(Path::new(&obj_path))?;

  if args.emit == EmitType::Obj {
    eprintln!("compiled: {obj_path}");
    return Ok(());
  }

  // link to executable
  let exe_path = match &args.output {
    Some(out) => out.clone(),
    None => {
      if cfg!(windows) {
        format!("{stem}.exe")
      } else {
        stem.to_string()
      }
    }
  };

  // this kinda sucks but it avoids adding a dependency on a rust linker crate and
  // works cross platform so long as clang is installed...
  // - todo: implement our own linker for this garbage

  let status = Command::new("clang")
    .args([&obj_path, "-o", &exe_path])
    .status()
    .map_err(|e| anyhow::anyhow!("failed to run linker (clang) - {e}"))?;

  if !status.success() {
    anyhow::bail!("linker failed with {status}");
  }

  // clean up intermediate object file
  let _ = fs::remove_file(&obj_path);

  eprintln!("compiled: {exe_path}");

  Ok(())
}
