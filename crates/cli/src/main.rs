use std::fmt;
use std::fs;
use std::path::Path;
use std::process::Command;

use argh::FromArgs;

use lib_robin_core::ast::Visitable;
use lib_robin_core::codegen::Codegen;
use lib_robin_core::lexer::Lexer;
use lib_robin_core::parser::Parser;

#[derive(Debug, Clone, PartialEq, Eq)]
enum EmitType {
  Obj,
  Exe,
}

impl fmt::Display for EmitType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      EmitType::Obj => write!(f, "obj"),
      EmitType::Exe => write!(f, "exe"),
    }
  }
}

impl argh::FromArgValue for EmitType {
  fn from_arg_value(value: &str) -> Result<Self, String> {
    match value {
      "obj" => Ok(EmitType::Obj),
      "exe" => Ok(EmitType::Exe),
      other => Err(format!(
        "unknown emit type '{other}', expected 'obj' or 'exe'"
      )),
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

  /// emit type: obj or exe (default: exe)
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

  // codegen
  let mut codegen = Codegen::new(opt_str).map_err(|e| anyhow::anyhow!("codegen init - {e}"))?;
  program.accept(&mut codegen)?;

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
  };

  codegen
    .write_object_file(Path::new(&obj_path))
    .map_err(|e| anyhow::anyhow!("{e}"))?;

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
