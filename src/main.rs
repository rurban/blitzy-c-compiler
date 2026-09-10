//! # BCC CLI Entry Point and Compilation Driver
//!
//! This is the binary crate entry point for BCC (Blitzy's C Compiler). It
//! orchestrates the entire compilation pipeline from command-line argument
//! parsing through final ELF output.
//!
//! ## Critical Architecture Decisions
//!
//! - **64 MiB Worker Thread**: All compilation work runs on a dedicated worker
//!   thread spawned with `std::thread::Builder::new().stack_size(64 * 1024 * 1024)`.
//!   The main thread only spawns this worker and waits for its result.
//! - **512-Recursion-Depth Limit**: Configured in [`CompilationContext`] and
//!   propagated to the parser and macro expander.
//! - **Zero-Dependency Mandate**: No external crates. All argument parsing is
//!   hand-rolled using `std::env::args()`.
//! - **GCC-Compatible CLI**: Flag syntax is compatible with `make CC=./bcc`
//!   for Linux kernel builds.
//!
//! ## Pipeline Stages
//!
//! ```text
//! Phase 1-2: Preprocessor  (trigraphs, line splicing, macro expansion)
//! Phase 3:   Lexer          (tokenization)
//! Phase 4:   Parser         (AST construction)
//! Phase 5:   Sema           (type checking, scope, builtins)
//! Phase 6:   IR Lowering    (AST → IR with allocas)
//! Phase 7:   mem2reg        (SSA construction)
//! Phase 8:   Optimization   (constant folding, DCE, CFG simplify)
//! Phase 9:   Phi Elimination(SSA → register-friendly form)
//! Phase 10:  Code Generation(architecture-specific)
//! Phase 11:  Assembly       (built-in assembler → .o)
//! Phase 12:  Linking        (built-in linker → ELF executable/shared object)
//! ```

// ============================================================================
// Standard library imports — the ONLY allowed external dependency
// ============================================================================

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::thread;

// ============================================================================
// BCC library crate imports (via `bcc::` prefix)
// ============================================================================

use bcc::common::diagnostics::DiagnosticEngine;
use bcc::common::source_map::SourceMap;
use bcc::common::string_interner::Interner;
use bcc::common::target::Target;
use bcc::common::temp_files::{
    create_temp_assembly_file, create_temp_object_file, TempDir, TempFile,
};
use bcc::common::type_builder::TypeBuilder;

use bcc::backend::generation::{generate_code, CodegenContext};
use bcc::frontend::lexer::Lexer;
use bcc::frontend::parser::Parser;
use bcc::frontend::preprocessor::Preprocessor;
use bcc::frontend::sema::SemanticAnalyzer;
use bcc::ir::lowering::lower_translation_unit;
use bcc::ir::mem2reg::{eliminate_phi_nodes, run_mem2reg};
use bcc::passes::run_optimization_pipeline;

// ============================================================================
// Constants
// ============================================================================

/// Worker thread stack size: 64 MiB. Required for deeply nested kernel macro
/// expansions and complex AST structures. Mandated by AAP §0.7.3.
const WORKER_STACK_SIZE: usize = 64 * 1024 * 1024;

/// Maximum recursion depth for the parser and macro expander.
/// Prevents stack overflow on deeply nested constructs. Mandated by AAP §0.7.3.
const MAX_RECURSION_DEPTH: usize = 512;

/// Program name used in diagnostic messages (GCC-compatible format).
const PROGRAM_NAME: &str = "bcc";

/// Version string for `--version` output.
const VERSION: &str = "0.1.0";

// ============================================================================
// CliArgs — parsed command-line arguments
// ============================================================================

/// Holds all parsed command-line arguments for BCC.
///
/// This struct captures every GCC-compatible flag that BCC supports. The
/// [`parse_args`] function constructs this from `std::env::args()`.
///
/// # Supported Flags
///
/// | Flag                | Field              | Description                          |
/// |---------------------|--------------------|--------------------------------------|
/// | `--target=<arch>`   | `target`           | Target architecture                  |
/// | `-o <file>`         | `output_file`      | Output file path                     |
/// | `-c`                | `compile_only`     | Produce .o only (no linking)         |
/// | `-S`                | `emit_assembly`    | Emit assembly text                   |
/// | `-E`                | `preprocess_only`  | Preprocess only                      |
/// | `-g`                | `debug_info`       | Emit DWARF v4 debug info             |
/// | `-O0`               | `optimization_level` | Optimization level (only 0 supported) |
/// | `-fPIC` / `-fpic`   | `pic`              | Position-independent code            |
/// | `-shared`           | `shared`           | Produce shared object (.so)          |
/// | `-mretpoline`       | `retpoline`        | Retpoline thunks (x86-64 only)       |
/// | `-fcf-protection`   | `cf_protection`    | CET/IBT (x86-64 only)               |
/// | `-I<dir>`           | `include_paths`    | Include search path                  |
/// | `-D<macro>[=val]`   | `defines`          | Preprocessor define                  |
/// | `-L<dir>`           | `library_paths`    | Library search path                  |
/// | `-l<lib>`           | `libraries`        | Link library                         |
#[derive(Debug, Clone)]
struct CliArgs {
    /// Input .c source file paths.
    input_files: Vec<String>,
    /// Output file path (`-o`). None means default naming.
    output_file: Option<String>,
    /// Target architecture. Defaults to X86_64.
    target: Target,
    /// `-c`: compile to .o without linking.
    compile_only: bool,
    /// `-S`: emit assembly text output.
    emit_assembly: bool,
    /// `-E`: preprocess only, output to stdout or -o file.
    preprocess_only: bool,
    /// `-g`: emit DWARF v4 debug information.
    debug_info: bool,
    /// Optimization level (only 0 is supported).
    optimization_level: u8,
    /// `-fPIC` / `-fpic`: generate position-independent code.
    pic: bool,
    /// `-shared`: produce a shared object (ET_DYN).
    shared: bool,
    /// `-mretpoline`: enable retpoline thunks (x86-64 only).
    retpoline: bool,
    /// `-fcf-protection`: enable CET/IBT (x86-64 only).
    cf_protection: bool,
    /// `-I<dir>`: include search paths (multiple allowed).
    include_paths: Vec<String>,
    /// `-D<macro>[=value]`: preprocessor defines (multiple allowed).
    defines: Vec<(String, Option<String>)>,
    /// `-U<macro>`: preprocessor undefines (multiple allowed).
    /// Applied after all `-D` defines during preprocessor initialization.
    undefs: Vec<String>,
    /// `-L<dir>`: library search paths (multiple allowed).
    library_paths: Vec<String>,
    /// `-l<lib>`: libraries to link (multiple allowed).
    libraries: Vec<String>,
    /// `-include <file>`: force-included headers (processed before main source).
    forced_includes: Vec<String>,
    /// `-MD` / `-MMD`: generate dependency file alongside compilation.
    gen_depfile: bool,
    /// `-MF <file>`: explicit dependency file output path.
    depfile_path: Option<String>,
    /// `-MT <target>`: dependency file target name.
    depfile_target: Option<String>,
    /// `-MP`: emit phony targets for each dependency.
    depfile_phony: bool,
    /// `-nostdinc`: suppress default system include paths.
    nostdinc: bool,
    /// `--verbose`: enable verbose compilation debugging output.
    verbose: bool,
    /// `-time`: enable `[BCC-TIMING]` phase-timing diagnostics on stderr.
    time: bool,
    /// `-march=<value>`: architecture string to pass through to the external
    /// assembler for `.S` files (e.g. `rv64imafdc`).
    march: Option<String>,
    /// `-mabi=<value>`: ABI string to pass through to the external assembler
    /// for `.S` files (e.g. `lp64d`).
    mabi: Option<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        CliArgs {
            input_files: Vec::new(),
            output_file: None,
            target: Target::X86_64,
            compile_only: false,
            emit_assembly: false,
            preprocess_only: false,
            debug_info: false,
            optimization_level: 0,
            pic: false,
            shared: false,
            retpoline: false,
            cf_protection: false,
            include_paths: Vec::new(),
            defines: Vec::new(),
            undefs: Vec::new(),
            library_paths: Vec::new(),
            libraries: Vec::new(),
            forced_includes: Vec::new(),
            gen_depfile: false,
            depfile_path: None,
            depfile_target: None,
            depfile_phony: false,
            nostdinc: false,
            verbose: false,
            time: false,
            march: None,
            mabi: None,
        }
    }
}

// ============================================================================
// CompilationContext — settings propagated through the pipeline
// ============================================================================

/// Compilation settings propagated through the entire pipeline.
///
/// Constructed from [`CliArgs`] by [`CompilationContext::from_cli_args`].
/// Every pipeline stage receives a reference to this context to access
/// target information, flags, and limits.
///
/// Some fields (library_paths, libraries, max_recursion_depth) are populated
/// but consumed later during the linker phase; suppress dead_code warnings.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CompilationContext {
    /// Target architecture.
    target: Target,
    /// Whether to emit DWARF v4 debug info (`-g`).
    debug_info: bool,
    /// Optimization level (0 = no optimization).
    optimization_level: u8,
    /// Whether to generate position-independent code (`-fPIC`).
    pic: bool,
    /// Whether to produce a shared library (`-shared`).
    shared: bool,
    /// Whether retpoline thunks are enabled (`-mretpoline`, x86-64 only).
    retpoline: bool,
    /// Whether CET/IBT is enabled (`-fcf-protection`, x86-64 only).
    cf_protection: bool,
    /// Include search paths from `-I` flags.
    include_paths: Vec<String>,
    /// Preprocessor defines from `-D` flags.
    defines: Vec<(String, Option<String>)>,
    /// Preprocessor undefines from `-U` flags.
    /// Applied after defines during preprocessor initialization.
    undefs: Vec<String>,
    /// Library search paths from `-L` flags.
    library_paths: Vec<String>,
    /// Libraries to link from `-l` flags.
    libraries: Vec<String>,
    /// Force-included headers from `-include` flags.
    forced_includes: Vec<String>,
    /// Maximum recursion depth for parser and macro expander.
    /// Fixed at 512 per AAP §0.7.3.
    max_recursion_depth: usize,
    /// Whether to suppress default system include paths (`-nostdinc`).
    nostdinc: bool,
    /// Whether verbose compilation debugging output is enabled (`--verbose`).
    verbose: bool,
}

impl CompilationContext {
    /// Construct a [`CompilationContext`] from parsed CLI arguments.
    ///
    /// Sets the recursion depth limit to 512 as mandated by AAP §0.7.3.
    fn from_cli_args(args: &CliArgs) -> Self {
        CompilationContext {
            target: args.target,
            debug_info: args.debug_info,
            optimization_level: args.optimization_level,
            pic: args.pic || args.shared, // -shared implies PIC
            shared: args.shared,
            retpoline: args.retpoline,
            cf_protection: args.cf_protection,
            include_paths: args.include_paths.clone(),
            defines: args.defines.clone(),
            undefs: args.undefs.clone(),
            library_paths: args.library_paths.clone(),
            libraries: args.libraries.clone(),
            forced_includes: args.forced_includes.clone(),
            max_recursion_depth: MAX_RECURSION_DEPTH,
            nostdinc: args.nostdinc,
            verbose: args.verbose,
        }
    }
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

/// Parse command-line arguments into a [`CliArgs`] struct.
///
/// Hand-rolled argument parser that supports GCC-compatible flag syntax.
/// Flags can be combined as `-Idir` (attached) or `-I dir` (separate).
///
/// # Errors
///
/// Returns `Err(message)` if:
/// - An unknown `--target=` value is provided
/// - `-o` is missing its argument
/// - Mutually exclusive flags (`-c`, `-S`, `-E`) are combined
/// - No input files are provided
fn parse_args(args: &[String]) -> Result<CliArgs, String> {
    let mut cli = CliArgs::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        // --target=<arch>
        if let Some(target_str) = arg.strip_prefix("--target=") {
            match Target::from_str(target_str) {
                Some(t) => cli.target = t,
                None => {
                    return Err(format!(
                        "unknown target '{}'. Supported targets: x86-64, i686, aarch64, riscv64",
                        target_str
                    ));
                }
            }
            i += 1;
            continue;
        }

        // --version
        if arg == "--version" {
            println!("{} version {}", PROGRAM_NAME, VERSION);
            process::exit(0);
        }

        // --help
        if arg == "--help" || arg == "-h" {
            print_usage();
            process::exit(0);
        }

        // -o <output>
        if arg == "-o" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-o'".to_string());
            }
            cli.output_file = Some(args[i].clone());
            i += 1;
            continue;
        }

        // -c (compile only)
        if arg == "-c" {
            cli.compile_only = true;
            i += 1;
            continue;
        }

        // -S (emit assembly)
        if arg == "-S" {
            cli.emit_assembly = true;
            i += 1;
            continue;
        }

        // -E (preprocess only)
        if arg == "-E" {
            cli.preprocess_only = true;
            i += 1;
            continue;
        }

        // -g (debug info)
        if arg == "-g" {
            cli.debug_info = true;
            i += 1;
            continue;
        }

        // -O<level>
        if let Some(level_str) = arg.strip_prefix("-O") {
            let level = match level_str {
                "0" | "" => 0,
                "1" => 1,
                "2" => 2,
                "3" => 3,
                "s" => 2, // -Os maps to -O2
                _ => {
                    // GCC silently treats unknown -O levels as -O2
                    eprintln!(
                        "{}: warning: unknown optimization level '{}', using -O0",
                        PROGRAM_NAME, level_str
                    );
                    0
                }
            };
            cli.optimization_level = level;
            i += 1;
            continue;
        }

        // -fPIC / -fpic
        if arg == "-fPIC" || arg == "-fpic" {
            cli.pic = true;
            i += 1;
            continue;
        }

        // -shared
        if arg == "-shared" {
            cli.shared = true;
            i += 1;
            continue;
        }

        // -mretpoline
        if arg == "-mretpoline" {
            cli.retpoline = true;
            i += 1;
            continue;
        }

        // -fcf-protection
        if arg == "-fcf-protection" || arg == "-fcf-protection=full" {
            cli.cf_protection = true;
            i += 1;
            continue;
        }
        if arg == "-fcf-protection=none" {
            cli.cf_protection = false;
            i += 1;
            continue;
        }

        // -I<dir> or -I <dir>
        if arg == "-I" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-I'".to_string());
            }
            cli.include_paths.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(dir) = arg.strip_prefix("-I") {
            cli.include_paths.push(dir.to_string());
            i += 1;
            continue;
        }

        // -D<macro>[=value] or -D <macro>[=value]
        if arg == "-D" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-D'".to_string());
            }
            let define_str = &args[i];
            cli.defines.push(parse_define(define_str));
            i += 1;
            continue;
        }
        if let Some(define_str) = arg.strip_prefix("-D") {
            cli.defines.push(parse_define(define_str));
            i += 1;
            continue;
        }

        // -U<macro> (undefine — store and propagate to preprocessor)
        if arg == "-U" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-U'".to_string());
            }
            cli.undefs.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(undef_str) = arg.strip_prefix("-U") {
            cli.undefs.push(undef_str.to_string());
            i += 1;
            continue;
        }

        // -L<dir> or -L <dir>
        if arg == "-L" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-L'".to_string());
            }
            cli.library_paths.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(dir) = arg.strip_prefix("-L") {
            cli.library_paths.push(dir.to_string());
            i += 1;
            continue;
        }

        // -l<lib> or -l <lib>
        if arg == "-l" {
            i += 1;
            if i >= args.len() {
                return Err("missing argument to '-l'".to_string());
            }
            cli.libraries.push(args[i].clone());
            i += 1;
            continue;
        }
        if let Some(lib) = arg.strip_prefix("-l") {
            cli.libraries.push(lib.to_string());
            i += 1;
            continue;
        }

        // -w (suppress all warnings — GCC compat)
        if arg == "-w" {
            i += 1;
            continue;
        }

        // -Wp,<flags> — pass flags to preprocessor (GCC compat).
        // Must be checked before generic -W handler below.
        if let Some(sub_flags) = arg.strip_prefix("-Wp,") {
            let parts: Vec<&str> = sub_flags.split(',').collect();
            let mut j = 0;
            while j < parts.len() {
                if parts[j] == "-MMD" || parts[j] == "-MD" {
                    cli.gen_depfile = true;
                    if j + 1 < parts.len() {
                        cli.depfile_path = Some(parts[j + 1].to_string());
                        j += 1;
                    }
                } else if parts[j] == "-MP" {
                    cli.depfile_phony = true;
                } else if parts[j] == "-MT" && j + 1 < parts.len() {
                    cli.depfile_target = Some(parts[j + 1].to_string());
                    j += 1;
                }
                j += 1;
            }
            i += 1;
            continue;
        }

        // -Wl,<flags> — pass flags to linker (GCC compat).
        // Must be checked before generic -W handler below.
        if arg.starts_with("-Wl,") {
            i += 1;
            continue;
        }

        // -Wa,<flags> — pass flags to assembler (GCC compat).
        // Must be checked before generic -W handler below.
        if arg.starts_with("-Wa,") {
            i += 1;
            continue;
        }

        // -Wall, -Wextra, -Werror, -W<anything> — accepted silently for GCC compat
        if arg.starts_with("-W") {
            i += 1;
            continue;
        }

        // -std=<standard> — accepted silently (we always target C11)
        if arg.starts_with("-std=") {
            i += 1;
            continue;
        }

        // -m32, -m64, -march=, -mtune=, -mabi= — accepted for GCC compat.
        // Additionally, -march= and -mabi= are used to infer the target
        // architecture when --target= is not explicitly provided.
        // We also STORE -march= and -mabi= so they can be forwarded to the
        // external assembler when compiling `.S` files.
        if arg.starts_with("-m") && arg != "-mretpoline" {
            // Infer target from -march=rv64... or -march=rv32...
            if let Some(march_val) = arg.strip_prefix("-march=") {
                // Store the value for pass-through to the external assembler.
                cli.march = Some(march_val.to_string());
                if march_val.starts_with("rv64") {
                    if cli.target == Target::X86_64 {
                        // Only override if target hasn't been explicitly set
                        cli.target = Target::RiscV64;
                    }
                } else if march_val.starts_with("rv32") {
                    // 32-bit RISC-V — not fully supported but detect it
                    if cli.target == Target::X86_64 {
                        cli.target = Target::RiscV64; // best effort
                    }
                } else if (march_val.starts_with("armv8") || march_val.starts_with("aarch64"))
                    && cli.target == Target::X86_64
                {
                    cli.target = Target::AArch64;
                }
            }
            // Infer from -mabi=lp64 (RISC-V LP64D ABI) or -mabi=ilp32
            if let Some(mabi_val) = arg.strip_prefix("-mabi=") {
                // Store the value for pass-through to the external assembler.
                cli.mabi = Some(mabi_val.to_string());
                if mabi_val.starts_with("lp64") && cli.target == Target::X86_64 {
                    cli.target = Target::RiscV64;
                } else if mabi_val.starts_with("ilp32") && cli.target == Target::X86_64 {
                    cli.target = Target::I686;
                }
            }
            i += 1;
            continue;
        }

        // -f<flag> — accept known flags, warn on unknown
        if arg.starts_with("-f")
            && arg != "-fPIC"
            && arg != "-fpic"
            && arg != "-fcf-protection"
            && arg != "-fcf-protection=full"
            && arg != "-fcf-protection=none"
        {
            // Silently accept common GCC flags for kernel build compatibility
            i += 1;
            continue;
        }

        // -pipe — accepted silently (GCC compat)
        if arg == "-pipe" {
            i += 1;
            continue;
        }

        // --verbose — enable verbose compilation debugging output
        if arg == "--verbose" {
            cli.verbose = true;
            i += 1;
            continue;
        }

        // -time — enable [BCC-TIMING] phase-timing diagnostics on stderr
        if arg == "-time" {
            cli.time = true;
            i += 1;
            continue;
        }

        // -nostdinc — suppress default system include paths
        if arg == "-nostdinc" {
            cli.nostdinc = true;
            i += 1;
            continue;
        }

        // -nostdlib, -nodefaultlibs, -nostartfiles — accepted silently
        if arg == "-nostdlib" || arg == "-nodefaultlibs" || arg == "-nostartfiles" {
            i += 1;
            continue;
        }

        // -static
        if arg == "-static" {
            i += 1;
            continue;
        }

        // -pthread
        if arg == "-pthread" {
            i += 1;
            continue;
        }

        // -MD / -MMD — generate dependency file
        if arg == "-MD" || arg == "-MMD" {
            cli.gen_depfile = true;
            i += 1;
            continue;
        }

        // -MF <file> — explicit dependency file path
        if arg == "-MF" {
            i += 1;
            if i < args.len() {
                cli.depfile_path = Some(args[i].clone());
                cli.gen_depfile = true;
                i += 1;
            }
            continue;
        }

        // -MT <target> — dependency target name
        if arg == "-MT" {
            i += 1;
            if i < args.len() {
                cli.depfile_target = Some(args[i].clone());
                i += 1;
            }
            continue;
        }

        // -MQ <target> — dependency target name (quoted)
        if arg == "-MQ" {
            i += 1;
            if i < args.len() {
                cli.depfile_target = Some(args[i].clone());
                i += 1;
            }
            continue;
        }

        // -MP — emit phony targets
        if arg == "-MP" {
            cli.depfile_phony = true;
            i += 1;
            continue;
        }

        // -M — generate dependency list only (preprocess mode)
        if arg == "-M" || arg == "-MM" {
            cli.gen_depfile = true;
            i += 1;
            continue;
        }

        // -x <language> — specify input language (GCC compat)
        // Accepted values: c, assembler-with-cpp, assembler, none
        // BCC treats everything as C, but accepts these for GCC compatibility.
        if arg == "-x" {
            i += 1;
            if i < args.len() {
                // Consume the language argument
                i += 1;
            }
            continue;
        }
        if let Some(_lang) = arg.strip_prefix("-x") {
            // -xc or -xassembler-with-cpp (attached form)
            i += 1;
            continue;
        }

        // -P — suppress line markers in preprocessor output (GCC compat)
        if arg == "-P" {
            i += 1;
            continue;
        }

        // -C — keep comments in preprocessor output (GCC compat, silently accepted).
        // BCC's preprocessor strips comments; this flag is accepted for GCC
        // compatibility (e.g., VDSO linker script preprocessing in the Linux
        // kernel build system).
        if arg == "-C" || arg == "-CC" {
            i += 1;
            continue;
        }

        // -Wa,<flags> — pass flags to assembler (GCC compat, silently accepted)
        if arg.starts_with("-Wa,") {
            i += 1;
            continue;
        }

        // -Wl,<flags> — pass flags to linker (GCC compat, silently accepted)
        if arg.starts_with("-Wl,") {
            i += 1;
            continue;
        }

        // -Wp,<flags> — pass flags to preprocessor (GCC compat)
        // Parse -Wp,-MMD,<path> for dependency file generation.
        // NOTE: -Wp, handling is now above the generic -W handler (line ~500).

        // --param=<name>=<value> or --param <name>=<value> (GCC compat, silently accepted)
        if arg == "--param" {
            i += 1;
            if i < args.len() {
                i += 1; // consume the param value
            }
            continue;
        }
        if arg.starts_with("--param=") {
            i += 1;
            continue;
        }

        // -include <file> — force-include a header before compilation (GCC compat)
        if arg == "-include" {
            i += 1;
            if i < args.len() {
                cli.forced_includes.push(args[i].clone());
                i += 1;
            }
            continue;
        }

        // -isystem <dir> — add system include path (GCC compat)
        if arg == "-isystem" {
            i += 1;
            if i < args.len() {
                cli.include_paths.push(args[i].clone());
                i += 1;
            }
            continue;
        }

        // -idirafter <dir> — add include path searched after -I (GCC compat)
        if arg == "-idirafter" {
            i += 1;
            if i < args.len() {
                cli.include_paths.push(args[i].clone());
                i += 1;
            }
            continue;
        }

        // Unknown flags starting with '-' — warn but continue (GCC compat)
        if arg.starts_with('-') && arg != "-" {
            eprintln!("{}: warning: unrecognized flag '{}'", PROGRAM_NAME, arg);
            i += 1;
            continue;
        }

        // "-" as input file means read from stdin
        if arg == "-" {
            cli.input_files.push("-".to_string());
            i += 1;
            continue;
        }

        // Anything else is an input file
        cli.input_files.push(arg.clone());
        i += 1;
    }

    // Validate: at least one input file required
    if cli.input_files.is_empty() {
        return Err("no input files".to_string());
    }

    // Validate: mutual exclusivity of -c, -S, -E
    let mode_count = cli.compile_only as u8 + cli.emit_assembly as u8 + cli.preprocess_only as u8;
    if mode_count > 1 {
        return Err("only one of '-c', '-S', '-E' may be specified".to_string());
    }

    Ok(cli)
}

/// Parse a `-D` define string into a `(name, optional_value)` pair.
///
/// Handles both `-DFOO` (define as empty) and `-DFOO=bar` (define with value).
fn parse_define(s: &str) -> (String, Option<String>) {
    if let Some(eq_pos) = s.find('=') {
        let name = s[..eq_pos].to_string();
        let value = s[eq_pos + 1..].to_string();
        (name, Some(value))
    } else {
        (s.to_string(), None)
    }
}

// ============================================================================
// Usage / Help
// ============================================================================

/// Print usage information to stderr (GCC-compatible format).
///
/// Shows all supported flags with brief descriptions. Called when `--help`
/// is passed or when argument parsing fails.
fn print_usage() {
    eprintln!("Usage: {} [options] <input.c> [-o output]", PROGRAM_NAME);
    eprintln!();
    eprintln!("BCC — Blitzy's C Compiler (C11 with GCC extensions)");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --target=<arch>     Set target architecture:");
    eprintln!("                        x86-64, i686, aarch64, riscv64");
    eprintln!("  -o <file>           Write output to <file>");
    eprintln!("  -c                  Compile and assemble, but do not link");
    eprintln!("  -S                  Compile only; output assembly text");
    eprintln!("  -E                  Preprocess only; output to stdout");
    eprintln!("  -g                  Emit DWARF v4 debug information");
    eprintln!("  -O0                 No optimization (default)");
    eprintln!("  -fPIC / -fpic       Generate position-independent code");
    eprintln!("  -shared             Produce a shared object (.so)");
    eprintln!("  -mretpoline         Enable retpoline thunks (x86-64 only)");
    eprintln!("  -fcf-protection     Enable CET/IBT (x86-64 only)");
    eprintln!("  -I<dir>             Add include search path");
    eprintln!("  -D<macro>[=value]   Define preprocessor macro");
    eprintln!("  -L<dir>             Add library search path");
    eprintln!("  -l<lib>             Link library");
    eprintln!("  -time               Print [BCC-TIMING] phase-timing diagnostics to stderr");
    eprintln!("  --version           Print version information and exit");
    eprintln!("  --help              Print this help message and exit");
}

// ============================================================================
// Output Path Resolution
// ============================================================================

/// Determine the output file path based on CLI args and compilation mode.
///
/// - If `-o <file>` is specified, use that path.
/// - If `-E`, output goes to stdout (returns `None`).
/// - If `-S`, change extension to `.s`.
/// - If `-c`, change extension to `.o`.
/// - Otherwise, default to `a.out`.
fn resolve_output_path(args: &CliArgs) -> Option<String> {
    if args.preprocess_only && args.output_file.is_none() {
        // -E without -o: output to stdout
        return None;
    }

    if let Some(ref out) = args.output_file {
        return Some(out.clone());
    }

    // Derive from first input file
    if let Some(ref input) = args.input_files.first() {
        let input_path = Path::new(input);
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        if args.emit_assembly {
            return Some(format!("{}.s", stem));
        }
        if args.compile_only {
            return Some(format!("{}.o", stem));
        }
    }

    // Default: a.out
    Some("a.out".to_string())
}

// ============================================================================
// Compilation Pipeline
// ============================================================================

/// Validate that all input files exist and are readable.
///
/// Uses `std::fs::read_to_string` to verify the file is accessible
/// before starting the compilation pipeline. This provides early,
/// clear error messages rather than cryptic failures deep in the
/// preprocessor.
///
/// # Returns
///
/// `Ok(())` if all files are readable, `Err(message)` for the first
/// unreadable file.
fn validate_input_files(input_files: &[String]) -> Result<(), String> {
    for input in input_files {
        // "-" means read from stdin — always valid, skip file checks
        if input == "-" {
            continue;
        }
        // "/dev/null" — special file, always valid for kernel build compat
        if input == "/dev/null" {
            continue;
        }
        let path = Path::new(input);
        if !path.exists() {
            return Err(format!("'{}': No such file or directory", input));
        }
        // Verify the file is actually readable by attempting to open it.
        // This catches permission errors early without reading the entire
        // contents (which could hang on special files like /dev/zero).
        // The preprocessor will re-read the file with its own PUA-aware
        // encoding, which enforces a maximum file size limit.
        match fs::File::open(path) {
            Ok(_) => {} // File is readable — discard the handle.
            Err(e) => {
                return Err(format!("'{}': {}", input, e));
            }
        }
    }
    Ok(())
}

/// Run the full compilation pipeline for all input files.
///
/// This is the core function that executes on the 64 MiB worker thread.
/// It processes each input file through the complete compilation pipeline
/// (or subset thereof, depending on `-E`, `-S`, `-c` flags) and produces
/// the final output.
///
/// # Pipeline Modes
///
/// - **`-E` (preprocess only)**: Preprocessor → output to stdout or file
/// - **`-S` (assembly output)**: Full pipeline through codegen → assembly text
/// - **`-c` (compile only)**: Full pipeline through assembler → .o file
/// - **Default (link)**: Full pipeline through linker → ELF executable or .so
///
/// # Multi-File Compilation
///
/// When multiple input files are provided without `-E` or `-S`:
/// 1. Each file is compiled to a temporary .o file
/// 2. All .o files are linked together to produce the final output
///
/// # Returns
///
/// `Ok(())` on success, `Err(message)` on fatal error.
fn run_compilation(args: CliArgs) -> Result<(), String> {
    // Validate all input files exist and are readable before starting the pipeline.
    validate_input_files(&args.input_files)?;

    // Warn when -shared is used without explicit -fPIC — the compiler will
    // implicitly enable PIC, but the user should be aware that creating a
    // shared library without -fPIC may produce incorrect code with text
    // relocations in some scenarios.
    if args.shared && !args.pic {
        eprintln!(
            "{}: warning: creating shared library without -fPIC; \
             PIC will be enabled implicitly, but explicit -fPIC is recommended",
            PROGRAM_NAME
        );
    }

    let ctx = CompilationContext::from_cli_args(&args);
    let output_path = resolve_output_path(&args);

    // Verbose mode: print compilation parameters for debugging
    if ctx.verbose {
        eprintln!(
            "{}: verbose: target={:?} opt={} pic={} shared={} debug={}",
            PROGRAM_NAME, ctx.target, ctx.optimization_level, ctx.pic, ctx.shared, ctx.debug_info
        );
        for input in &args.input_files {
            eprintln!("{}: verbose: input file: {}", PROGRAM_NAME, input);
        }
        if let Some(ref out) = output_path {
            eprintln!("{}: verbose: output file: {}", PROGRAM_NAME, out);
        }
        for path in &ctx.include_paths {
            eprintln!("{}: verbose: include path: {}", PROGRAM_NAME, path);
        }
        for (name, val) in &ctx.defines {
            match val {
                Some(v) => eprintln!("{}: verbose: define: {}={}", PROGRAM_NAME, name, v),
                None => eprintln!("{}: verbose: define: {}", PROGRAM_NAME, name),
            }
        }
    }

    // -E mode: preprocess only
    if args.preprocess_only {
        return run_preprocess_only(&args, &ctx, output_path.as_deref());
    }

    // Single-file fast path
    if args.input_files.len() == 1 {
        let input = &args.input_files[0];
        // If the single input is an object file (.o) and we're linking (not -c/-S),
        // pass it directly to the linker (e.g., `bcc -shared -o lib.so foo.o`).
        if is_object_or_archive(input) && !args.compile_only && !args.emit_assembly {
            let final_output = output_path.as_deref().unwrap_or("a.out");
            let obj_path = PathBuf::from(input);
            return link_object_files(&[obj_path], final_output, &ctx);
        }
        return compile_single_file(
            input,
            output_path.as_deref().unwrap_or("a.out"),
            &args,
            &ctx,
        );
    }

    // Multi-file: compile each to .o, then link
    if args.compile_only || args.emit_assembly {
        // -c or -S with multiple files: process each independently
        for input in &args.input_files {
            let out = if let Some(ref explicit) = args.output_file {
                // With -o and multiple files, only the last file gets -o
                // (GCC behavior with -c -o applies to each file individually)
                if args.input_files.len() == 1 {
                    explicit.clone()
                } else {
                    let stem = Path::new(input)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");
                    if args.emit_assembly {
                        format!("{}.s", stem)
                    } else {
                        format!("{}.o", stem)
                    }
                }
            } else {
                let stem = Path::new(input)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                if args.emit_assembly {
                    format!("{}.s", stem)
                } else {
                    format!("{}.o", stem)
                }
            };
            compile_single_file(input, &out, &args, &ctx)?;
        }
        return Ok(());
    }

    // Multi-file link mode: compile each .c to temp .o, pass .o files
    // through directly, then link all objects together.
    let _temp_dir =
        TempDir::new().map_err(|e| format!("failed to create temporary directory: {}", e))?;

    let mut object_files: Vec<PathBuf> = Vec::new();
    // Track which .o files BCC created (temps) vs user-provided (keep).
    let mut temp_object_files: Vec<PathBuf> = Vec::new();

    for input in &args.input_files {
        // If the input is already an object file or archive, pass it
        // directly to the linker without recompiling.
        if is_object_or_archive(input) {
            object_files.push(PathBuf::from(input));
            continue;
        }

        let temp_obj: TempFile = create_temp_object_file()
            .map_err(|e| format!("failed to create temporary object file: {}", e))?;
        let obj_path = temp_obj.path().to_path_buf();

        // Compile to temporary .o
        let mut compile_args = args.clone();
        compile_args.compile_only = true;
        compile_single_file(
            input,
            obj_path.to_str().unwrap_or("temp.o"),
            &compile_args,
            &ctx,
        )?;

        object_files.push(obj_path.clone());
        temp_object_files.push(obj_path);
        // Keep the temp file alive (don't drop it yet)
        temp_obj.keep();
    }

    // Link all object files together
    let final_output = output_path.as_deref().unwrap_or("a.out");
    link_object_files(&object_files, final_output, &ctx)?;

    // Clean up ONLY temporary .o files created by BCC — never delete
    // user-provided .o files or archives.
    for obj_path in &temp_object_files {
        let _ = fs::remove_file(obj_path);
    }

    Ok(())
}

/// Preprocess-only mode (`-E`): run preprocessor and output result.
fn run_preprocess_only(
    args: &CliArgs,
    ctx: &CompilationContext,
    output_path: Option<&str>,
) -> Result<(), String> {
    for input in &args.input_files {
        let mut source_map = SourceMap::new();
        let mut diagnostics = DiagnosticEngine::new();
        let mut interner = Interner::new();

        let mut pp =
            Preprocessor::new(&mut source_map, &mut diagnostics, ctx.target, &mut interner);

        // In `-E` mode, preserve `#pragma` directives in the output stream
        // so they appear in the preprocessed output, matching GCC behaviour.
        pp.preserve_pragmas = true;

        // Add include paths from CLI
        for path in &ctx.include_paths {
            pp.add_include_path(path);
        }

        // When -nostdinc is NOT active, add compiler-builtin and system paths.
        if !ctx.nostdinc {
            // Add compiler-builtin include path (for stdarg.h, stddef.h, etc.).
            // Locate the `include/` directory relative to the bcc binary.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(exe_dir) = exe.parent() {
                    let builtin = exe_dir.join("../../include");
                    if builtin.is_dir() {
                        if let Some(s) = builtin
                            .canonicalize()
                            .ok()
                            .and_then(|p| p.to_str().map(|s| s.to_string()))
                        {
                            pp.add_system_include_path(&s);
                        }
                    }
                    // Also try directly next to executable.
                    let builtin2 = exe_dir.join("include");
                    if builtin2.is_dir() {
                        if let Some(s) = builtin2
                            .canonicalize()
                            .ok()
                            .and_then(|p| p.to_str().map(|s| s.to_string()))
                        {
                            pp.add_system_include_path(&s);
                        }
                    }
                }
            }

            // Add default system include paths appropriate for the selected
            // target architecture.
            let target_sys_paths = ctx.target.system_include_paths();
            let extra_sys_paths: &[&str] = &["/usr/local/include", "/usr/include/linux"];
            for sp in target_sys_paths.iter().chain(extra_sys_paths.iter()) {
                if std::path::Path::new(sp).is_dir() {
                    pp.add_system_include_path(sp);
                }
            }
        }

        // Add defines from CLI
        for (name, value) in &ctx.defines {
            let val = value.as_deref().unwrap_or("1");
            pp.add_define(name, val);
        }

        // Apply undefs from CLI (after defines, matching GCC behaviour)
        for name in &ctx.undefs {
            pp.add_undef(name);
        }

        // Process forced-include files (-include <file>) before the main source.
        // Collect their tokens so they appear in the preprocessed output.
        let mut forced_tokens: Vec<bcc::frontend::preprocessor::PPToken> = Vec::new();
        for forced in &ctx.forced_includes {
            match pp.preprocess_file(forced) {
                Ok(mut ft) => {
                    forced_tokens.append(&mut ft);
                }
                Err(()) => {
                    diagnostics.print_all(&source_map);
                    return Err(format!("forced include '{}' failed", forced));
                }
            }
        }

        // Run preprocessor on main source
        let main_tokens = match pp.preprocess_file(input) {
            Ok(t) => t,
            Err(()) => {
                diagnostics.print_all(&source_map);
                return Err(format!("preprocessing failed for '{}'", input));
            }
        };

        // Combine forced-include tokens with main file tokens
        let tokens = if forced_tokens.is_empty() {
            main_tokens
        } else {
            forced_tokens.extend(main_tokens);
            forced_tokens
        };

        // Reconstruct preprocessed output from tokens, preserving newlines
        // so that `-E` output has correct line structure for downstream
        // tools (e.g. `wc -l`, diff, further compilation).
        //
        // CRITICAL: Suppress space insertion around `.` punctuators to keep
        // compound names like `.hash`, `.text`, `.gnu.hash`, `.rodata.*`,
        // `.data.*`, `.bss` intact — essential for linker script
        // preprocessing (`-E` on `.lds.S` files) and assembly directives.
        //
        // Strategy: track whether the last non-whitespace token was `.` and
        // whether the current token IS `.`. Suppress spaces in both
        // directions so that `identifier.identifier` and `.identifier`
        // patterns are preserved without internal spaces.
        let mut output = String::new();
        use bcc::frontend::preprocessor::PPTokenKind;
        let mut pending_space = false;
        for token in &tokens {
            match token.kind {
                PPTokenKind::Newline => {
                    output.push('\n');
                    pending_space = false;
                }
                PPTokenKind::Whitespace => {
                    // Mark that whitespace was seen; actual emission is
                    // deferred until the next non-whitespace token.
                    // When NO whitespace token appears between two tokens
                    // (e.g. `.` immediately followed by `hash`), pending_space
                    // stays false and no space is inserted — this reconstructs
                    // compound names like `.hash`, `.gnu.hash`, `.rodata.*`.
                    if !output.is_empty() && !output.ends_with('\n') {
                        pending_space = true;
                    }
                }
                _ if !token.text.is_empty() => {
                    if pending_space {
                        output.push(' ');
                    }
                    pending_space = false;
                    output.push_str(&token.text);
                }
                _ => {}
            }
        }

        // Write output
        if let Some(path) = output_path {
            fs::write(path, &output)
                .map_err(|e| format!("failed to write output to '{}': {}", path, e))?;
        } else {
            // Write to stdout
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            handle
                .write_all(output.as_bytes())
                .map_err(|e| format!("failed to write to stdout: {}", e))?;
        }

        // Print any accumulated diagnostics
        if diagnostics.has_errors() {
            diagnostics.print_all(&source_map);
            return Err(format!("preprocessing failed for '{}'", input));
        }

        // Generate dependency file when requested (kernel builds use -E -MD
        // together, and fixdep requires the .d file to exist even for
        // preprocess-only invocations).
        if args.gen_depfile {
            let dep_path = if let Some(ref dp) = args.depfile_path {
                dp.clone()
            } else {
                // Default: .{stem}.o.d in the same directory as input
                let inp = Path::new(input);
                let stem = inp.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
                let parent = inp.parent().unwrap_or(Path::new("."));
                parent
                    .join(format!(".{}.o.d", stem))
                    .to_string_lossy()
                    .to_string()
            };
            let dep_target = if let Some(ref tgt) = args.depfile_target {
                tgt.as_str()
            } else if let Some(op) = output_path {
                op
            } else {
                input
            };
            // Collect included files from the preprocessor for a more accurate
            // dependency list.  The preprocessor tracks all files it has opened
            // via the SourceMap.
            let mut deps = vec![input.to_string()];
            let mut fid: u32 = 0;
            loop {
                match source_map.get_filename(fid) {
                    Some(name) if name != input => {
                        deps.push(name.to_string());
                    }
                    None => break,
                    _ => {}
                }
                fid += 1;
            }
            let mut dep_content = format!("{}: {}\n", dep_target, deps.join(" \\\n  "));
            // Emit phony targets when -MP is active (prevents make errors when
            // a header is deleted).
            if args.depfile_phony {
                for d in &deps[1..] {
                    dep_content.push_str(&format!("\n{}:\n", d));
                }
            }
            let _ = fs::write(&dep_path, dep_content);
        }
    }

    Ok(())
}

/// Compile a single source file through the full pipeline.
///
/// Runs all applicable phases (preprocessing through code generation/linking)
/// based on the compilation flags.
/// Check if a path refers to a pre-compiled object file (`.o`) or static
/// archive (`.a`) that should be passed directly to the linker rather
/// than through the compilation pipeline.
fn is_object_or_archive(path: &str) -> bool {
    let p = Path::new(path);
    // Check by file extension first
    match p.extension().and_then(|e| e.to_str()) {
        Some("o") | Some("a") | Some("so") => return true,
        _ => {}
    }
    // Fall back to checking ELF magic bytes for files without recognized extensions
    if let Ok(mut f) = std::fs::File::open(p) {
        use std::io::Read;
        let mut magic = [0u8; 4];
        if f.read_exact(&mut magic).is_ok() {
            // ELF magic: 0x7f 'E' 'L' 'F'
            return magic == [0x7f, b'E', b'L', b'F'];
        }
    }
    false
}

/// Parse a Unix `ar` archive (`.a` file) and extract individual ELF
/// object members.  Returns a list of `(member_name, member_data)` pairs.
///
/// The archive format consists of:
///   - 8-byte global header: `!<arch>\n`
///   - Per-member headers (60 bytes each): name(16), date(12), uid(6),
///     gid(6), mode(8), size(10), fmag(2="`\n"`)
///   - Member data padded to 2-byte alignment
///
/// Special members like `//` (extended name table) and `/` (symbol table)
/// are handled or skipped appropriately.
fn parse_ar_archive(data: &[u8], archive_name: &str) -> Vec<(String, Vec<u8>)> {
    let mut members = Vec::new();
    let mut offset = 8; // skip "!<arch>\n"
    let mut extended_names: &[u8] = &[];

    while offset + 60 <= data.len() {
        let header = &data[offset..offset + 60];
        // Validate fmag bytes "`\n" at offset 58-59
        if header[58] != b'`' || header[59] != b'\n' {
            break; // corrupt header
        }

        // Parse name field (16 bytes)
        let name_raw = &header[0..16];
        // Parse size field (10 bytes, ASCII decimal)
        let size_str = std::str::from_utf8(&header[48..58]).unwrap_or("0").trim();
        let member_size: usize = size_str.parse().unwrap_or(0);
        let member_start = offset + 60;
        let member_end = (member_start + member_size).min(data.len());

        // Determine the member name.
        let name_trimmed = std::str::from_utf8(name_raw).unwrap_or("").trim_end();

        if name_trimmed == "/" || name_trimmed == "/SYM64/" {
            // Symbol table — skip.
        } else if name_trimmed == "//" {
            // Extended name table — store for lookups.
            extended_names = &data[member_start..member_end];
        } else {
            // Regular member or extended-name reference.
            let member_name = if let Some(stripped) = name_trimmed.strip_prefix('/') {
                // Extended name: "/NN" where NN is offset into extended names table.
                let ext_offset: usize = stripped.trim().parse().unwrap_or(0);
                if ext_offset < extended_names.len() {
                    // Name ends at '/' or '\n' within the extended names table.
                    let end = extended_names[ext_offset..]
                        .iter()
                        .position(|&b| b == b'/' || b == b'\n')
                        .map(|p| ext_offset + p)
                        .unwrap_or(extended_names.len());
                    std::str::from_utf8(&extended_names[ext_offset..end])
                        .unwrap_or("member.o")
                        .to_string()
                } else {
                    format!("{}:member_{}", archive_name, members.len())
                }
            } else {
                // Short name: strip trailing '/'.
                name_trimmed.trim_end_matches('/').to_string()
            };

            let member_data = data[member_start..member_end].to_vec();

            // Only include ELF members (skip any non-ELF data).
            if member_data.len() >= 4 && &member_data[..4] == b"\x7fELF" {
                members.push((member_name, member_data));
            }
        }

        // Advance to next member (2-byte aligned).
        offset = member_start + member_size;
        if offset % 2 != 0 {
            offset += 1;
        }
    }

    members
}

/// Compile an assembly source file (.S) by preprocessing with BCC's
/// preprocessor and then invoking the system cross-assembler appropriate
/// for the target architecture.
///
/// This is required because the Linux kernel build system passes `.S` files
/// through `$(CC) -c`, expecting the compiler to preprocess and assemble
/// them.
fn compile_assembly_file(
    input: &str,
    output: &str,
    args: &CliArgs,
    ctx: &CompilationContext,
) -> Result<(), String> {
    use std::process::Command;

    // Step 1: Preprocess the assembly file using BCC's preprocessor.
    let mut source_map = SourceMap::new();
    let mut diagnostics = DiagnosticEngine::new();
    let mut interner = Interner::new();

    let mut pp = Preprocessor::new(&mut source_map, &mut diagnostics, ctx.target, &mut interner);

    for path in &ctx.include_paths {
        pp.add_include_path(path);
    }
    if !ctx.nostdinc {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                let builtin = exe_dir.join("../../include");
                if builtin.is_dir() {
                    if let Some(s) = builtin
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
                let builtin2 = exe_dir.join("include");
                if builtin2.is_dir() {
                    if let Some(s) = builtin2
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
            }
        }
        let target_sys_paths = ctx.target.system_include_paths();
        let extra_sys_paths: &[&str] = &["/usr/local/include", "/usr/include/linux"];
        for sp in target_sys_paths.iter().chain(extra_sys_paths.iter()) {
            if std::path::Path::new(sp).is_dir() {
                pp.add_system_include_path(sp);
            }
        }
    }
    // GCC automatically defines __ASSEMBLER__ when preprocessing .S files.
    // We must do the same so that kernel headers (e.g. linux/elfnote.h) can
    // use `#ifdef __ASSEMBLER__` guards to select assembly-only code paths.
    pp.add_define("__ASSEMBLER__", "1");

    for (name, value) in &ctx.defines {
        let val = value.as_deref().unwrap_or("1");
        pp.add_define(name, val);
    }
    for name in &ctx.undefs {
        pp.add_undef(name);
    }
    // Process forced-include files (-include <file>) before the main source.
    // Collect their tokens so they appear in the preprocessed output (e.g. for
    // -E mode) and so that typedefs/macros from forced headers are visible to
    // the main source file during parsing.
    let mut forced_asm_tokens: Vec<bcc::frontend::preprocessor::PPToken> = Vec::new();
    for forced in &ctx.forced_includes {
        match pp.preprocess_file(forced) {
            Ok(mut ft) => {
                forced_asm_tokens.append(&mut ft);
            }
            Err(()) => {
                diagnostics.print_all(&source_map);
                return Err(format!("forced include '{}' failed", forced));
            }
        }
    }

    let main_asm_tokens = match pp.preprocess_file(input) {
        Ok(t) => t,
        Err(()) => {
            diagnostics.print_all(&source_map);
            return Err(format!("preprocessing failed for '{}'", input));
        }
    };

    let pp_tokens = if forced_asm_tokens.is_empty() {
        main_asm_tokens
    } else {
        forced_asm_tokens.extend(main_asm_tokens);
        forced_asm_tokens
    };

    // Reconstruct preprocessed output preserving whitespace structure.
    // For assembly files, it is critical that `.text` stays as `.text`
    // (no space between `.` and `text`) because the GNU assembler treats
    // them as a single directive.  We suppress space insertion after `.'
    // punctuators so that assembly directives like `.text`, `.globl`,
    // `.cfi_startproc`, `.L__local_label` are emitted correctly.
    let mut pp_output = String::new();
    use bcc::frontend::preprocessor::PPTokenKind;
    let mut need_space = false;
    let mut last_was_dot = false;
    for token in &pp_tokens {
        match token.kind {
            PPTokenKind::Newline => {
                pp_output.push('\n');
                need_space = false;
                last_was_dot = false;
            }
            PPTokenKind::Whitespace => {
                if !pp_output.is_empty() && !pp_output.ends_with('\n') {
                    need_space = true;
                }
            }
            _ if !token.text.is_empty() => {
                // Suppress space after `.` when followed by an identifier or
                // keyword — this forms assembly directives like `.text`,
                // `.globl`, `.cfi_startproc`, `.L__local_label`, `.type`.
                if need_space && !last_was_dot {
                    pp_output.push(' ');
                }
                pp_output.push_str(&token.text);
                last_was_dot = token.kind == PPTokenKind::Punctuator && token.text == ".";
                need_space = false;
            }
            _ => {
                last_was_dot = false;
            }
        }
    }

    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("preprocessing failed for '{}'", input));
    }

    // Step 2: Write preprocessed assembly to a temporary file.
    let temp_asm = create_temp_assembly_file()
        .map_err(|e| format!("failed to create temp assembly file: {}", e))?;
    fs::write(temp_asm.path(), &pp_output)
        .map_err(|e| format!("failed to write preprocessed assembly: {}", e))?;

    // Step 3: Determine the system cross-assembler for the target.
    let assembler = match ctx.target {
        Target::X86_64 => "as",
        Target::I686 => "i686-linux-gnu-as",
        Target::AArch64 => "aarch64-linux-gnu-as",
        Target::RiscV64 => "riscv64-linux-gnu-as",
    };

    // Step 4: Invoke the system assembler, forwarding architecture flags.
    let mut asm_cmd = Command::new(assembler);
    asm_cmd.arg("-o").arg(output);

    // Forward -march= and -mabi= to the external assembler so that
    // architecture-specific instructions (e.g. RISC-V compressed `c.li`)
    // are accepted. Without these flags the assembler rejects instructions
    // requiring extensions like 'c' or 'zca'.
    if let Some(ref march) = args.march {
        asm_cmd.arg(format!("-march={}", march));
    } else {
        // Provide sensible default -march when none was specified on the
        // command line, so that compressed-instruction assembly files
        // assemble correctly out of the box.
        match ctx.target {
            Target::RiscV64 => {
                asm_cmd.arg("-march=rv64imafdc");
            }
            Target::AArch64 => {
                asm_cmd.arg("-march=armv8-a");
            }
            _ => {}
        }
    }
    if let Some(ref mabi) = args.mabi {
        asm_cmd.arg(format!("-mabi={}", mabi));
    } else {
        // Default ABI when not specified.
        if ctx.target == Target::RiscV64 {
            asm_cmd.arg("-mabi=lp64d");
        }
    }

    asm_cmd.arg(temp_asm.path());

    let status = asm_cmd
        .status()
        .map_err(|e| format!("failed to invoke assembler '{}': {}", assembler, e))?;

    if !status.success() {
        return Err(format!(
            "assembler '{}' failed with exit code {:?}",
            assembler,
            status.code()
        ));
    }

    // Step 5: Generate dependency file if requested.
    if args.gen_depfile {
        let dep_path = if let Some(ref dp) = args.depfile_path {
            dp.clone()
        } else {
            let out_path = Path::new(output);
            let stem = out_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let parent = out_path.parent().unwrap_or(Path::new("."));
            parent
                .join(format!(".{}.o.d", stem))
                .to_string_lossy()
                .to_string()
        };
        let dep_target = args.depfile_target.as_deref().unwrap_or(output);
        let mut deps = vec![input.to_string()];
        let mut fid: u32 = 0;
        loop {
            match source_map.get_filename(fid) {
                Some(name) if name != input => {
                    deps.push(name.to_string());
                }
                None => break,
                _ => {}
            }
            fid += 1;
        }
        let mut dep_content = format!("{}: {}\n", dep_target, deps.join(" \\\n  "));
        if args.depfile_phony {
            for d in &deps[1..] {
                dep_content.push_str(&format!("\n{}:\n", d));
            }
        }
        let _ = fs::write(&dep_path, dep_content);
    }

    Ok(())
}

fn compile_single_file(
    input: &str,
    output: &str,
    args: &CliArgs,
    ctx: &CompilationContext,
) -> Result<(), String> {
    // Handle assembly source files (.S, .s) — preprocess with BCC, then
    // invoke the system cross-assembler for the target architecture.
    let input_lower = input.to_lowercase();
    if input_lower.ends_with(".s") {
        return compile_assembly_file(input, output, args, ctx);
    }

    let mut source_map = SourceMap::new();
    let mut diagnostics = DiagnosticEngine::new();
    let mut interner = Interner::new();
    let type_builder = TypeBuilder::new(ctx.target);

    // ========================================================================
    // Phase 1-2: Preprocessing
    // ========================================================================

    let mut pp = Preprocessor::new(&mut source_map, &mut diagnostics, ctx.target, &mut interner);

    // Configure preprocessor with CLI options
    for path in &ctx.include_paths {
        pp.add_include_path(path);
    }
    // When -nostdinc is NOT active, add compiler-builtin and system paths.
    if !ctx.nostdinc {
        // Compiler-builtin include path (for stdarg.h, stddef.h, etc.).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                let builtin = exe_dir.join("../../include");
                if builtin.is_dir() {
                    if let Some(s) = builtin
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
                let builtin2 = exe_dir.join("include");
                if builtin2.is_dir() {
                    if let Some(s) = builtin2
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                    {
                        pp.add_system_include_path(&s);
                    }
                }
            }
        }
        // Default system include paths — target-aware.
        let target_sys_paths = ctx.target.system_include_paths();
        let extra_sys_paths: &[&str] = &["/usr/local/include", "/usr/include/linux"];
        for sp in target_sys_paths.iter().chain(extra_sys_paths.iter()) {
            if std::path::Path::new(sp).is_dir() {
                pp.add_system_include_path(sp);
            }
        }
    }
    for (name, value) in &ctx.defines {
        let val = value.as_deref().unwrap_or("1");
        pp.add_define(name, val);
    }

    // Apply undefs from CLI (after defines, matching GCC behaviour)
    for name in &ctx.undefs {
        pp.add_undef(name);
    }

    // Process forced-include files (-include <file>) before the main source.
    let mut forced_asm_tokens: Vec<bcc::frontend::preprocessor::PPToken> = Vec::new();
    for forced in &ctx.forced_includes {
        match pp.preprocess_file(forced) {
            Ok(mut ft) => {
                forced_asm_tokens.append(&mut ft);
            }
            Err(()) => {
                diagnostics.print_all(&source_map);
                return Err(format!("forced include '{}' failed", forced));
            }
        }
    }

    let main_asm_tokens = match pp.preprocess_file(input) {
        Ok(t) => t,
        Err(()) => {
            diagnostics.print_all(&source_map);
            return Err(format!("preprocessing failed for '{}'", input));
        }
    };

    let pp_tokens = if forced_asm_tokens.is_empty() {
        main_asm_tokens
    } else {
        forced_asm_tokens.extend(main_asm_tokens);
        forced_asm_tokens
    };

    // Check for preprocessing errors
    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("preprocessing failed for '{}'", input));
    }

    // ========================================================================
    // Phase 3: Lexing
    // ========================================================================

    // Reconstruct source text from preprocessor tokens for the lexer.
    // The lexer expects a contiguous source string; we build one from the
    // preprocessed token stream, using the first file's content registered
    // in the source map if available, or reconstructing from tokens.
    let pp_text: String = pp_tokens
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Register the preprocessed text as a virtual file in the source map
    // so the lexer can produce spans that the diagnostic engine can resolve.
    let pp_file_id = source_map.add_file(format!("<preprocessed:{}>", input), pp_text.clone());

    let lexer = Lexer::new(&pp_text, pp_file_id, &mut interner, &mut diagnostics);

    // Check for lexer errors (emitted during tokenization via next_token calls
    // from the parser, so we don't tokenize_all here — the parser drives the lexer)

    // ========================================================================
    // Phase 4: Parsing
    // ========================================================================

    let t_parse_start = std::time::Instant::now();
    let mut parser = Parser::new(lexer, ctx.target);
    let mut translation_unit = parser.parse();
    let t_parse_elapsed = t_parse_start.elapsed();
    if bcc::common::timing::timing_enabled() {
        eprintln!("[BCC-TIMING] parse: {:.3}s", t_parse_elapsed.as_secs_f64());
    }

    // Check for parse errors
    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("parsing failed for '{}'", input));
    }

    // ========================================================================
    // Phase 5: Semantic Analysis
    // ========================================================================

    // The SemanticAnalyzer borrows `diagnostics` mutably, so we scope its
    // lifetime in a block to release the borrow before checking diagnostics.
    let t_sema_start = std::time::Instant::now();
    let sema_ok = {
        let mut sema =
            SemanticAnalyzer::new(&mut diagnostics, &type_builder, ctx.target, &interner);

        let analyze_result = sema.analyze(&mut translation_unit);
        let finalize_result = sema.finalize();

        analyze_result.is_ok() && finalize_result.is_ok()
    };
    let t_sema_elapsed = t_sema_start.elapsed();
    if bcc::common::timing::timing_enabled() {
        eprintln!("[BCC-TIMING] sema: {:.3}s", t_sema_elapsed.as_secs_f64());
    }
    // `sema` is now dropped — mutable borrow of `diagnostics` is released.

    if !sema_ok || diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("semantic analysis failed for '{}'", input));
    }

    // Create a fresh symbol table for lowering. The type-annotated AST from
    // semantic analysis carries resolved types in its nodes. The lowering
    // pass uses the symbol table primarily for linkage and storage class
    // information, which is also encoded in the AST attributes.
    let symbol_table = bcc::frontend::sema::SymbolTable::new();

    // ========================================================================
    // Phase 6: IR Lowering (AST → IR with allocas)
    // ========================================================================

    let t_ir_start = std::time::Instant::now();
    let mut ir_module = lower_translation_unit(
        &translation_unit,
        &symbol_table,
        &ctx.target,
        &type_builder,
        &source_map,
        &mut diagnostics,
        &interner,
    )
    .map_err(|e| {
        diagnostics.print_all(&source_map);
        format!("IR lowering failed for '{}': {}", input, e)
    })?;
    let t_ir_elapsed = t_ir_start.elapsed();
    if bcc::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] ir-lowering: {:.3}s",
            t_ir_elapsed.as_secs_f64()
        );
    }

    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("IR lowering failed for '{}'", input));
    }

    // Phase 7: SSA Construction (mem2reg — alloca promotion)
    // ========================================================================

    run_mem2reg(&mut ir_module);

    if std::env::var("BCC_DEBUG_CFG").is_ok() {
        for func in ir_module.functions() {
            if !func.is_definition {
                continue;
            }
            eprintln!(
                "[CFG-AFTER-MEM2REG] func={} blocks={}",
                func.name,
                func.block_count()
            );
            for (i, blk) in func.blocks().iter().enumerate() {
                eprintln!(
                    "  block {} succs={:?} preds={:?} insts={}",
                    i,
                    blk.successors(),
                    blk.predecessors(),
                    blk.instruction_count()
                );
                if std::env::var("BCC_DEBUG_IR").is_ok() {
                    for inst in blk.instructions() {
                        eprintln!("    {}", inst);
                    }
                }
            }
        }
    }

    // Phase 8: Optimization (constant folding, DCE, CFG simplification)
    // ========================================================================

    let _optimized = run_optimization_pipeline(&mut ir_module, ctx.optimization_level);

    if std::env::var("BCC_DEBUG_CFG").is_ok() {
        for func in ir_module.functions() {
            if !func.is_definition {
                continue;
            }
            eprintln!(
                "[CFG-AFTER-OPT] func={} blocks={}",
                func.name,
                func.block_count()
            );
            for (i, blk) in func.blocks().iter().enumerate() {
                eprintln!(
                    "  block {} succs={:?} preds={:?} insts={}",
                    i,
                    blk.successors(),
                    blk.predecessors(),
                    blk.instruction_count()
                );
            }
        }
    }

    // Phase 9: Phi Elimination (SSA → register-friendly form)
    // ========================================================================

    for func in ir_module.functions.iter_mut() {
        if func.is_definition {
            eliminate_phi_nodes(func);
        }
    }

    if std::env::var("BCC_DEBUG_CFG").is_ok() {
        for func in ir_module.functions() {
            if !func.is_definition {
                continue;
            }
            eprintln!(
                "[CFG-AFTER-PHI-ELIM] func={} blocks={}",
                func.name,
                func.block_count()
            );
            for (i, blk) in func.blocks().iter().enumerate() {
                eprintln!(
                    "  block {} succs={:?} preds={:?} insts={}",
                    i,
                    blk.successors(),
                    blk.predecessors(),
                    blk.instruction_count()
                );
            }
        }
    }

    // Phase 10-12: Code Generation, Assembly, Linking
    // ========================================================================

    let codegen_ctx = CodegenContext {
        target: ctx.target,
        debug_info: ctx.debug_info,
        optimization_level: ctx.optimization_level,
        pic: ctx.pic,
        shared: ctx.shared,
        retpoline: ctx.retpoline,
        cf_protection: ctx.cf_protection,
        output_path: output.to_string(),
        compile_only: args.compile_only,
        emit_assembly: args.emit_assembly,
        library_paths: ctx.library_paths.clone(),
        libraries: ctx.libraries.clone(),
    };

    let t_codegen_start = std::time::Instant::now();
    let output_bytes = generate_code(&ir_module, &codegen_ctx, &mut diagnostics, &source_map)
        .map_err(|e| {
            diagnostics.print_all(&source_map);
            format!("code generation failed for '{}': {}", input, e)
        })?;
    let t_codegen_elapsed = t_codegen_start.elapsed();
    if bcc::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] codegen: {:.3}s",
            t_codegen_elapsed.as_secs_f64()
        );
    }

    if diagnostics.has_errors() {
        diagnostics.print_all(&source_map);
        return Err(format!("code generation failed for '{}'", input));
    }

    // Write the output bytes to the destination file
    fs::write(output, &output_bytes)
        .map_err(|e| format!("failed to write output to '{}': {}", output, e))?;

    // For ELF executables, set the executable permission bit
    if !args.compile_only && !args.emit_assembly {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = fs::Permissions::from_mode(0o755);
            let _ = fs::set_permissions(output, perms);
        }
    }

    // Generate dependency file if -MD/-MMD/-MF was specified.
    if args.gen_depfile {
        let dep_path = if let Some(ref dp) = args.depfile_path {
            dp.clone()
        } else {
            // Default: replace .o extension with .d
            let out_path = Path::new(output);
            let stem = out_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let parent = out_path.parent().unwrap_or(Path::new("."));
            parent
                .join(format!(".{}.o.d", stem))
                .to_string_lossy()
                .to_string()
        };
        let dep_target = args.depfile_target.as_deref().unwrap_or(output);
        // Write a minimal dependency file listing the source file.
        // The kernel build system processes this with fixdep.
        let dep_content = format!("{}: {}\n", dep_target, input);
        let _ = fs::write(&dep_path, dep_content);
    }

    // Print any accumulated warnings (non-fatal diagnostics)
    if diagnostics.warning_count() > 0 {
        diagnostics.print_all(&source_map);
    }

    Ok(())
}

/// Link multiple object files into a final executable or shared library.
///
/// Invokes the built-in linker infrastructure (`bcc::backend::linker_common`)
/// to perform symbol resolution, section merging, relocation application,
/// and ELF output generation.
///
/// # Pipeline
///
/// 1. Read each `.o` file from disk.
/// 2. Parse each ELF object into a [`LinkerInput`] via the ELF parser in
///    `generation.rs`.
/// 3. Construct a [`LinkerConfig`] from the [`CompilationContext`].
/// 4. Create the architecture-specific relocation handler.
/// 5. Call `link()` from `linker_common` to produce a [`LinkerOutput`].
/// 6. Write the output bytes to the destination file with executable
///    permissions.
///
/// # Errors
///
/// Returns `Err(message)` if any object file cannot be read, the linker
/// reports unresolved symbols, or the output cannot be written.
/// Extract the DT_SONAME string from a minimal ELF parse.
/// Returns `None` if the ELF structure cannot be parsed or has no SONAME.
fn extract_elf_soname(data: &[u8]) -> Option<String> {
    if data.len() < 64 || &data[..4] != b"\x7fELF" {
        return None;
    }
    let is_64 = data[4] == 2;
    let le = data[5] == 1; // 1 = little-endian

    macro_rules! read_u16 {
        ($off:expr) => {
            if le {
                u16::from_le_bytes([data[$off], data[$off + 1]])
            } else {
                u16::from_be_bytes([data[$off], data[$off + 1]])
            }
        };
    }
    macro_rules! read_u32 {
        ($off:expr) => {
            if le {
                u32::from_le_bytes([data[$off], data[$off + 1], data[$off + 2], data[$off + 3]])
            } else {
                u32::from_be_bytes([data[$off], data[$off + 1], data[$off + 2], data[$off + 3]])
            }
        };
    }
    macro_rules! read_u64 {
        ($off:expr) => {
            if le {
                u64::from_le_bytes([
                    data[$off],
                    data[$off + 1],
                    data[$off + 2],
                    data[$off + 3],
                    data[$off + 4],
                    data[$off + 5],
                    data[$off + 6],
                    data[$off + 7],
                ])
            } else {
                u64::from_be_bytes([
                    data[$off],
                    data[$off + 1],
                    data[$off + 2],
                    data[$off + 3],
                    data[$off + 4],
                    data[$off + 5],
                    data[$off + 6],
                    data[$off + 7],
                ])
            }
        };
    }

    // Parse program headers to find PT_DYNAMIC.
    let (ph_off, ph_ent_size, ph_num) = if is_64 {
        (
            read_u64!(32) as usize,
            read_u16!(54) as usize,
            read_u16!(56) as usize,
        )
    } else {
        (
            read_u32!(28) as usize,
            read_u16!(42) as usize,
            read_u16!(44) as usize,
        )
    };

    let mut dyn_off: usize = 0;
    let mut dyn_size: usize = 0;
    let pt_dynamic: u32 = 2;
    for i in 0..ph_num {
        let base = ph_off + i * ph_ent_size;
        if base + ph_ent_size > data.len() {
            break;
        }
        let p_type = read_u32!(base);
        if p_type == pt_dynamic {
            if is_64 {
                dyn_off = read_u64!(base + 8) as usize;
                dyn_size = read_u64!(base + 32) as usize;
            } else {
                dyn_off = read_u32!(base + 4) as usize;
                dyn_size = read_u32!(base + 16) as usize;
            }
            break;
        }
    }
    if dyn_off == 0 || dyn_size == 0 {
        return None;
    }

    // Parse .dynamic entries to find DT_SONAME (14) and DT_STRTAB (5).
    let dt_soname_tag: u64 = 14;
    let dt_strtab_tag: u64 = 5;
    let entry_size = if is_64 { 16 } else { 8 };
    let mut soname_offset: Option<u64> = None;
    let mut strtab_vaddr: u64 = 0;

    let mut off = dyn_off;
    while off + entry_size <= data.len() && off < dyn_off + dyn_size {
        let (tag, val) = if is_64 {
            (read_u64!(off), read_u64!(off + 8))
        } else {
            (read_u32!(off) as u64, read_u32!(off + 4) as u64)
        };
        if tag == 0 {
            break; // DT_NULL
        }
        if tag == dt_soname_tag {
            soname_offset = Some(val);
        }
        if tag == dt_strtab_tag {
            strtab_vaddr = val;
        }
        off += entry_size;
    }

    let soname_off = soname_offset?;
    if strtab_vaddr == 0 {
        return None;
    }

    // Convert strtab virtual address to file offset by scanning section headers
    // or program headers for the LOAD segment containing strtab_vaddr.
    let mut strtab_file_off: Option<usize> = None;
    for i in 0..ph_num {
        let base = ph_off + i * ph_ent_size;
        if base + ph_ent_size > data.len() {
            break;
        }
        let p_type = read_u32!(base);
        if p_type != 1 {
            continue; // Not PT_LOAD
        }
        let (p_vaddr, p_offset, p_filesz) = if is_64 {
            (
                read_u64!(base + 16),
                read_u64!(base + 8),
                read_u64!(base + 32),
            )
        } else {
            (
                read_u32!(base + 8) as u64,
                read_u32!(base + 4) as u64,
                read_u32!(base + 16) as u64,
            )
        };
        if strtab_vaddr >= p_vaddr && strtab_vaddr < p_vaddr + p_filesz {
            strtab_file_off = Some((p_offset + (strtab_vaddr - p_vaddr)) as usize);
            break;
        }
    }

    let str_base = strtab_file_off?;
    let name_start = str_base + soname_off as usize;
    if name_start >= data.len() {
        return None;
    }
    // Read null-terminated string.
    let mut end = name_start;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    String::from_utf8(data[name_start..end].to_vec()).ok()
}

fn link_object_files(
    object_files: &[PathBuf],
    output: &str,
    ctx: &CompilationContext,
) -> Result<(), String> {
    use bcc::backend::generation::{
        create_relocation_handler, parse_elf_object_to_linker_input_uniquified,
    };
    use bcc::backend::linker_common::{link, LinkerConfig, OutputType};

    // Determine output type based on compilation flags.
    let output_type = if ctx.shared {
        OutputType::SharedLibrary
    } else {
        OutputType::Executable
    };

    let mut config = LinkerConfig::new(ctx.target, output_type);
    config.output_path = output.to_string();
    config.pic = ctx.pic;
    config.library_paths = ctx.library_paths.clone();
    config.libraries = ctx.libraries.clone();
    config.emit_debug = ctx.debug_info;

    // Resolve -l<lib> flags to DT_NEEDED entries.
    // For each library name, search -L paths + standard dirs for lib<name>.so.
    // If the found .so is a linker script (not a real ELF), resolve the actual
    // soname by trying versioned variants (lib<name>.so.6, .so.1, etc.) or by
    // reading the ELF DT_SONAME of the real shared object.
    let standard_lib_dirs: &[&str] = &[
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/lib",
        "/usr/lib",
    ];

    for lib_name in &ctx.libraries {
        let base_so = format!("lib{}.so", lib_name);
        let mut resolved_soname: Option<String> = None;

        // Build the search path list: user -L dirs + standard dirs.
        let search_dirs: Vec<&str> = ctx
            .library_paths
            .iter()
            .map(|s| s.as_str())
            .chain(standard_lib_dirs.iter().copied())
            .collect();

        for dir in &search_dirs {
            let candidate = std::path::Path::new(dir).join(&base_so);
            if !candidate.exists() {
                continue;
            }
            // Check if the file is a real ELF (starts with \x7fELF).
            if let Ok(header) = std::fs::read(&candidate) {
                if header.len() >= 4 && &header[..4] == b"\x7fELF" {
                    // Real ELF — try to extract DT_SONAME.
                    if let Some(soname) = extract_elf_soname(&header) {
                        resolved_soname = Some(soname);
                    } else {
                        // No DT_SONAME; use the filename.
                        resolved_soname = Some(base_so.clone());
                    }
                    break;
                }
                // Not an ELF — likely a GNU ld linker script.
                // Try versioned variants: lib<name>.so.6, .so.2, .so.1, etc.
                for ver in &["6", "2", "1", "0"] {
                    let versioned = format!("lib{}.so.{}", lib_name, ver);
                    let vpath = std::path::Path::new(dir).join(&versioned);
                    if vpath.exists() {
                        resolved_soname = Some(versioned);
                        break;
                    }
                }
                if resolved_soname.is_some() {
                    break;
                }
            }
        }

        // If no plain lib<name>.so was found, try versioned variants
        // (lib<name>.so.6, .so.2, .so.1, .so.0) in all search dirs.
        // This handles libraries like libatomic that only ship a
        // versioned .so.1 symlink without a plain .so in the runtime
        // library directories.
        if resolved_soname.is_none() {
            'versioned_search: for dir in &search_dirs {
                for ver in &["6", "2", "1", "0"] {
                    let versioned = format!("lib{}.so.{}", lib_name, ver);
                    let vpath = std::path::Path::new(dir).join(&versioned);
                    if vpath.exists() {
                        resolved_soname = Some(versioned);
                        break 'versioned_search;
                    }
                }
            }
        }

        // Also try GCC's cross-compiler library directories if nothing
        // was found in the standard paths. Libraries like libatomic
        // have their development symlinks under the GCC tree.
        if resolved_soname.is_none() {
            let gcc_lib_dirs: &[&str] = &[
                "/usr/lib/gcc/x86_64-linux-gnu/13",
                "/usr/lib/gcc/x86_64-linux-gnu/14",
                "/usr/lib/gcc/x86_64-linux-gnu/12",
            ];
            for dir in gcc_lib_dirs {
                let candidate = std::path::Path::new(dir).join(&base_so);
                if candidate.exists() {
                    if let Ok(header) = std::fs::read(&candidate) {
                        if header.len() >= 4 && &header[..4] == b"\x7fELF" {
                            if let Some(soname) = extract_elf_soname(&header) {
                                resolved_soname = Some(soname);
                            }
                        }
                    }
                    break;
                }
            }
        }

        // If no resolution found, fall back to lib<name>.so.
        let final_soname = resolved_soname.unwrap_or(base_so);

        // Avoid duplicating implicit libs (e.g., libm.so.6 already added).
        if !config.needed_libs.contains(&final_soname) {
            config.needed_libs.push(final_soname);
        }
    }

    // Always add libc.so.6 for executables unless the user explicitly
    // linked with -lc (which would already be in needed_libs).
    if output_type == OutputType::Executable
        && !config.needed_libs.iter().any(|n| n.starts_with("libc.so"))
    {
        config.needed_libs.push("libc.so.6".to_string());
    }

    // Parse each object file (or archive) into LinkerInput(s).
    let mut inputs = Vec::with_capacity(object_files.len());
    let mut input_idx: u32 = 0;
    for obj_path in object_files.iter() {
        let data = fs::read(obj_path)
            .map_err(|e| format!("failed to read object file '{}': {}", obj_path.display(), e))?;
        let filename = obj_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.o");

        // Check if this is a static archive (.a file) by looking for
        // the "!<arch>\n" magic header.
        if data.len() >= 8 && &data[..8] == b"!<arch>\n" {
            // Parse archive members and add each .o member as a
            // separate LinkerInput.
            let members = parse_ar_archive(&data, filename);
            for (member_name, member_data) in &members {
                let li = parse_elf_object_to_linker_input_uniquified(
                    input_idx,
                    member_name,
                    member_data,
                );
                inputs.push(li);
                input_idx += 1;
            }
        } else {
            let linker_input =
                parse_elf_object_to_linker_input_uniquified(input_idx, filename, &data);
            inputs.push(linker_input);
            input_idx += 1;
        }
    }

    // Create the architecture-specific relocation handler.
    let handler = create_relocation_handler(ctx.target);

    // Run the built-in linker.
    let mut diagnostics = DiagnosticEngine::new();
    let linker_output = link(&config, inputs, handler.as_ref(), &mut diagnostics)
        .map_err(|e| format!("linking failed: {}", e))?;

    if diagnostics.has_errors() {
        let source_map = SourceMap::new();
        diagnostics.print_all(&source_map);
        return Err("linking failed due to errors".to_string());
    }

    // Write the linked output to disk.
    fs::write(output, &linker_output.elf_data)
        .map_err(|e| format!("failed to write output to '{}': {}", output, e))?;

    // Set executable permission bit on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        let _ = fs::set_permissions(output, perms);
    }

    Ok(())
}

// ============================================================================
// Entry Point
// ============================================================================

/// BCC entry point.
///
/// 1. Parses command-line arguments
/// 2. On error, prints usage to stderr and exits with code 1
/// 3. Spawns a 64 MiB worker thread for all compilation work
/// 4. Joins the worker thread and propagates its exit code
///
/// Exit codes:
/// - 0: success
/// - 1: compilation error or internal error
fn main() {
    // Collect args (skip program name)
    let all_args: Vec<String> = env::args().collect();
    let args_slice = if all_args.len() > 1 {
        &all_args[1..]
    } else {
        // No arguments — print usage and exit
        print_usage();
        process::exit(1);
    };

    // Handle special flags before normal parsing.

    // `-Wa,--version`: The kernel build system uses this to detect the
    // assembler version. Output a GNU-assembler-compatible version string
    // to satisfy `scripts/as-version.sh`, then exit successfully.
    if args_slice.iter().any(|a| a == "-Wa,--version") {
        println!("GNU assembler (BCC built-in) 2.42");
        process::exit(0);
    }

    // `-print-file-name=<file>`: GCC prints the full path to a file in
    // its installation. BCC outputs an empty string for unsupported files,
    // which the kernel build system interprets as "not found."
    if let Some(arg) = args_slice
        .iter()
        .find(|a| a.starts_with("-print-file-name="))
    {
        let _name = arg.strip_prefix("-print-file-name=").unwrap_or("");
        // Return the queried name as-is (like "plugin"), indicating not found
        println!("{}", _name);
        process::exit(0);
    }

    // `-dumpversion`: Print just the version number (used by build systems).
    if args_slice.iter().any(|a| a == "-dumpversion") {
        println!("12.2.0");
        process::exit(0);
    }

    // `-dumpmachine`: Print the target triple (used by build systems).
    if args_slice.iter().any(|a| a == "-dumpmachine") {
        // Default to x86-64; check for --target= in other args
        let target = args_slice
            .iter()
            .find_map(|a| a.strip_prefix("--target="))
            .unwrap_or("x86-64");
        let triple = match target {
            "riscv64" => "riscv64-linux-gnu",
            "aarch64" => "aarch64-linux-gnu",
            "i686" => "i686-linux-gnu",
            _ => "x86_64-linux-gnu",
        };
        println!("{}", triple);
        process::exit(0);
    }

    // `--version` or `-v`: Print compiler version.
    if args_slice.iter().any(|a| a == "--version" || a == "-v") {
        println!("bcc (BCC) {}", VERSION);
        println!("Copyright (C) 2024 BCC Project");
        println!("This is free software; see the source for copying conditions.");
        process::exit(0);
    }

    // Parse command-line arguments
    let cli_args = match parse_args(args_slice) {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("{}: error: {}", PROGRAM_NAME, msg);
            print_usage();
            process::exit(1);
        }
    };

    // Propagate `-time` / `--verbose` to their process-wide flags. These
    // are plain `AtomicBool`s (not thread-local) because compilation runs
    // on the dedicated worker thread spawned below.
    bcc::common::timing::set_timing_enabled(cli_args.time);
    bcc::common::verbosity::set_verbose_enabled(cli_args.verbose);

    // Spawn the worker thread with 64 MiB stack for compilation work.
    // The main thread only spawns this worker and waits for its result.
    // This is mandated by AAP §0.7.3 to handle deeply nested kernel
    // macro expansions and complex AST structures without stack overflow.
    let builder = thread::Builder::new()
        .name("bcc-worker".to_string())
        .stack_size(WORKER_STACK_SIZE);

    let handle = builder
        .spawn(move || run_compilation(cli_args))
        .expect("failed to spawn worker thread");

    // Join the worker thread and propagate its exit code
    match handle.join() {
        Ok(Ok(())) => {
            // Successful compilation
            process::exit(0);
        }
        Ok(Err(msg)) => {
            // Compilation error — message already printed by the pipeline
            eprintln!("{}: error: {}", PROGRAM_NAME, msg);
            process::exit(1);
        }
        Err(_panic_payload) => {
            // Worker thread panicked — internal compiler error
            eprintln!(
                "{}: internal compiler error: worker thread panicked",
                PROGRAM_NAME
            );
            eprintln!(
                "This is a bug in BCC. Please report it with the source file \
                 that triggered the error."
            );
            process::exit(1);
        }
    }
}
