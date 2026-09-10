//! Global `--verbose` diagnostics flag.
//!
//! Controls whether internal backend diagnostics — e.g. `[inline-asm]`
//! warnings emitted while assembling unsupported inline-asm operand shapes
//! to a NOP placeholder — are printed to stderr. Off by default; enabled
//! for the lifetime of the process by passing `--verbose` on the command
//! line (see `print_usage` in `src/main.rs`).
//!
//! A process-wide [`AtomicBool`] is used — rather than threading a flag
//! through every backend call — because compilation runs on a dedicated
//! worker thread (see `main()` in `src/main.rs`) and the call sites sit
//! deep behind stable public APIs (the x86-64 inline-asm assembler).

use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable verbose backend diagnostic output for the process.
///
/// Set once from `main()` after parsing the `--verbose` CLI flag, before
/// the compilation worker thread is spawned.
pub fn set_verbose_enabled(enabled: bool) {
    VERBOSE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns whether verbose backend diagnostic output is currently enabled.
pub fn verbose_enabled() -> bool {
    VERBOSE_ENABLED.load(Ordering::Relaxed)
}
