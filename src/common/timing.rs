//! Global `-time` diagnostics flag.
//!
//! Controls whether internal `[BCC-TIMING]` phase-timing traces are printed
//! to stderr. Off by default; enabled for the lifetime of the process by
//! passing `-time` on the command line (see `print_usage` in `src/main.rs`).
//!
//! A process-wide [`AtomicBool`] is used — rather than threading a flag
//! through every semantic-analysis and IR-lowering call — because
//! compilation runs on a dedicated worker thread (see `main()` in
//! `src/main.rs`) and the timing call sites sit deep behind stable public
//! APIs (`SemanticAnalyzer::analyze`, `lower_translation_unit`, ...).

use std::sync::atomic::{AtomicBool, Ordering};

static TIMING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable `[BCC-TIMING]` diagnostic output for the process.
///
/// Set once from `main()` after parsing the `-time` CLI flag, before the
/// compilation worker thread is spawned.
pub fn set_timing_enabled(enabled: bool) {
    TIMING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns whether `[BCC-TIMING]` diagnostic output is currently enabled.
pub fn timing_enabled() -> bool {
    TIMING_ENABLED.load(Ordering::Relaxed)
}
