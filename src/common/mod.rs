//! Common infrastructure for BCC.
//!
//! This module provides foundational types and utilities used by every layer
//! of the compiler: frontend, IR, optimization passes, and backend.
//! It has no dependencies on other BCC modules.

// ---------------------------------------------------------------------------
// Foundational utilities
// ---------------------------------------------------------------------------

/// FxHasher — fast, non-cryptographic hash function for symbol tables and
/// performance-critical lookup maps. Provides `FxHashMap` and `FxHashSet`
/// type aliases wrapping the standard-library collections.
pub mod fx_hash;

/// PUA/UTF-8 encoding — maps non-UTF-8 bytes (0x80–0xFF) to Unicode Private
/// Use Area code points (U+E080–U+E0FF) for byte-exact round-tripping through
/// the Rust `String`/`char` pipeline.
pub mod encoding;

/// Software long-double arithmetic — IEEE 754 extended-precision (80-bit)
/// add, subtract, multiply, divide, comparison, and conversion routines
/// implemented without any external math library.
pub mod long_double;

/// RAII-based temporary file and directory management for intermediate
/// compilation artifacts (`.o` files, assembler output) that are
/// automatically cleaned up when the owning handle is dropped.
pub mod temp_files;

/// Global `-time` diagnostics flag — controls whether `[BCC-TIMING]`
/// phase-timing traces are printed to stderr.
pub mod timing;

/// Global `--verbose` diagnostics flag — controls whether backend notes
/// (e.g. `[inline-asm]` warnings) are printed to stderr.
pub mod verbosity;

// ---------------------------------------------------------------------------
// Type system
// ---------------------------------------------------------------------------

/// Dual type system — C language types (`CType`) for the frontend and
/// target-machine ABI types (`MachineType`) for the backend, with
/// `sizeof`/`alignof` functions parameterised by target architecture.
pub mod types;

/// Builder-pattern API for constructing complex C and machine types,
/// including struct layout computation with `packed`/`aligned` attribute
/// support and flexible array member handling.
pub mod type_builder;

// ---------------------------------------------------------------------------
// Infrastructure services
// ---------------------------------------------------------------------------

/// Multi-error diagnostic reporting engine — collects errors, warnings, and
/// notes with source spans and renders them with formatted context for the
/// user.
pub mod diagnostics;

/// Source file tracking — assigns file IDs, computes line/column offsets for
/// O(log n) lookups, and handles `#line` directive remapping.
pub mod source_map;

// ---------------------------------------------------------------------------
// Higher-level utilities
// ---------------------------------------------------------------------------

/// String interning for identifiers, keywords, and string literals. Uses
/// `FxHashMap` for deduplication and returns zero-cost `Symbol` handles for
/// O(1) equality comparison.
pub mod string_interner;

/// Target architecture definitions and constants — `Target` enum
/// (X86_64, I686, AArch64, RiscV64), pointer widths, endianness, data
/// models (LP64 / ILP32), and per-architecture predefined macro sets.
pub mod target;
