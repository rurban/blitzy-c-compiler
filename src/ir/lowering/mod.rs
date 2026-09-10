//! # IR Lowering — Phase 6
//!
//! This module implements Phase 6 of the BCC compilation pipeline: lowering the
//! semantically-validated, type-annotated AST into BCC's intermediate representation (IR).
//!
//! ## Architecture: Alloca-Then-Promote
//!
//! The lowering follows the mandated "alloca-first" pattern (matching LLVM's approach):
//!
//! 1. **Alloca insertion**: Every local variable is initially placed as an `alloca`
//!    instruction in the function's entry block, regardless of whether it could live
//!    in a register. Parameters are also alloca'd and their incoming values stored.
//!
//! 2. **Body lowering**: The function body is lowered using load/store instructions
//!    to access all local variables through their alloca pointers.
//!
//! 3. **SSA promotion** (Phase 7, NOT this module): The subsequent `mem2reg` pass
//!    promotes eligible allocas (scalar, non-address-taken) to SSA virtual registers.
//!
//! This design simplifies lowering enormously — every variable access is a simple
//! load or store to a known pointer, and SSA construction is deferred to a dedicated pass.
//!
//! ## Submodules
//!
//! - [`expr_lowering`]: Expression lowering (arithmetic, casts, calls, builtins, etc.)
//! - [`stmt_lowering`]: Statement lowering (control flow, loops, switch, goto, labels)
//! - [`decl_lowering`]: Declaration lowering (globals, function skeletons, static locals)
//! - [`asm_lowering`]: Inline assembly lowering (constraints, operands, clobbers, asm goto)
//!
//! ## Pipeline Integration
//!
//! - **Input**: Semantically-validated AST from `crate::frontend::sema`
//! - **Output**: `IrModule` containing `IrFunction`s and `GlobalVariable`s
//! - **Next stage**: `crate::ir::mem2reg` (Phase 7 — SSA construction)
//! - **Does NOT depend on**: `crate::ir::mem2reg`, `crate::passes`, or `crate::backend`
//!
//! ## Recursion Limit
//!
//! A hard 512-depth recursion limit is enforced for deeply nested structures
//! (kernel macros can produce deeply nested ASTs). This is mandated by the
//! project requirements (AAP §0.7.3).

// ============================================================================
// Submodule declarations
// ============================================================================

pub mod asm_lowering;
pub mod decl_lowering;
pub mod expr_lowering;
pub mod stmt_lowering;

// ============================================================================
// Imports — only `std` and `crate::` references (zero-dependency mandate)
// ============================================================================

use std::cell::RefCell;

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::source_map::SourceMap;
use crate::common::string_interner::Interner;
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{alignof_ctype, sizeof_ctype, CType};
use crate::frontend::parser::ast;
use crate::frontend::sema::symbol_table::SymbolTable;
use crate::ir::builder::IrBuilder;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BlockId, Value};
use crate::ir::module::IrModule;
use crate::ir::types::IrType;

// ============================================================================
// Thread-local interner snapshot for symbol name resolution
// ============================================================================

thread_local! {
    // Snapshot of the string interner content, populated once at the start of
    // `lower_translation_unit` so that all name extraction helpers can resolve
    // Symbol handles to real C identifier strings without threading a borrow of
    // the interner through deeply nested call chains.
    //
    // Wrapped in Rc so that callers can cheaply clone a reference-counted
    // handle instead of deep-copying the entire Vec<String> every time.
    static INTERNER_SNAPSHOT: RefCell<Option<std::rc::Rc<Vec<String>>>> = const { RefCell::new(None) };

    // Struct/union definitions registry, populated once at the start of
    // `lower_translation_unit` so that sizeof/alignof evaluation in global
    // constant expressions can resolve forward-referenced struct types.
    static SIZEOF_STRUCT_DEFS: RefCell<Option<FxHashMap<String, CType>>> = const { RefCell::new(None) };

    // Target architecture, populated at the start of `lower_translation_unit`
    // so that `evaluate_const_int_expr` can resolve sizeof/alignof in array
    // dimension expressions without threading the target through every call.
    static LOWERING_TARGET: RefCell<Option<crate::common::target::Target>> = const { RefCell::new(None) };

    // typeof resolution context — maps variable names to their resolved CTypes.
    // Populated incrementally during `collect_locals_from_declaration` so that
    // `typeof(var)` can look up a previously declared variable's type.  Also
    // pre-seeded with function parameter types at the start of
    // `lower_function_definition`.
    static TYPEOF_CONTEXT: RefCell<Option<FxHashMap<String, CType>>> = const { RefCell::new(None) };

    // Typedef resolution context — maps typedef names to their resolved CTypes.
    // Populated at the start of `lower_translation_unit` by scanning all
    // top-level typedef declarations and the builtin typedefs registered by
    // the semantic analyzer. This allows `resolve_base_type_fast` to correctly
    // resolve `TypedefName` to the actual underlying C type instead of
    // defaulting to `CType::Int`.
    static TYPEDEF_MAP: RefCell<Option<FxHashMap<String, CType>>> = const { RefCell::new(None) };

    /// Enum underlying type map.  Maps enum tag name → underlying CType
    /// (e.g., `"code"` → `CType::UInt` when all enumerator values ≥ 0).
    /// Populated during pass 0.5 (enum constant collection) and consumed by
    /// `map_single_type_specifier_with_names` in `decl_lowering.rs` so that
    /// struct field types for enum bitfields use the correct signedness.
    pub(crate) static ENUM_UNDERLYING_TYPES: RefCell<Option<FxHashMap<String, CType>>> = const { RefCell::new(None) };

    // Name table (interned-string vector) used by constant expression evaluators
    // that only receive an `ast::Symbol` and need to recover the string.
    static NAME_TABLE: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };

    // Anonymous globals generated from compound literals in global initializers.
    // Each entry is (name, ir_type, constant, c_type).  Populated during
    // `evaluate_constant_expr` when a CompoundLiteral is encountered in a
    // global initializer context (e.g., `struct A *p = &(struct A){1,2};`).
    // After Pass 1, these anonymous globals are added to the IR module.
    pub(super) static COMPOUND_LITERAL_GLOBALS: RefCell<Vec<(String, crate::ir::types::IrType, crate::ir::module::Constant, crate::common::types::CType)>> = const { RefCell::new(Vec::new()) };

    // Counter for generating unique anonymous compound literal global names.
    pub(super) static COMPOUND_LITERAL_COUNTER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    // Enum constants map — maps enum constant names to their integer values.
    // Populated BEFORE struct collection (Pass 0) so that array dimensions
    // using enum constants (e.g., `TString *tmname[TM_N]`) can be resolved
    // during struct field extraction.
    pub(super) static ENUM_CONSTANTS_TL: RefCell<Option<FxHashMap<String, i128>>> = const { RefCell::new(None) };
}

// ============================================================================
// Re-export for external convenience
// ============================================================================

pub use self::expr_lowering::ExprLoweringContext;

// ============================================================================
// Constants
// ============================================================================

/// Default maximum recursion depth for nested constructs during IR lowering.
/// This limit protects against stack overflow on deeply nested kernel macro
/// expansions. Mandated by AAP §0.7.3.
const DEFAULT_MAX_RECURSION_DEPTH: u32 = 512;

// ============================================================================
// LoweringError — errors that can occur during IR lowering
// ============================================================================

/// Errors that can occur during Phase 6 (AST-to-IR lowering).
///
/// These errors represent conditions that prevent the lowering pipeline
/// from producing correct IR for part or all of the translation unit.
/// Non-fatal issues are reported as diagnostics via [`DiagnosticEngine`]
/// instead.
#[derive(Debug)]
pub enum LoweringError {
    /// Recursion depth exceeded during nested construct lowering.
    ///
    /// The `limit` field records the maximum allowed depth (default: 512).
    /// This typically occurs with deeply nested macro expansions in the
    /// Linux kernel source tree.
    RecursionLimitExceeded {
        /// The recursion limit that was exceeded.
        limit: u32,
    },

    /// A C type could not be converted to an IR type.
    ///
    /// This can occur for unsupported type constructs or when the target
    /// architecture does not support a particular type configuration.
    TypeConversionError {
        /// Description of the source C type that failed conversion.
        from: Box<CType>,
        /// The IR type that was expected or partially produced.
        to: Box<IrType>,
        /// Source location of the type reference.
        span: Span,
    },

    /// An AST construct is not yet supported by the lowering pipeline.
    ///
    /// This error is emitted when the lowering encounters a language
    /// feature or GCC extension that has not been implemented.
    UnsupportedConstruct {
        /// Human-readable description of the unsupported construct.
        description: String,
        /// Source location of the unsupported construct.
        span: Span,
    },

    /// An internal compiler error (bug) occurred during lowering.
    ///
    /// This should never happen in correct operation — it indicates a
    /// logic error in the lowering implementation itself.
    InternalError {
        /// Human-readable description of the internal error.
        message: String,
    },
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoweringError::RecursionLimitExceeded { limit } => {
                write!(
                    f,
                    "recursion limit exceeded during IR lowering (limit: {})",
                    limit
                )
            }
            LoweringError::TypeConversionError { from, span, .. } => {
                write!(
                    f,
                    "type conversion error at {}:{}: {:?}",
                    span.file_id, span.start, from
                )
            }
            LoweringError::UnsupportedConstruct { description, span } => {
                write!(
                    f,
                    "unsupported construct at {}:{}: {}",
                    span.file_id, span.start, description
                )
            }
            LoweringError::InternalError { message } => {
                write!(f, "internal compiler error during IR lowering: {}", message)
            }
        }
    }
}

impl std::error::Error for LoweringError {}

// ============================================================================
// SwitchContext — coordination for switch statement lowering
// ============================================================================

/// Context for switch statement lowering — tracks case labels and default.
///
/// Created by [`stmt_lowering`] when a `switch` statement is entered and
/// consumed when the switch body has been fully lowered. Case and default
/// labels register their target blocks here so the switch dispatch
/// instruction can be built after the entire switch body is processed.
pub struct SwitchContext {
    /// The exit block (merge point after the switch body).
    pub exit_block: BlockId,
    /// Collected case labels: (constant value, target block).
    ///
    /// The `i128` value accommodates all C integer constant expression
    /// widths including `unsigned long long`.
    pub cases: Vec<(i128, BlockId)>,
    /// The default block, if a `default:` label was encountered.
    pub default_block: Option<BlockId>,
    /// The switch condition value (the discriminant being switched on).
    pub condition: Value,
}

// ============================================================================
// LoweringContext — central shared context for all lowering submodules
// ============================================================================

/// The central context for AST-to-IR lowering (Phase 6).
///
/// Holds all mutable and immutable state needed during the lowering of a
/// single translation unit. This includes the IR builder, variable mappings,
/// diagnostic engine, target architecture info, and recursion depth tracking.
///
/// The context is created by [`lower_translation_unit`] and threaded through
/// all four submodules ([`decl_lowering`], [`stmt_lowering`], [`expr_lowering`],
/// [`asm_lowering`]) during function body lowering.
///
/// # Alloca-First Pattern
///
/// The `local_vars` map stores alloca pointers (SSA values representing
/// stack addresses) for every local variable in the current function.
/// All variable accesses go through load/store from these pointers,
/// and the subsequent mem2reg pass (Phase 7) promotes eligible allocas
/// to SSA virtual registers.
pub struct LoweringContext<'a> {
    /// The IR builder for constructing instructions and basic blocks.
    ///
    /// Tracks the current insertion point and assigns sequential SSA
    /// value numbers to result-producing instructions.
    pub builder: IrBuilder,

    /// The IR module being constructed (for global variable and string
    /// literal pool access).
    pub module: &'a mut IrModule,

    /// Mapping from variable names to their alloca `Value` (pointer to
    /// the variable's stack storage).
    ///
    /// Populated during function prologue lowering (alloca-first pattern)
    /// and queried during expression lowering for variable references.
    pub local_vars: FxHashMap<String, Value>,

    /// Symbol table from semantic analysis (Phase 5).
    ///
    /// Provides type information, linkage classification, storage class,
    /// definition tracking, and attribute data for all declared symbols.
    /// Read-only during IR lowering.
    pub symbol_table: &'a SymbolTable,

    /// Target architecture information.
    ///
    /// Provides pointer width, type sizes, endianness, and other
    /// architecture-dependent constants needed for correct type lowering
    /// and ABI compliance.
    pub target: &'a Target,

    /// Type builder for constructing complex IR types from C types.
    ///
    /// Provides struct layout computation with packed/aligned attribute
    /// support and flexible array member handling.
    pub type_builder: &'a TypeBuilder,

    /// Source map for span-to-location resolution in diagnostics.
    ///
    /// Used to produce human-readable error messages with filename,
    /// line, and column information.
    pub source_map: &'a SourceMap,

    /// Diagnostic engine for error, warning, and note reporting.
    ///
    /// Non-fatal issues discovered during lowering are emitted here
    /// rather than causing immediate failure.
    pub diagnostics: &'a mut DiagnosticEngine,

    /// Current function being lowered (`None` when lowering globals).
    ///
    /// Set by [`decl_lowering::lower_function_definition`] at the start
    /// of function lowering and cleared when the function is complete.
    pub current_function: Option<IrFunction>,

    /// Current recursion depth for nested construct lowering.
    ///
    /// Incremented by [`enter_recursion`](LoweringContext::enter_recursion)
    /// and decremented by [`exit_recursion`](LoweringContext::exit_recursion).
    pub recursion_depth: u32,

    /// Maximum allowed recursion depth (default: 512).
    ///
    /// Exceeding this limit produces a [`LoweringError::RecursionLimitExceeded`]
    /// error and halts lowering for the current construct.
    pub max_recursion_depth: u32,

    /// Label targets for goto resolution: label name → target [`BlockId`].
    ///
    /// Populated when `label:` statements are encountered during function
    /// body lowering. Used to resolve both backward and forward gotos.
    pub label_targets: FxHashMap<String, BlockId>,

    /// Forward-referenced gotos that need patching: label name →
    /// `Vec<(source_block, instruction_index)>`.
    ///
    /// When a `goto label;` is encountered before the label is defined,
    /// the goto is recorded here. When the label is later encountered,
    /// [`resolve_pending_gotos`](LoweringContext::resolve_pending_gotos)
    /// patches the branch instructions.
    pub pending_gotos: FxHashMap<String, Vec<(BlockId, usize)>>,

    /// String literal deduplication pool: byte content → SSA `Value`.
    ///
    /// Caches the SSA value (global string reference) for each unique
    /// byte sequence so that identical string literals share a single
    /// `.rodata` entry and a single IR value.
    pub string_literal_pool: FxHashMap<Vec<u8>, Value>,

    /// Break target stack for `break` statements in loops and switches.
    ///
    /// Each loop or switch body pushes its exit block onto this stack;
    /// `break` statements branch to the top entry. Popped when the
    /// construct's body lowering is complete.
    pub break_targets: Vec<BlockId>,

    /// Continue target stack for `continue` statements in loops.
    ///
    /// Each loop pushes its increment/condition block onto this stack;
    /// `continue` statements branch to the top entry. Popped when the
    /// loop body lowering is complete.
    pub continue_targets: Vec<BlockId>,

    /// Current switch context for `case`/`default` label lowering.
    ///
    /// Set when entering a `switch` body and cleared when the switch
    /// dispatch instruction is built after the body is fully lowered.
    pub switch_context: Option<SwitchContext>,
}

// ============================================================================
// LoweringContext — construction and lifecycle
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Create a new lowering context for processing a translation unit.
    ///
    /// The context is initialized with:
    /// - An empty IR builder (no insertion point set)
    /// - Empty variable/label/string maps
    /// - No current function
    /// - Recursion depth 0, max depth 512
    /// - No break/continue targets or switch context
    ///
    /// # Parameters
    ///
    /// - `module`: The IR module being constructed.
    /// - `symbol_table`: The symbol table from semantic analysis.
    /// - `target`: Target architecture information.
    /// - `type_builder`: Type construction utilities.
    /// - `source_map`: Source file tracking for diagnostics.
    /// - `diagnostics`: Error/warning reporting engine.
    pub fn new(
        module: &'a mut IrModule,
        symbol_table: &'a SymbolTable,
        target: &'a Target,
        type_builder: &'a TypeBuilder,
        source_map: &'a SourceMap,
        diagnostics: &'a mut DiagnosticEngine,
    ) -> Self {
        LoweringContext {
            builder: IrBuilder::new(),
            module,
            local_vars: FxHashMap::default(),
            symbol_table,
            target,
            type_builder,
            source_map,
            diagnostics,
            current_function: None,
            recursion_depth: 0,
            max_recursion_depth: DEFAULT_MAX_RECURSION_DEPTH,
            label_targets: FxHashMap::default(),
            pending_gotos: FxHashMap::default(),
            string_literal_pool: FxHashMap::default(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            switch_context: None,
        }
    }
}

// ============================================================================
// LoweringContext — recursion depth management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Enter a new recursion level.
    ///
    /// Returns `Err(LoweringError::RecursionLimitExceeded)` if the depth
    /// limit (default: 512) would be exceeded. The caller should propagate
    /// this error to halt lowering of the current construct.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.enter_recursion()?;
    /// // ... lower nested construct ...
    /// ctx.exit_recursion();
    /// ```
    pub fn enter_recursion(&mut self) -> Result<(), LoweringError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            return Err(LoweringError::RecursionLimitExceeded {
                limit: self.max_recursion_depth,
            });
        }
        Ok(())
    }

    /// Exit a recursion level.
    ///
    /// Decrements the recursion depth counter. Uses `saturating_sub` to
    /// prevent underflow if called without a matching `enter_recursion`.
    pub fn exit_recursion(&mut self) {
        self.recursion_depth = self.recursion_depth.saturating_sub(1);
    }
}

// ============================================================================
// LoweringContext — variable management (alloca-first pattern)
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Look up a local variable's alloca pointer by name.
    ///
    /// Returns `Some(Value)` if the variable has been allocated (i.e., an
    /// alloca instruction exists for it in the entry block), or `None` if
    /// the name is not found in the current function's local variable map.
    ///
    /// The returned `Value` is a pointer to the variable's stack storage.
    /// Use `build_load` to read and `build_store` to write the variable.
    #[inline]
    pub fn get_variable(&self, name: &str) -> Option<Value> {
        self.local_vars.get(name).copied()
    }

    /// Register a new local variable alloca.
    ///
    /// Associates `name` with the alloca `Value` (pointer to stack storage)
    /// in the local variable map. Called during the alloca-first prologue
    /// for each local variable and function parameter.
    ///
    /// If a variable with the same name already exists (e.g., from a
    /// shadowed outer scope), the new alloca replaces it in the map.
    #[inline]
    pub fn set_variable(&mut self, name: String, alloca: Value) {
        self.local_vars.insert(name, alloca);
    }
}

// ============================================================================
// LoweringContext — type conversion
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Convert a C type to an IR type using this context's target info.
    ///
    /// Delegates to [`IrType::from_ctype`] which handles all C11 type
    /// variants including target-dependent sizes (e.g., `long` is I32 on
    /// i686 but I64 on x86-64), opaque pointers, union-as-byte-array
    /// representation, and qualifier/typedef stripping.
    #[inline]
    pub fn ctype_to_irtype(&self, ctype: &CType) -> IrType {
        IrType::from_ctype(ctype, self.target)
    }
}

// ============================================================================
// LoweringContext — string literal interning
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Intern a string literal, returning an SSA `Value` referencing it.
    ///
    /// Deduplicates identical byte sequences: if the same bytes have been
    /// interned before, the cached `Value` is returned without creating a
    /// new string pool entry. Otherwise:
    ///
    /// 1. The byte sequence is added to the module's string literal pool
    ///    via [`IrModule::intern_string`].
    /// 2. A fresh SSA `Value` is allocated via the builder to serve as an
    ///    IR-level reference to the global string constant.
    /// 3. The (bytes → Value) mapping is cached for future lookups.
    ///
    /// # Parameters
    ///
    /// - `bytes`: Raw byte content of the string literal, including any
    ///   null terminator. Bytes are stored with PUA-decoded byte-exact
    ///   fidelity.
    ///
    /// # Returns
    ///
    /// An SSA `Value` representing the address of the interned string
    /// constant in the `.rodata` section.
    pub fn intern_string_literal(&mut self, bytes: &[u8]) -> Value {
        // Check the deduplication cache first.
        if let Some(&cached_val) = self.string_literal_pool.get(bytes) {
            return cached_val;
        }

        // Intern in the module's string pool (returns a u32 string ID).
        let _string_id = self.module.intern_string(bytes.to_vec());

        // Allocate a fresh SSA value as the IR-level reference to this
        // global string constant.
        let val = self.builder.fresh_value();

        // Cache for deduplication.
        self.string_literal_pool.insert(bytes.to_vec(), val);

        val
    }
}

// ============================================================================
// LoweringContext — label / goto management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Register a label target block for goto resolution.
    ///
    /// Called by [`stmt_lowering`] when a labeled statement (`label:`) is
    /// encountered during function body lowering. The association between
    /// the label name and its target `BlockId` is stored in `label_targets`
    /// for use by subsequent goto statements.
    ///
    /// # Parameters
    ///
    /// - `name`: The label name (e.g., `"error_exit"`).
    /// - `target_block`: The `BlockId` of the basic block that the label
    ///   identifies.
    pub fn register_label(&mut self, name: &str, target_block: BlockId) {
        self.label_targets.insert(name.to_string(), target_block);
    }

    /// Resolve forward-referenced gotos to a label.
    ///
    /// Removes any pending forward-goto entries for the given label name
    /// from `pending_gotos`. The caller ([`stmt_lowering`]) is responsible
    /// for actually patching the branch instructions in the pending blocks
    /// to target the resolved label block.
    ///
    /// This method should be called immediately after [`register_label`]
    /// to handle any gotos that were emitted before the label was defined.
    ///
    /// # Parameters
    ///
    /// - `name`: The label name whose pending gotos should be resolved.
    ///
    /// # Returns
    ///
    /// The list of `(source_block, instruction_index)` pairs for forward
    /// gotos that referenced this label before it was defined. Returns
    /// an empty Vec if there were no pending gotos.
    pub fn resolve_pending_gotos(&mut self, name: &str) -> Vec<(BlockId, usize)> {
        self.pending_gotos.remove(name).unwrap_or_default()
    }
}

// ============================================================================
// LoweringContext — break / continue target management
// ============================================================================

impl<'a> LoweringContext<'a> {
    /// Push a break target (entering a loop or switch body).
    ///
    /// The `target` block is the merge point after the loop/switch body.
    /// `break` statements branch to the topmost break target.
    #[inline]
    pub fn push_break_target(&mut self, target: BlockId) {
        self.break_targets.push(target);
    }

    /// Pop a break target (exiting a loop or switch body).
    ///
    /// Returns the popped `BlockId`, or `None` if the stack was empty
    /// (which would indicate a compiler bug — `break` outside a loop/switch).
    #[inline]
    pub fn pop_break_target(&mut self) -> Option<BlockId> {
        self.break_targets.pop()
    }

    /// Push a continue target (entering a loop body).
    ///
    /// The `target` block is the loop's increment or condition block.
    /// `continue` statements branch to the topmost continue target.
    #[inline]
    pub fn push_continue_target(&mut self, target: BlockId) {
        self.continue_targets.push(target);
    }

    /// Pop a continue target (exiting a loop body).
    ///
    /// Returns the popped `BlockId`, or `None` if the stack was empty.
    #[inline]
    pub fn pop_continue_target(&mut self) -> Option<BlockId> {
        self.continue_targets.pop()
    }
}

// ============================================================================
// Standalone function: lower_translation_unit — PRIMARY ENTRY POINT
// ============================================================================

/// Lower an entire AST translation unit into an IR module.
///
/// This is the **primary entry point** for Phase 6, called by the
/// compilation pipeline driver. It processes all top-level declarations
/// in a two-pass strategy:
///
/// 1. **First pass — Global declarations**: Processes global variable
///    declarations/definitions, external function declarations, and
///    file-scope inline assembly. Typedefs and type-only declarations
///    are skipped (already handled by semantic analysis).
///
/// 2. **Second pass — Function definitions**: Lowers each function
///    definition using the alloca-first pattern. For each function:
///    - Create an `IrFunction` with an entry block
///    - Emit allocas for all parameters and local variables
///    - Lower the function body via [`stmt_lowering`]
///    - Add the completed function to the module
///
/// # Parameters
///
/// - `translation_unit`: The complete, semantically-validated AST.
/// - `symbol_table`: The symbol table from semantic analysis.
/// - `target`: Target architecture for type sizing and ABI.
/// - `type_builder`: Type construction utilities.
/// - `source_map`: Source file tracking for diagnostics.
/// - `diagnostics`: Error/warning reporting engine.
///
/// # Returns
///
/// An [`IrModule`] containing all lowered functions and global variables,
/// or a [`LoweringError`] if a fatal error occurred during lowering.
///
/// # Two-Pass Rationale
///
/// The two-pass strategy ensures that all global symbols (variables and
/// function declarations) are registered in the IR module before any
/// function body is lowered. This allows function bodies to reference
/// globals and call other functions by name without forward-declaration
/// ordering issues.
pub fn lower_translation_unit(
    translation_unit: &ast::TranslationUnit,
    symbol_table: &SymbolTable,
    target: &Target,
    type_builder: &TypeBuilder,
    source_map: &SourceMap,
    diagnostics: &mut DiagnosticEngine,
    interner: &Interner,
) -> Result<IrModule, LoweringError> {
    // Snapshot the interner content into thread-local storage so that all
    // name extraction helpers (extract_declarator_name, etc.) can resolve
    // Symbol handles to real C identifier strings without threading the
    // interner through every call.
    let snapshot: Vec<String> = (0..interner.len())
        .map(|idx| {
            let sym = crate::common::string_interner::Symbol::from_u32(idx as u32);
            interner.resolve(sym).to_string()
        })
        .collect();
    INTERNER_SNAPSHOT.with(|snap| {
        *snap.borrow_mut() = Some(std::rc::Rc::new(snapshot));
    });

    // The symbol table from semantic analysis (Phase 5) provides type
    // information, linkage classification, storage class, and attribute
    // data for all declared symbols.  It is threaded into the
    // LoweringContext for use during declaration and expression lowering.
    let _ = symbol_table; // Used below via LoweringContext and name_table pre-population.

    // Derive a module name from the source map (first file, or "<unknown>").
    let module_name = source_map
        .get_filename(0)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<unknown>".to_string());

    let mut module = IrModule::new(module_name);

    // Use the interner snapshot as the name table.  The snapshot maps
    // each Symbol's as_u32() index to its interned string, so
    // resolve_sym(name_table, sym) can resolve any identifier — not just
    // top-level declaration names.  This ensures function parameters,
    // local variables, and all other identifiers are resolvable during
    // expression and statement lowering.
    let name_table_rc: std::rc::Rc<Vec<String>> = INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc;

    // ====================================================================
    // Pass 0.5 — Collect typedef definitions for type resolution
    // ====================================================================
    // MUST run BEFORE struct/union collection (Pass 0), because struct
    // field extraction resolves typedef names from TYPEDEF_MAP.  Without
    // this, fields with typedef'd types (e.g., `u8` → `unsigned char`)
    // would fall back to `CType::Int`, corrupting struct layouts.
    //
    // Populate the thread-local target BEFORE Pass 0.5 so that
    // `evaluate_const_int_expr` can resolve sizeof/alignof in array
    // dimension expressions during typedef struct field collection.
    // Without this, `typedef struct { long r[(19+sizeof(long))/sizeof(long)]; } A;`
    // would get Array(long, None) because SizeofType handler finds no target.
    LOWERING_TARGET.with(|t| {
        *t.borrow_mut() = Some(*target);
    });

    // Populate NAME_TABLE early so evaluate_const_int_expr can resolve
    // Symbol → String for sizeof(identifier) lookups during typedef collection.
    NAME_TABLE.with(|nt| {
        *nt.borrow_mut() = Some(name_table.to_vec());
    });

    // ====================================================================
    // Pass 0-pre — Early enum constant collection for struct array sizes
    // ====================================================================
    // Enum constants must be available BEFORE both typedef collection
    // (Pass 0.5) and struct collection (Pass 0) because struct field
    // array dimensions may reference enum constants (e.g.,
    // `TString *tmname[TM_N]` in Lua's global_State).  When a typedef
    // wraps such a struct (e.g., `typedef struct global_State { ... }
    // global_State;`), the typedef pass calls extract_struct_union_fields
    // → evaluate_const_int_expr which must be able to resolve enum
    // constant identifiers to their integer values.
    //
    // This pass MUST run BEFORE Pass 0.5 (typedef collection).
    {
        let mut early_enum_map = FxHashMap::default();
        for ext_decl in &translation_unit.declarations {
            match ext_decl {
                ast::ExternalDeclaration::Declaration(decl) => {
                    collect_enum_constants_from_specifiers(
                        &decl.specifiers.type_specifiers,
                        name_table,
                        &mut early_enum_map,
                    );
                }
                ast::ExternalDeclaration::FunctionDefinition(func_def) => {
                    collect_enum_constants_from_specifiers(
                        &func_def.specifiers.type_specifiers,
                        name_table,
                        &mut early_enum_map,
                    );
                    collect_enum_constants_from_compound(
                        &func_def.body,
                        name_table,
                        &mut early_enum_map,
                    );
                }
                _ => {}
            }
        }
        ENUM_CONSTANTS_TL.with(|ec| {
            *ec.borrow_mut() = Some(early_enum_map);
        });
    }

    // Initialize the typedef map with builtin typedefs BEFORE scanning
    // user code, so that typedef chains (e.g., typedef __builtin_va_list
    // va_list) can resolve through the map incrementally.
    {
        let mut typedef_map: FxHashMap<String, CType> = FxHashMap::default();
        // Pre-seed with the compiler builtin typedef.
        typedef_map.insert(
            "__builtin_va_list".to_string(),
            CType::Pointer(
                Box::new(CType::Void),
                crate::common::types::TypeQualifiers::default(),
            ),
        );
        // Install the map into thread-local BEFORE the scan loop so that
        // resolve_base_type_fast can find __builtin_va_list when resolving
        // `typedef __builtin_va_list va_list;`.
        TYPEDEF_MAP.with(|map| {
            *map.borrow_mut() = Some(typedef_map);
        });
    }
    // Now scan all top-level typedef declarations and incrementally
    // add resolved types to the map.
    for ext_decl in &translation_unit.declarations {
        if let ast::ExternalDeclaration::Declaration(decl) = ext_decl {
            if matches!(
                decl.specifiers.storage_class,
                Some(ast::StorageClass::Typedef)
            ) {
                for init_decl in &decl.declarators {
                    let declarator = &init_decl.declarator;
                    if let Some(td_name) =
                        decl_lowering::extract_declarator_name(declarator, name_table)
                    {
                        let full_type = decl_lowering::resolve_declaration_type(
                            &decl.specifiers,
                            declarator,
                            target,
                            name_table,
                        );
                        // Chase Typedef wrappers to get the real underlying type.
                        let resolved = crate::common::types::resolve_typedef(&full_type).clone();
                        TYPEDEF_MAP.with(|map| {
                            if let Some(ref mut m) = *map.borrow_mut() {
                                m.insert(td_name, resolved);
                            }
                        });
                    }
                }
            }
        }
    }

    // LOWERING_TARGET and NAME_TABLE were already initialized before
    // Pass 0.5 (typedef collection) so that evaluate_const_int_expr can
    // resolve sizeof/alignof in array dimension expressions during both
    // typedef struct field collection and struct definition collection.

    // ====================================================================
    // Pass 0 — Collect struct/union/enum definitions before global lowering
    // ====================================================================
    let _t0 = std::time::Instant::now();
    // Struct/union definitions at file scope (or embedded in declarations)
    // populate a registry mapping tag name → CType.  This MUST run after
    // typedef collection (Pass 0.5) so that struct field types can resolve
    // typedef names, and MUST run before global variable lowering so that
    // constant expressions like `sizeof(struct S)` can resolve the full type.
    let mut struct_defs: FxHashMap<String, CType> = {
        let mut map = FxHashMap::default();
        for ext_decl in &translation_unit.declarations {
            match ext_decl {
                ast::ExternalDeclaration::Declaration(decl) => {
                    collect_struct_defs_from_specifiers(
                        &decl.specifiers.type_specifiers,
                        name_table,
                        &mut map,
                    );
                }
                ast::ExternalDeclaration::FunctionDefinition(func_def) => {
                    collect_struct_defs_from_specifiers(
                        &func_def.specifiers.type_specifiers,
                        name_table,
                        &mut map,
                    );
                    // Also collect struct/union definitions from within the
                    // function body. This handles function-local struct defs
                    // like `void f(void) { struct S { int x; }; ... }`.
                    collect_struct_defs_from_compound(&func_def.body, name_table, &mut map);
                }
                _ => {}
            }
            // Incrementally update SIZEOF_STRUCT_DEFS so that subsequent
            // struct definitions whose field array sizes depend on
            // sizeof/offsetof of previously collected structs can resolve
            // correctly during pass-0 collection.
            // (Example: `struct B { char a[sizeof(struct A) - offsetof(struct A, x)]; };`)
            SIZEOF_STRUCT_DEFS.with(|defs| {
                *defs.borrow_mut() = Some(map.clone());
            });
        }
        map
    };
    let _t0a = _t0.elapsed().as_secs_f64();

    // After collecting all struct definitions, resolve forward-referenced
    // typedef field types so that sizeof_ctype returns correct sizes.
    //
    // CRITICAL: A single pass is insufficient because the HashMap iteration
    // order is non-deterministic.  If struct S2 contains field of type S1,
    // and S1 contains field of type S0, and S2 is resolved *before* S1,
    // then S2 gets the UNRESOLVED S1 (with S0 still as a forward-ref).
    // Iterating until convergence ensures that nested forward references
    // are fully resolved regardless of iteration order.
    {
        let max_passes = 8; // depth limit to prevent infinite loops in pathological cases
        for _pass in 0..max_passes {
            let mut changed = false;
            let tags_to_resolve: Vec<String> = struct_defs.keys().cloned().collect();
            for tag in tags_to_resolve {
                if let Some(mut def) = struct_defs.remove(&tag) {
                    let had_empty = has_empty_struct_refs(&def);
                    resolve_struct_field_forward_refs(&mut def, &struct_defs);
                    let still_empty = has_empty_struct_refs(&def);
                    if had_empty && !still_empty {
                        changed = true;
                    }
                    struct_defs.insert(tag, def);
                }
            }
            if !changed {
                break;
            }
        }
    }
    let _t0b = _t0.elapsed().as_secs_f64();

    if crate::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] ir-pass0-struct-collect: {:.3}s (gather={:.3}s, resolve={:.3}s, {} tags)",
            _t0b,
            _t0a,
            _t0b - _t0a,
            struct_defs.len()
        );
    }
    // Populate the thread-local struct definitions registry so that
    // sizeof/alignof evaluation in global constant expressions can
    // resolve forward-referenced struct/union types.
    SIZEOF_STRUCT_DEFS.with(|defs| {
        *defs.borrow_mut() = Some(struct_defs.clone());
    });

    // Also populate the common::types tag resolution registry so that
    // sizeof_ctype / alignof_ctype (used inside compute_struct_size and
    // compute_union_size) can resolve lightweight tag references.
    // Without this, any struct containing a field whose type is a
    // lightweight tag reference (name + empty fields) would compute the
    // wrong struct layout.
    {
        let mut std_map = std::collections::HashMap::new();
        for (k, v) in &struct_defs {
            std_map.insert(k.clone(), v.clone());
        }
        crate::common::types::set_tag_type_defs(std_map);
    }

    // ====================================================================
    // Pass 0.5 — Collect enum constants from the AST (needed by globals)
    // ====================================================================
    let _t05 = std::time::Instant::now();
    // Enum constants are file-scope integer constants that must be
    // resolvable during global initializer evaluation and expression
    // lowering.  Collect them before Pass 1 so that global variable
    // initializers like `SAC_NoConsole` resolve to their integer values.
    let mut enum_underlying_map: FxHashMap<String, CType> = FxHashMap::default();
    let enum_constants: FxHashMap<String, i128> = {
        let mut map = FxHashMap::default();
        for ext_decl in &translation_unit.declarations {
            match ext_decl {
                ast::ExternalDeclaration::Declaration(decl) => {
                    collect_enum_constants_from_specifiers(
                        &decl.specifiers.type_specifiers,
                        name_table,
                        &mut map,
                    );
                    collect_enum_underlying_types(
                        &decl.specifiers.type_specifiers,
                        name_table,
                        &map,
                        &mut enum_underlying_map,
                    );
                }
                ast::ExternalDeclaration::FunctionDefinition(func_def) => {
                    collect_enum_constants_from_specifiers(
                        &func_def.specifiers.type_specifiers,
                        name_table,
                        &mut map,
                    );
                    collect_enum_underlying_types(
                        &func_def.specifiers.type_specifiers,
                        name_table,
                        &map,
                        &mut enum_underlying_map,
                    );
                    // Also collect block-scope enum constants from
                    // inside the function body (e.g., kernel patterns
                    // like `enum { MIX_INFLIGHT = 1U << 31 };`).
                    collect_enum_constants_from_compound(&func_def.body, name_table, &mut map);
                }
                _ => {}
            }
        }
        map
    };
    // Populate the thread-local enum underlying types map so that
    // struct field type resolution correctly uses UInt for enums
    // whose values are all non-negative (GCC-compatible bitfield
    // signedness).
    ENUM_UNDERLYING_TYPES.with(|m| {
        *m.borrow_mut() = Some(enum_underlying_map);
    });

    if crate::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] ir-pass0.5-enums: {:.3}s",
            _t05.elapsed().as_secs_f64()
        );
    }
    // ====================================================================
    // Pass 1 — Global declarations, function prototypes, file-scope asm
    // ====================================================================
    let _t1 = std::time::Instant::now();
    for ext_decl in &translation_unit.declarations {
        match ext_decl {
            ast::ExternalDeclaration::Declaration(decl) => {
                // Skip typedef declarations — type aliases are already
                // resolved by the semantic analysis phase.
                if matches!(
                    decl.specifiers.storage_class,
                    Some(ast::StorageClass::Typedef)
                ) {
                    continue;
                }

                // Skip declarations with no declarators (tag-only struct/enum
                // definitions, _Static_assert, etc.).
                if decl.declarators.is_empty() {
                    continue;
                }

                // Determine if this declaration contains function-type
                // declarators (function prototypes).
                let has_func_shape = decl
                    .declarators
                    .iter()
                    .any(|id| declarator_has_function_shape(&id.declarator));

                // Function declarations (prototypes without bodies) produce
                // FunctionDeclaration entries in the IR module.
                // Non-function declarations produce GlobalVariable entries.
                if has_func_shape && !declaration_has_initializer(decl) {
                    decl_lowering::lower_function_declaration(
                        decl,
                        &mut module,
                        target,
                        diagnostics,
                        name_table,
                        &struct_defs,
                    );
                } else {
                    decl_lowering::lower_global_variable(
                        decl,
                        &mut module,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        &enum_constants,
                    );
                }
            }

            ast::ExternalDeclaration::AsmStatement(asm_stmt) => {
                // File-scope inline assembly: emit verbatim into the module.
                let template_text = extract_asm_template(asm_stmt);
                if !template_text.is_empty() {
                    module.add_inline_asm(template_text);
                }
            }

            // Function definitions are handled in pass 2; Empty is skipped.
            ast::ExternalDeclaration::FunctionDefinition(_) | ast::ExternalDeclaration::Empty => {}
        }
    }

    if crate::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] ir-pass1-globals: {:.3}s",
            _t1.elapsed().as_secs_f64()
        );
    }
    // Seed TYPEOF_CONTEXT with global variable types so that
    // `typeof(global_var)` resolves correctly inside function bodies.
    TYPEOF_CONTEXT.with(|ctx| {
        let mut map = ctx.borrow_mut().take().unwrap_or_default();
        for (gname, gctype) in &module.global_c_types {
            map.entry(gname.clone()).or_insert_with(|| gctype.clone());
        }
        *ctx.borrow_mut() = Some(map);
    });

    // Drain anonymous compound literal globals created during Pass 1 constant evaluation.
    // Compound literals are anonymous and local to the translation unit — they must
    // use internal (static) linkage to avoid multiple-definition errors when linking
    // multiple object files that each contain compound literals with the same generated
    // name pattern (`__compound_literal.N`).
    COMPOUND_LITERAL_GLOBALS.with(|cl| {
        let globals = cl.borrow_mut().drain(..).collect::<Vec<_>>();
        for (anon_name, ir_type, initializer, c_type) in globals {
            let mut gv = crate::ir::module::GlobalVariable::new(
                anon_name.clone(),
                ir_type,
                Some(initializer),
            );
            gv.linkage = crate::ir::module::Linkage::Internal;
            module.add_global(gv);
            module.global_c_types.insert(anon_name, c_type);
        }
    });

    // ====================================================================
    // Pass 2 — Function definitions (alloca-first pattern)
    // ====================================================================
    let _t2 = std::time::Instant::now();
    let mut _fn_count = 0u32;
    let mut _fn_slow_count = 0u32;
    for ext_decl in &translation_unit.declarations {
        if let ast::ExternalDeclaration::FunctionDefinition(func_def) = ext_decl {
            let _tfn = std::time::Instant::now();
            // Lowering function definitions to IR follows the alloca-first
            // pattern: local variables are allocated in the entry block, then
            // the function body is lowered as statement/expression IR.
            decl_lowering::lower_function_definition(
                func_def,
                &mut module,
                target,
                type_builder,
                diagnostics,
                name_table,
                &enum_constants,
                &struct_defs,
            );
            _fn_count += 1;
            let _fn_elapsed = _tfn.elapsed().as_secs_f64();
            if _fn_elapsed > 0.5 {
                // Report slow function lowering
                let fname = {
                    fn extract_name_dd(
                        dd: &ast::DirectDeclarator,
                        nt: &[String],
                    ) -> Option<String> {
                        match dd {
                            ast::DirectDeclarator::Identifier(sym, _) => {
                                let idx = sym.as_u32() as usize;
                                if idx < nt.len() {
                                    Some(nt[idx].clone())
                                } else {
                                    None
                                }
                            }
                            ast::DirectDeclarator::Function { base, .. } => {
                                extract_name_dd(base, nt)
                            }
                            ast::DirectDeclarator::Array { base, .. } => extract_name_dd(base, nt),
                            ast::DirectDeclarator::Parenthesized(inner) => {
                                extract_name_dd(&inner.direct, nt)
                            }
                        }
                    }
                    extract_name_dd(&func_def.declarator.direct, name_table)
                        .unwrap_or_else(|| "<anon>".to_string())
                };
                if crate::common::timing::timing_enabled() {
                    eprintln!("[BCC-TIMING] slow fn: {} = {:.3}s", fname, _fn_elapsed);
                }
                _fn_slow_count += 1;
            }

            // Check if a fatal error was emitted during function lowering.
            // We continue lowering other functions to collect more diagnostics,
            // but the final result will reflect the error state.
        }
    }
    if crate::common::timing::timing_enabled() {
        eprintln!(
            "[BCC-TIMING] ir-pass2-functions: {} fns in {:.3}s ({} slow)",
            _fn_count,
            _t2.elapsed().as_secs_f64(),
            _fn_slow_count
        );
    }

    // Drain anonymous compound literal / string globals created during Pass 2
    // function lowering (e.g. static locals with string-pointer initializers
    // like `static const char *p = "hello" + 1;`).
    // Same as Pass 1: compound literals use internal linkage to prevent
    // cross-TU symbol collisions.
    COMPOUND_LITERAL_GLOBALS.with(|cl| {
        let globals = cl.borrow_mut().drain(..).collect::<Vec<_>>();
        for (anon_name, ir_type, initializer, c_type) in globals {
            let mut gv = crate::ir::module::GlobalVariable::new(
                anon_name.clone(),
                ir_type,
                Some(initializer),
            );
            gv.linkage = crate::ir::module::Linkage::Internal;
            module.add_global(gv);
            module.global_c_types.insert(anon_name, c_type);
        }
    });

    // If any fatal errors were emitted during lowering, report the failure
    // but still return the partially-constructed module (the diagnostic
    // engine retains all error details for the pipeline driver to display).
    if diagnostics.has_errors() {
        return Err(LoweringError::InternalError {
            message: format!(
                "IR lowering produced {} error(s)",
                diagnostics.error_count()
            ),
        });
    }

    Ok(module)
}

// ============================================================================
// Standalone function: ctype_to_irtype — C type to IR type conversion
// ============================================================================

/// Convert a C language type to an IR type, respecting target architecture
/// type sizes and alignment rules.
///
/// This is a convenience wrapper around [`IrType::from_ctype`] that
/// provides the canonical C-to-IR type mapping for the lowering pipeline.
///
/// # Conversion Highlights
///
/// - `long` → `I32` on i686 (ILP32), `I64` on x86-64/aarch64/riscv64 (LP64)
/// - `long double` → `F80` on x86 platforms (80-bit extended precision)
/// - `_Atomic(T)` → same representation as `T` (atomicity is in access, not storage)
/// - `_Complex T` → `[2 x T]` (array of two base-type elements)
/// - `union` → `[N x i8]` (byte array sized to the largest member)
/// - All pointers → `Ptr` (opaque pointer, LLVM-style)
/// - Typedefs → resolved to underlying type
/// - Qualifiers → stripped (no effect on IR data layout)
///
/// # Parameters
///
/// - `ctype`: The C type to convert.
/// - `target`: The target architecture for size resolution.
///
/// # Returns
///
/// The corresponding [`IrType`].
pub fn ctype_to_irtype(ctype: &CType, target: &Target) -> IrType {
    IrType::from_ctype(ctype, target)
}

// ============================================================================
// Standalone function: alignment_of — IR-level alignment for a C type
// ============================================================================

/// Get the IR-level alignment (in bytes) for a C type on the given target.
///
/// Wraps [`crate::common::types::alignof_ctype`] for convenient use during
/// IR lowering. The alignment is always a power of two and ≥ 1.
///
/// # Target-Dependent Behavior
///
/// - `long double`: 4-byte alignment on i686, 16-byte on 64-bit targets
/// - `long long`/`double`: 4-byte alignment on i686, 8-byte on 64-bit
/// - Packed structs: alignment 1 regardless of field types
/// - Explicit `__attribute__((aligned(N)))` overrides natural alignment
///
/// # Parameters
///
/// - `ctype`: The C type whose alignment to query.
/// - `target`: The target architecture.
///
/// # Returns
///
/// The required alignment in bytes.
#[inline]
pub fn alignment_of(ctype: &CType, target: &Target) -> usize {
    alignof_ctype(ctype, target)
}

// ============================================================================
// Standalone function: size_of — IR-level size for a C type
// ============================================================================

/// Get the IR-level size (in bytes) for a C type on the given target.
///
/// Wraps [`crate::common::types::sizeof_ctype`] for convenient use during
/// IR lowering.
///
/// # Key Sizes
///
/// - `void`: 1 (GCC extension for pointer arithmetic)
/// - `_Bool`: 1
/// - `long`: 4 on i686, 8 on LP64 targets
/// - `long double`: 12 on i686, 16 on other targets
/// - `_Complex T`: 2 × sizeof(T)
/// - Struct: includes inter-field padding and tail padding
/// - Union: max of all field sizes, padded to alignment
///
/// # Parameters
///
/// - `ctype`: The C type whose size to query.
/// - `target`: The target architecture.
///
/// # Returns
///
/// The size in bytes.
#[inline]
pub fn size_of(ctype: &CType, target: &Target) -> usize {
    sizeof_ctype(ctype, target)
}

// ============================================================================
// Private helpers — declaration classification
// ============================================================================

/// Check if a declarator has a function shape (is a function prototype).
///
/// A declarator has function shape if its direct declarator is the
/// `Function` variant, indicating a parameter list rather than a simple
/// identifier or array shape.
fn declarator_has_function_shape(decl: &ast::Declarator) -> bool {
    // Distinguish between function declarations and function pointer variables:
    //   void foo(int);          → Function { base: Identifier("foo") }  → true
    //   void (*fp)(int);        → Function { base: Parenthesized(Declarator{ pointer: Some(..) }) } → false
    //   void (**fp)(int);       → same pattern, still a variable → false
    //   void (*fp[5])(int);     → array of fn ptrs, still a variable → false
    //   void (foo)(int);        → Function { base: Parenthesized(Declarator{ pointer: None }) } → true
    match &decl.direct {
        ast::DirectDeclarator::Function { base, .. } => {
            // If the base of the function declarator is a parenthesized
            // declarator that contains a pointer, this is a function-pointer
            // variable (e.g., `void (*fp)(int)`) — NOT a function declaration.
            if let ast::DirectDeclarator::Parenthesized(inner_decl) = base.as_ref() {
                if inner_decl.pointer.is_some() {
                    return false;
                }
            }
            true
        }
        ast::DirectDeclarator::Parenthesized(inner) => declarator_has_function_shape(inner),
        _ => false,
    }
}

/// Check if a declaration has any initializer (i.e., is a definition
/// rather than a pure declaration).
fn declaration_has_initializer(decl: &ast::Declaration) -> bool {
    decl.declarators.iter().any(|id| id.initializer.is_some())
}

/// Extract the template string from a file-scope `AsmStatement`.
///
/// For file-scope assembly, the template is the raw assembly text to
/// emit verbatim before any generated code. The template is stored
/// as `Vec<u8>` (raw bytes with PUA-decoded non-UTF-8 content) in the
/// AST `AsmStatement` node.
///
/// Returns the template as a `String` using lossy UTF-8 conversion,
/// which is appropriate for assembly directives that are typically
/// ASCII-only.
fn extract_asm_template(asm_stmt: &ast::AsmStatement) -> String {
    // AsmStatement.template is Vec<u8> (raw bytes, PUA-decoded).
    // For file-scope asm, the template contains the literal assembly
    // text without any operand substitutions.
    String::from_utf8_lossy(&asm_stmt.template).to_string()
}

// Dead functions removed: extract_declarator_name and extract_direct_declarator_name
// were superseded by the versions in decl_lowering.rs that accept a name_table parameter.

/// Scan type specifiers for `enum { ... }` definitions and collect
/// all enumerator name → integer value mappings into `out`.
///
/// This handles both explicit values (`RED = 5`) and implicit auto-
/// Collect the underlying CType for each named enum definition.
/// GCC treats enum bitfields as unsigned when all enumerator values
/// are non-negative.  This function determines the correct underlying
/// type by examining the min/max values of each enum's enumerators.
fn collect_enum_underlying_types(
    specs: &[ast::TypeSpecifier],
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
    out: &mut FxHashMap<String, CType>,
) {
    for spec in specs {
        match spec {
            ast::TypeSpecifier::Enum(enum_spec) => {
                if let (Some(ref tag), Some(ref enumerators)) =
                    (&enum_spec.tag, &enum_spec.enumerators)
                {
                    let tag_idx = tag.as_u32() as usize;
                    if tag_idx >= name_table.len() {
                        continue;
                    }
                    let tag_name = &name_table[tag_idx];
                    // Check if __attribute__((packed))
                    let is_packed = enum_spec.attributes.iter().any(|a| {
                        let n = decl_lowering::resolve_sym(name_table, &a.name);
                        n == "packed" || n == "__packed__"
                    });
                    // Compute min/max of enumerator values.
                    let mut min_val: i128 = i128::MAX;
                    let mut max_val: i128 = i128::MIN;
                    let mut next_val: i128 = 0;
                    for enumerator in enumerators {
                        if let Some(ref val_expr) = enumerator.value {
                            if let Some(v) =
                                try_eval_const_int_expr_ctx(val_expr, name_table, enum_constants)
                            {
                                next_val = v;
                            }
                        }
                        if next_val < min_val {
                            min_val = next_val;
                        }
                        if next_val > max_val {
                            max_val = next_val;
                        }
                        next_val += 1;
                    }
                    let min_val = if min_val == i128::MAX { 0 } else { min_val };
                    let max_val = if max_val == i128::MIN { 0 } else { max_val };
                    let underlying = if is_packed {
                        if min_val >= 0 && max_val <= 255 {
                            CType::UChar
                        } else if min_val >= -128 && max_val <= 127 {
                            CType::SChar
                        } else if min_val >= 0 && max_val <= 65535 {
                            CType::UShort
                        } else if min_val >= -32768 && max_val <= 32767 {
                            CType::Short
                        } else {
                            CType::Int
                        }
                    } else if min_val >= 0 {
                        CType::UInt
                    } else {
                        CType::Int
                    };
                    out.insert(tag_name.clone(), underlying);
                }
            }
            ast::TypeSpecifier::Struct(su_spec) | ast::TypeSpecifier::Union(su_spec) => {
                if let Some(ref members) = su_spec.members {
                    for member in members {
                        collect_enum_underlying_types(
                            &member.specifiers.type_specifiers,
                            name_table,
                            enum_constants,
                            out,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// incrementing values.  If an enumerator has a constant expression,
/// we attempt simple integer literal evaluation; otherwise we fall
/// back to sequential assignment.
fn collect_enum_constants_from_specifiers(
    specs: &[ast::TypeSpecifier],
    name_table: &[String],
    out: &mut FxHashMap<String, i128>,
) {
    for spec in specs {
        match spec {
            ast::TypeSpecifier::Enum(enum_spec) => {
                if let Some(ref enumerators) = enum_spec.enumerators {
                    let mut next_val: i128 = 0;
                    for enumerator in enumerators {
                        // Try to evaluate the explicit value expression.
                        // Pass name_table and the partially-built map so that
                        // enum aliases like `E = A` can resolve `A` from the
                        // map of already-processed enumerators (Bug 36 fix).
                        if let Some(ref val_expr) = enumerator.value {
                            if let Some(v) = try_eval_const_int_expr_ctx(val_expr, name_table, out)
                            {
                                next_val = v;
                            }
                        }
                        // Resolve the enumerator name from the symbol.
                        let idx = enumerator.name.as_u32() as usize;
                        if idx < name_table.len() {
                            out.insert(name_table[idx].clone(), next_val);
                        }
                        next_val += 1;
                    }
                }
            }
            // Recurse into struct/union member declarations to find
            // anonymous enums used as member types.  In C, enum
            // constants declared inside a struct member definition
            // are visible at the enclosing scope (e.g., `struct S {
            // enum { A, B } kind; };` makes `A` and `B` file-scope).
            ast::TypeSpecifier::Struct(su_spec) | ast::TypeSpecifier::Union(su_spec) => {
                if let Some(ref members) = su_spec.members {
                    for member in members {
                        collect_enum_constants_from_specifiers(
                            &member.specifiers.type_specifiers,
                            name_table,
                            out,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Recursively collect enum constants from block-scope declarations
/// inside a compound statement (function body). The Linux kernel defines
/// anonymous enums inside function bodies (e.g., `enum { MIX_INFLIGHT = 1U << 31 };`)
/// and these must be resolvable during expression lowering.
fn collect_enum_constants_from_compound(
    compound: &ast::CompoundStatement,
    name_table: &[String],
    out: &mut FxHashMap<String, i128>,
) {
    for item in &compound.items {
        match item {
            ast::BlockItem::Declaration(decl) => {
                collect_enum_constants_from_specifiers(
                    &decl.specifiers.type_specifiers,
                    name_table,
                    out,
                );
            }
            ast::BlockItem::Statement(stmt) => {
                collect_enum_constants_from_statement(stmt, name_table, out);
            }
        }
    }
}

/// Recursively collect enum constants from statements (handles nested
/// compound statements, if/else, loops, switch, labeled statements, etc.)
fn collect_enum_constants_from_statement(
    stmt: &ast::Statement,
    name_table: &[String],
    out: &mut FxHashMap<String, i128>,
) {
    match stmt {
        ast::Statement::Compound(compound) => {
            collect_enum_constants_from_compound(compound, name_table, out);
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_enum_constants_from_statement(then_branch, name_table, out);
            if let Some(eb) = else_branch {
                collect_enum_constants_from_statement(eb, name_table, out);
            }
        }
        ast::Statement::While { body, .. } | ast::Statement::DoWhile { body, .. } => {
            collect_enum_constants_from_statement(body, name_table, out);
        }
        ast::Statement::For { init, body, .. } => {
            if let Some(ast::ForInit::Declaration(decl)) = init {
                collect_enum_constants_from_specifiers(
                    &decl.specifiers.type_specifiers,
                    name_table,
                    out,
                );
            }
            collect_enum_constants_from_statement(body, name_table, out);
        }
        ast::Statement::Switch { body, .. } => {
            collect_enum_constants_from_statement(body, name_table, out);
        }
        ast::Statement::Labeled { statement, .. }
        | ast::Statement::Case { statement, .. }
        | ast::Statement::Default { statement, .. } => {
            collect_enum_constants_from_statement(statement, name_table, out);
        }
        _ => {}
    }
}

/// Attempt to evaluate a simple integer constant expression from the AST.
///
/// Handles:
/// - Integer literals
/// - Unary minus on integer literals
/// - Simple binary arithmetic (+, -, *, /) on integer literals
///
/// Returns `None` if the expression is too complex for static evaluation.
/// Collect struct/union type definitions from type specifiers.
///
/// Scans top-level declaration specifiers for `struct` or `union`
/// definitions (those with a tag name AND member list) and registers
/// the full `CType` (with populated fields) in the output map, keyed
/// by tag name. This allows forward references like `struct S s;` in
/// function bodies to resolve the full struct layout.
/// Recursively resolve forward-referenced struct/union types inside the
/// field types of a struct or union definition.  This ensures that
/// `sizeof_ctype` returns the correct size for typedef fields whose
/// underlying struct was forward-declared at typedef creation time.
/// Check if a CType (struct/union) has any fields that are forward-
/// referenced struct/union types (i.e., named struct/union with 0 fields),
/// including nested fields up to a reasonable depth.
/// Used to detect convergence during multi-pass forward-ref resolution.
fn has_empty_struct_refs(ty: &CType) -> bool {
    has_empty_struct_refs_inner(ty, 0)
}

fn has_empty_struct_refs_inner(ty: &CType, depth: usize) -> bool {
    if depth > 8 {
        return false; // prevent infinite recursion
    }
    match ty {
        CType::Struct { ref fields, .. } | CType::Union { ref fields, .. } => {
            for field in fields {
                if has_empty_type_ref(&field.ty, depth + 1) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

fn has_empty_type_ref(ty: &CType, depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    match ty {
        CType::Struct {
            name: Some(_),
            fields,
            ..
        }
        | CType::Union {
            name: Some(_),
            fields,
            ..
        } if fields.is_empty() => true,
        CType::Struct { fields, .. } | CType::Union { fields, .. } => {
            for field in fields {
                if has_empty_type_ref(&field.ty, depth + 1) {
                    return true;
                }
            }
            false
        }
        CType::Pointer(inner, _)
        | CType::Array(inner, _)
        | CType::Atomic(inner)
        | CType::Qualified(inner, _)
        | CType::Typedef {
            underlying: inner, ..
        } => has_empty_type_ref(inner, depth + 1),
        _ => false,
    }
}

fn resolve_struct_field_forward_refs(ty: &mut CType, tags: &FxHashMap<String, CType>) {
    resolve_struct_field_forward_refs_inner(ty, tags, 0);
}

fn resolve_struct_field_forward_refs_inner(
    ty: &mut CType,
    tags: &FxHashMap<String, CType>,
    depth: usize,
) {
    // Limit recursion depth to prevent quadratic expansion on large
    // codebases like the Linux kernel.  Depth 6 covers the vast
    // majority of real-world struct nesting (typically 3-4 levels).
    if depth > 6 {
        return;
    }
    match ty {
        CType::Struct { ref mut fields, .. } | CType::Union { ref mut fields, .. } => {
            for field in fields.iter_mut() {
                resolve_type_forward_refs_deep(&mut field.ty, tags, depth + 1);
            }
        }
        _ => {}
    }
}

/// Deep forward-reference resolution: resolve the outermost lightweight
/// tag reference and then recurse into the resolved definition's fields
/// to resolve nested forward references.  Depth-limited to prevent
/// quadratic expansion on large codebases.
fn resolve_type_forward_refs_deep(ty: &mut CType, tags: &FxHashMap<String, CType>, depth: usize) {
    if depth > 6 {
        return;
    }
    match ty {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = tags.get(&tag_owned) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ty = full_def.clone();
                    // Recurse into the newly-resolved struct's fields
                    // to resolve nested forward refs (e.g., S2 → S1 → S0).
                    resolve_struct_field_forward_refs_inner(ty, tags, depth + 1);
                }
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = tags.get(&tag_owned) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ty = full_def.clone();
                    resolve_struct_field_forward_refs_inner(ty, tags, depth + 1);
                }
            }
        }
        CType::Struct { ref mut fields, .. } | CType::Union { ref mut fields, .. } => {
            // Not a forward ref, but still recurse into fields
            // to resolve nested forward refs.
            for field in fields.iter_mut() {
                resolve_type_forward_refs_deep(&mut field.ty, tags, depth + 1);
            }
        }
        CType::Typedef {
            ref mut underlying, ..
        } => {
            resolve_type_forward_refs_deep(underlying, tags, depth + 1);
        }
        CType::Array(ref mut inner, _) => {
            resolve_type_forward_refs_deep(inner, tags, depth + 1);
        }
        CType::Qualified(ref mut inner, _) => {
            resolve_type_forward_refs_deep(inner, tags, depth + 1);
        }
        CType::Atomic(ref mut inner) => {
            resolve_type_forward_refs_deep(inner, tags, depth + 1);
        }
        CType::Pointer(ref mut inner, _) => {
            resolve_type_forward_refs_deep(inner, tags, depth + 1);
        }
        _ => {}
    }
}

/// Shallow forward-reference resolution: resolve only the outermost
/// lightweight tag reference without recursing into the resolved
/// Recursively collect struct/union definitions from within a compound
/// statement (function body). This handles function-local struct definitions
/// like `void f() { struct S { int x; char y; }; ... }` which are not
/// visible to the top-level struct collection pass.
fn collect_struct_defs_from_compound(
    compound: &ast::CompoundStatement,
    name_table: &[String],
    out: &mut FxHashMap<String, CType>,
) {
    for item in &compound.items {
        match item {
            ast::BlockItem::Declaration(decl) => {
                collect_struct_defs_from_specifiers(
                    &decl.specifiers.type_specifiers,
                    name_table,
                    out,
                );
            }
            ast::BlockItem::Statement(stmt) => {
                collect_struct_defs_from_statement(stmt, name_table, out);
            }
        }
    }
}

/// Recursively collect struct/union definitions from a statement.
fn collect_struct_defs_from_statement(
    stmt: &ast::Statement,
    name_table: &[String],
    out: &mut FxHashMap<String, CType>,
) {
    match stmt {
        ast::Statement::Compound(compound) => {
            collect_struct_defs_from_compound(compound, name_table, out);
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_struct_defs_from_statement(then_branch, name_table, out);
            if let Some(eb) = else_branch {
                collect_struct_defs_from_statement(eb, name_table, out);
            }
        }
        ast::Statement::While { body, .. }
        | ast::Statement::DoWhile { body, .. }
        | ast::Statement::Switch { body, .. }
        | ast::Statement::Labeled {
            statement: body, ..
        }
        | ast::Statement::Case {
            statement: body, ..
        }
        | ast::Statement::Default {
            statement: body, ..
        } => {
            collect_struct_defs_from_statement(body, name_table, out);
        }
        ast::Statement::For { init, body, .. } => {
            if let Some(ast::ForInit::Declaration(decl)) = init {
                collect_struct_defs_from_specifiers(
                    &decl.specifiers.type_specifiers,
                    name_table,
                    out,
                );
            }
            collect_struct_defs_from_statement(body, name_table, out);
        }
        _ => {}
    }
}

fn collect_struct_defs_from_specifiers(
    specs: &[ast::TypeSpecifier],
    name_table: &[String],
    out: &mut FxHashMap<String, CType>,
) {
    for spec in specs {
        match spec {
            ast::TypeSpecifier::Struct(s) => {
                if let (Some(ref tag), Some(ref members)) = (&s.tag, &s.members) {
                    let tag_name_idx = tag.as_u32() as usize;
                    if tag_name_idx < name_table.len() {
                        let tag_name = name_table[tag_name_idx].clone();
                        let fields = decl_lowering::extract_struct_union_fields_fast(s, name_table);
                        let packed = s.attributes.iter().any(|a| {
                            let n = decl_lowering::resolve_sym(name_table, &a.name);
                            n == "packed" || n == "__packed__"
                        });
                        let mut aligned =
                            decl_lowering::extract_alignment_attribute(&s.attributes, name_table);
                        // Propagate field-level aligned to struct alignment.
                        if let Some(field_align) =
                            decl_lowering::extract_max_member_alignment(members, name_table)
                        {
                            aligned = Some(aligned.map_or(field_align, |a| a.max(field_align)));
                        }
                        let ctype = CType::Struct {
                            name: Some(tag_name.clone()),
                            fields,
                            packed,
                            aligned,
                        };
                        out.insert(tag_name, ctype);
                    }
                    // Recurse into member declarations to collect nested
                    // struct/union definitions (e.g. `struct _ht { ... }` defined
                    // inside a member of `struct Hash`).
                    for member in members {
                        collect_struct_defs_from_specifiers(
                            &member.specifiers.type_specifiers,
                            name_table,
                            out,
                        );
                    }
                } else if let Some(ref members) = s.members {
                    // Anonymous struct with members — still recurse.
                    for member in members {
                        collect_struct_defs_from_specifiers(
                            &member.specifiers.type_specifiers,
                            name_table,
                            out,
                        );
                    }
                }
            }
            ast::TypeSpecifier::Union(u) => {
                if let (Some(ref tag), Some(ref members)) = (&u.tag, &u.members) {
                    let tag_name_idx = tag.as_u32() as usize;
                    if tag_name_idx < name_table.len() {
                        let tag_name = name_table[tag_name_idx].clone();
                        let fields = decl_lowering::extract_struct_union_fields_fast(u, name_table);
                        let packed = u.attributes.iter().any(|a| {
                            let n = decl_lowering::resolve_sym(name_table, &a.name);
                            n == "packed" || n == "__packed__"
                        });
                        let mut aligned =
                            decl_lowering::extract_alignment_attribute(&u.attributes, name_table);
                        // Propagate field-level aligned to union alignment.
                        if let Some(field_align) =
                            decl_lowering::extract_max_member_alignment(members, name_table)
                        {
                            aligned = Some(aligned.map_or(field_align, |a| a.max(field_align)));
                        }
                        let ctype = CType::Union {
                            name: Some(tag_name.clone()),
                            fields,
                            packed,
                            aligned,
                        };
                        out.insert(tag_name, ctype);
                    }
                    // Recurse into member declarations.
                    for member in members {
                        collect_struct_defs_from_specifiers(
                            &member.specifiers.type_specifiers,
                            name_table,
                            out,
                        );
                    }
                } else if let Some(ref members) = u.members {
                    for member in members {
                        collect_struct_defs_from_specifiers(
                            &member.specifiers.type_specifiers,
                            name_table,
                            out,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Evaluate a compile-time integer constant expression, with access to
/// a name table and an existing enum constant map for resolving
/// identifiers that reference previously-defined enum constants.
/// This is essential for enum aliases like:
///   `PTHREAD_MUTEX_RECURSIVE = PTHREAD_MUTEX_RECURSIVE_NP`
/// where the value expression is an identifier.
fn try_eval_const_int_expr_ctx(
    expr: &ast::Expression,
    name_table: &[String],
    enum_map: &FxHashMap<String, i128>,
) -> Option<i128> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as i128),
        // Resolve identifier references to previously-defined enum constants.
        ast::Expression::Identifier { name, .. } => {
            let idx = name.as_u32() as usize;
            if idx < name_table.len() {
                enum_map.get(&name_table[idx]).copied()
            } else {
                None
            }
        }
        ast::Expression::UnaryOp { op, operand, .. } => match op {
            ast::UnaryOp::Negate => {
                try_eval_const_int_expr_ctx(operand, name_table, enum_map).map(|v| -v)
            }
            ast::UnaryOp::Plus => try_eval_const_int_expr_ctx(operand, name_table, enum_map),
            ast::UnaryOp::BitwiseNot => {
                try_eval_const_int_expr_ctx(operand, name_table, enum_map).map(|v| !v)
            }
            _ => None,
        },
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = try_eval_const_int_expr_ctx(left, name_table, enum_map)?;
            let r = try_eval_const_int_expr_ctx(right, name_table, enum_map)?;
            match op {
                ast::BinaryOp::Add => Some(l + r),
                ast::BinaryOp::Sub => Some(l - r),
                ast::BinaryOp::Mul => Some(l * r),
                ast::BinaryOp::Div => {
                    if r != 0 {
                        Some(l / r)
                    } else {
                        None
                    }
                }
                ast::BinaryOp::Mod => {
                    if r != 0 {
                        Some(l % r)
                    } else {
                        None
                    }
                }
                ast::BinaryOp::ShiftLeft => Some(l << (r as u32)),
                ast::BinaryOp::ShiftRight => Some(l >> (r as u32)),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                ast::BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                ast::BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                ast::BinaryOp::Less => Some(if l < r { 1 } else { 0 }),
                ast::BinaryOp::LessEqual => Some(if l <= r { 1 } else { 0 }),
                ast::BinaryOp::Greater => Some(if l > r { 1 } else { 0 }),
                ast::BinaryOp::GreaterEqual => Some(if l >= r { 1 } else { 0 }),
                ast::BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                ast::BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
            }
        }
        ast::Expression::Cast { operand, .. } => {
            try_eval_const_int_expr_ctx(operand, name_table, enum_map)
        }
        // Parenthesized expression — just unwrap.
        ast::Expression::Parenthesized { inner, .. } => {
            try_eval_const_int_expr_ctx(inner, name_table, enum_map)
        }
        // Ternary / conditional expression: `cond ? then : else`
        // Required for glibc's ctype.h which uses _ISbit() macro
        // that expands to a ternary in enum initializers.
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let cond = try_eval_const_int_expr_ctx(condition, name_table, enum_map)?;
            if cond != 0 {
                // If then_expr is None, this is the GCC `x ?: y` extension —
                // the result is the condition value itself.
                if let Some(ref then_e) = then_expr {
                    try_eval_const_int_expr_ctx(then_e, name_table, enum_map)
                } else {
                    Some(cond)
                }
            } else {
                try_eval_const_int_expr_ctx(else_expr, name_table, enum_map)
            }
        }
        // Comma expression — evaluate all sub-expressions, return the last value.
        ast::Expression::Comma { exprs, .. } => {
            let mut last = None;
            for e in exprs {
                last = try_eval_const_int_expr_ctx(e, name_table, enum_map);
            }
            last
        }
        // Sizeof(type) — needed for enum initializers that reference sizeof.
        ast::Expression::SizeofType { type_name, .. } => {
            let target = crate::common::target::Target::X86_64;
            let tb = crate::common::type_builder::TypeBuilder::new(target);
            let ct = resolve_type_name_for_const(type_name, &target, &tb);
            Some(crate::common::types::sizeof_ctype(&ct, &target) as i128)
        }
        _ => None,
    }
}

// =========================================================================
// resolve_type_name_for_const — standalone type name resolution for the
// constant expression evaluator (no ExprLoweringContext required).
// =========================================================================

/// Resolve an AST [`TypeName`] to a [`CType`] without requiring a full
/// `ExprLoweringContext`.  Used by the global-initializer constant evaluator
/// to apply correct C-type truncation on cast expressions.
pub fn resolve_type_name_for_const(
    tn: &ast::TypeName,
    _target: &Target,
    _type_builder: &TypeBuilder,
) -> CType {
    // Resolve the base type from the specifier-qualifier list.
    let base = decl_lowering::resolve_base_type_from_sqlist(&tn.specifier_qualifiers);

    // Apply the abstract declarator (pointer / array / function layers).
    if let Some(ref abs_decl) = tn.abstract_declarator {
        apply_abstract_declarator_const(base, abs_decl)
    } else {
        base
    }
}

/// Apply an abstract declarator to a base type (standalone helper for const eval).
fn apply_abstract_declarator_const(base: CType, abs: &ast::AbstractDeclarator) -> CType {
    let mut result = if let Some(ref pointer) = abs.pointer {
        apply_pointer_layers_const(base, pointer)
    } else {
        base
    };
    if let Some(ref direct) = abs.direct {
        result = apply_direct_abstract_declarator_const(result, direct);
    }
    result
}

fn apply_pointer_layers_const(base: CType, pointer: &ast::Pointer) -> CType {
    let quals = crate::common::types::TypeQualifiers::default();
    let mut current = CType::Pointer(Box::new(base), quals);
    if let Some(ref inner) = pointer.inner {
        current = apply_pointer_layers_const(current, inner);
    }
    current
}

fn apply_direct_abstract_declarator_const(
    base: CType,
    direct: &ast::DirectAbstractDeclarator,
) -> CType {
    match direct {
        ast::DirectAbstractDeclarator::Parenthesized(inner) => {
            apply_abstract_declarator_const(base, inner)
        }
        ast::DirectAbstractDeclarator::Array { size, .. } => {
            let len = size
                .as_ref()
                .and_then(|e| decl_lowering::evaluate_const_int_expr_pub(e));
            CType::Array(Box::new(base), len)
        }
        ast::DirectAbstractDeclarator::Function { .. } => CType::Pointer(
            Box::new(base),
            crate::common::types::TypeQualifiers::default(),
        ),
    }
}
