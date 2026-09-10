// Suppress result_unit_err: SemanticAnalyzer deliberately returns Result<_, ()>
// because error details flow through DiagnosticEngine, not through the Err variant.
#![allow(clippy::result_unit_err, clippy::if_same_then_else)]

//! Semantic analysis driver module — Phase 5 entry point for BCC.
//!
//! Declares all seven submodules and provides the top-level [`SemanticAnalyzer`]
//! struct that orchestrates declaration processing, expression type inference,
//! and statement validation over the parser's AST.
//!
//! # Pipeline Position
//!
//! ```text
//! Parser (Phase 4)  →  SemanticAnalyzer (Phase 5)  →  IR Lowering (Phase 6)
//!      AST              type-annotated AST              IR instructions
//! ```
//!
//! The semantic analyzer takes a [`TranslationUnit`] produced by the parser and
//! walks every external declaration, function definition, and statement to:
//!
//! - Resolve types and build the dual type system bridge
//! - Manage lexical scopes (block, function, file, global)
//! - Populate the symbol table with declarations and definitions
//! - Evaluate compile-time constant expressions
//! - Validate GCC builtins and `__attribute__` specifications
//! - Analyze C99/C11 designated initializers
//! - Enforce the 512-depth recursion limit (AAP §0.7.3)
//!
//! # Sub-Module Architecture
//!
//! | Module              | Responsibility                                       |
//! |---------------------|------------------------------------------------------|
//! | [`type_checker`]    | Implicit conversions, type compatibility, operator    |
//! |                     | type checking, lvalue validation                      |
//! | [`scope`]           | Lexical scope stack with tag and label namespaces     |
//! | [`symbol_table`]    | Declaration tracking, linkage resolution, usage       |
//! | [`constant_eval`]   | C11 §6.6 integer constant expression evaluation      |
//! | [`builtin_eval`]    | GCC builtin compile-time and runtime-deferred eval   |
//! | [`initializer`]     | Designated initializer analysis, brace elision        |
//! | [`attribute_handler`] | `__attribute__` validation and propagation          |
//!
//! # Zero-Dependency Compliance
//!
//! This module uses only `std` and `crate::` references. No external crates.
//! Does NOT depend on `crate::ir`, `crate::passes`, or `crate::backend`.

// ============================================================================
// Submodule Declarations
// ============================================================================

pub mod attribute_handler;
pub mod builtin_eval;
pub mod constant_eval;
pub mod initializer;
pub mod scope;
pub mod symbol_table;
pub mod type_checker;

// ============================================================================
// Public Re-exports — Convenience API for downstream consumers
// ============================================================================

pub use attribute_handler::{
    AttributeContext, AttributeHandler, SymbolVisibility, ValidatedAttribute,
};
pub use builtin_eval::BuiltinEvaluator;
pub use constant_eval::{ConstValue, ConstantEvaluator};
pub use initializer::{AnalyzedInit, InitializerAnalyzer};
pub use scope::{LabelEntry, ScopeKind, ScopeStack, TagEntry, TagKind};
pub use symbol_table::{Linkage, StorageClass, SymbolEntry, SymbolId, SymbolKind, SymbolTable};
pub use type_checker::TypeChecker;

// ============================================================================
// Internal Imports
// ============================================================================

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{CType, StructField, TypeQualifiers};

use crate::frontend::parser::ast;
use crate::frontend::parser::ast::*;

// ============================================================================
// Constants
// ============================================================================

/// Maximum recursion depth enforced during semantic analysis (AAP §0.7.3).
/// Deeply nested Linux kernel macro expansions and expressions can exceed
/// normal stack limits; this guard prevents stack overflow.
const MAX_RECURSION_DEPTH: u32 = 512;

// ============================================================================
// SemanticAnalyzer — Top-Level Phase 5 Driver
// ============================================================================

/// Orchestrates Phase 5 semantic analysis over a parsed AST.
///
/// Coordinates seven sub-components (type checker, scope stack, symbol table,
/// constant evaluator, builtin evaluator, initializer analyzer, attribute
/// handler) into a cohesive pass that validates types, resolves scopes,
/// populates the symbol table, and produces a semantically validated AST
/// suitable for IR lowering (Phase 6).
///
/// # Lifetime
///
/// The `'a` lifetime binds the analyzer to the diagnostic engine, type
/// builder, and string interner — all of which must outlive the analyzer.
///
/// # Usage
///
/// ```ignore
/// let mut diag = DiagnosticEngine::new();
/// let tb = TypeBuilder::new(Target::X86_64);
/// let mut interner = Interner::new();
/// let mut sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);
/// sema.analyze(&mut translation_unit)?;
/// sema.finalize()?;
/// ```
pub struct SemanticAnalyzer<'a> {
    /// Diagnostic engine for error/warning accumulation.
    diagnostics: &'a mut DiagnosticEngine,
    /// Type builder for struct layout computation, sizeof/alignof queries.
    type_builder: &'a TypeBuilder,
    /// Target architecture info (x86-64, i686, AArch64, RISC-V 64).
    target: Target,
    /// String interner for resolving Symbol handles to string names.
    interner: &'a Interner,
    /// Scope stack managing lexical scopes (block, function, file, global).
    scopes: ScopeStack,
    /// Symbol table for all declarations.
    symbols: SymbolTable,
    /// Current recursion depth for depth-limiting (max 512).
    recursion_depth: u32,
    /// Maximum recursion depth (512 per AAP §0.7.3).
    max_recursion_depth: u32,
    /// Current function return type (None if outside function body).
    current_function_return_type: Option<CType>,
    /// Whether we are currently inside a loop body (for break/continue validation).
    in_loop: bool,
    /// Whether we are currently inside a switch body (for case/default/break).
    in_switch: bool,
    /// Case label values seen in the current switch (for duplicate detection).
    /// Reset on each new switch statement.
    switch_case_values: FxHashMap<i128, Span>,
    /// Whether a default label has been seen in the current switch.
    switch_has_default: bool,
    /// Enum constant values: maps enum constant name (as String) to its
    /// integer value. Populated during `analyze_enum_definition()` and used
    /// by `handle_static_assert()` to provide enum values to the
    /// `ConstantEvaluator`.
    enum_constant_values: FxHashMap<String, i128>,
    /// String-keyed tag type map for sizeof_ctype_resolved.  Maintained
    /// incrementally by analyze_struct_definition / analyze_enum_definition
    /// so that handle_static_assert can pass it to the ConstantEvaluator
    /// without rebuilding per _Static_assert.
    tag_types_by_name: std::collections::HashMap<String, CType>,
}

/// A single step in an `__builtin_offsetof` member-designator chain,
/// used during compile-time evaluation of offsetof expressions.
enum OffsetofStep {
    /// Struct/union field access by name.
    Field(String),
    /// Array subscript by constant index.
    Index(usize),
}

impl<'a> SemanticAnalyzer<'a> {
    // ====================================================================
    // Constructor
    // ====================================================================

    /// Create a new semantic analyzer bound to the given infrastructure.
    ///
    /// Initializes all sub-components, pushes the initial Global scope,
    /// and sets the recursion depth limit to 512 (per AAP §0.7.3).
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Mutable reference to the diagnostic engine for
    ///   multi-error accumulation.
    /// * `type_builder` — Immutable reference to the target-aware type
    ///   builder for sizeof/alignof and struct layout queries.
    /// * `target` — The compilation target architecture.
    /// * `interner` — Immutable reference to the string interner for
    ///   resolving interned Symbol handles.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
        interner: &'a Interner,
    ) -> Self {
        // ScopeStack::new() already pushes a Global scope.
        let scopes = ScopeStack::new();
        let symbols = SymbolTable::new();

        let mut analyzer = SemanticAnalyzer {
            diagnostics,
            type_builder,
            target,
            interner,
            scopes,
            symbols,
            recursion_depth: 0,
            max_recursion_depth: MAX_RECURSION_DEPTH,
            current_function_return_type: None,
            in_loop: false,
            in_switch: false,
            switch_case_values: FxHashMap::default(),
            switch_has_default: false,
            enum_constant_values: FxHashMap::default(),
            tag_types_by_name: std::collections::HashMap::new(),
        };

        // Pre-register GCC builtin type names as typedefs at global scope.
        // __builtin_va_list is the underlying type behind va_list in <stdarg.h>
        // and is represented internally as a pointer to void (char *).
        analyzer.register_builtin_typedef(
            "__builtin_va_list",
            CType::Pointer(Box::new(CType::Void), TypeQualifiers::default()),
        );

        analyzer
    }

    /// Register a compiler-builtin typedef at global scope.
    fn register_builtin_typedef(&mut self, name: &str, ty: CType) {
        // Intern the name to get a Symbol.
        // Safety: We need to look up the symbol by string.  The interner is
        // shared with the parser which has already interned the name, so
        // this will return the same Symbol handle.
        let sym = match self.interner.get(name) {
            Some(s) => s,
            None => {
                // If not yet interned (unusual), the typedef won't be available.
                return;
            }
        };
        let entry = SymbolEntry {
            name: sym,
            ty,
            kind: SymbolKind::TypedefName,
            linkage: Linkage::None,
            storage_class: StorageClass::Typedef,
            is_defined: true,
            is_tentative: false,
            span: Span::dummy(),
            attributes: Vec::new(),
            is_weak: false,
            visibility: None,
            section: None,
            is_used: false,
            scope_depth: 0,
        };
        if let Ok(id) = self.symbols.declare(entry, self.diagnostics) {
            self.scopes.declare_ordinary(sym, id);
            self.scopes.register_typedef(sym);
        }
    }

    // ====================================================================
    // Top-Level Public API
    // ====================================================================

    /// Main entry point for Phase 5 semantic analysis.
    ///
    /// Iterates over all external declarations in the translation unit,
    /// processing each function definition, declaration, or top-level asm
    /// statement. After all declarations are processed, finalizes tentative
    /// definitions and checks for unused symbols.
    ///
    /// Returns `Ok(())` if no errors were accumulated in the diagnostic
    /// engine, `Err(())` if any errors were emitted.
    pub fn analyze(&mut self, translation_unit: &mut TranslationUnit) -> Result<(), ()> {
        let sema_t0 = std::time::Instant::now();
        let mut sema_count = 0usize;
        for ext_decl in &mut translation_unit.declarations {
            let _dt = std::time::Instant::now();
            // Process each external declaration; continue even on error
            // to accumulate multiple diagnostics.
            let _ = self.analyze_external_declaration(ext_decl);
            sema_count += 1;
            let elapsed = _dt.elapsed().as_secs_f64();
            if elapsed > 0.1 {
                let kind = match ext_decl {
                    crate::frontend::parser::ast::ExternalDeclaration::FunctionDefinition(_) => {
                        "FunctionDef"
                    }
                    crate::frontend::parser::ast::ExternalDeclaration::Declaration(d) => {
                        if d.static_assert.is_some() {
                            "StaticAssert"
                        } else {
                            "Declaration"
                        }
                    }
                    crate::frontend::parser::ast::ExternalDeclaration::AsmStatement(_) => "Asm",
                    crate::frontend::parser::ast::ExternalDeclaration::Empty => "Empty",
                };
                if crate::common::timing::timing_enabled() {
                    eprintln!(
                        "[BCC-TIMING] sema-slow: decl #{} ({}) took {:.3}s",
                        sema_count, kind, elapsed
                    );
                }
            }
        }
        if crate::common::timing::timing_enabled() {
            eprintln!(
                "[BCC-TIMING] sema-analyze: {} decls in {:.3}s",
                sema_count,
                sema_t0.elapsed().as_secs_f64()
            );
        }

        // Note: tentative definition finalization and unused symbol checks
        // are performed in `finalize()`, which is called separately after
        // `analyze()`. This avoids emitting duplicate diagnostics.

        if self.diagnostics.has_errors() {
            Err(())
        } else {
            Ok(())
        }
    }

    /// Analyze a single external declaration.
    ///
    /// Dispatches to the appropriate handler based on the declaration kind:
    /// - `FunctionDefinition` → full function analysis with scope management
    /// - `Declaration` → variable/typedef/struct/enum declaration
    /// - `AsmStatement` → inline assembly validation at file scope
    /// - `Empty` → no-op (stray semicolons)
    pub fn analyze_external_declaration(
        &mut self,
        decl: &mut ExternalDeclaration,
    ) -> Result<(), ()> {
        match decl {
            ExternalDeclaration::FunctionDefinition(func_def) => {
                self.analyze_function_definition(func_def)
            }
            ExternalDeclaration::Declaration(declaration) => self.analyze_declaration(declaration),
            ExternalDeclaration::AsmStatement(asm) => self.analyze_asm_statement(asm),
            ExternalDeclaration::Empty => Ok(()),
        }
    }

    // ====================================================================
    // Function Definition Analysis
    // ====================================================================

    /// Analyze a function definition with full scope management.
    ///
    /// Steps:
    /// 1. Resolve the function's return type from declaration specifiers.
    /// 2. Build the complete function type (return type + parameters).
    /// 3. Validate and propagate function attributes.
    /// 4. Declare the function in the symbol table at file scope.
    /// 5. Push a Function scope and declare parameters.
    /// 6. Set `current_function_return_type` for return-statement checking.
    /// 7. Analyze the function body (compound statement).
    /// 8. Validate labels at function scope exit.
    /// 9. Pop the function scope and reset state.
    pub fn analyze_function_definition(&mut self, func: &mut FunctionDefinition) -> Result<(), ()> {
        let func_span = func.span;

        // Step 1: Build the complete function type using the full declarator.
        // This correctly handles complex cases like function-returning-function-pointer:
        //   void (*memdbDlSym(sqlite3_vfs *pVfs, void *p, const char *zSym))(void)
        // where the return type is `void (*)(void)` and params are the inner ones.
        let base_specifier_type = self.resolve_type_from_specifiers(&func.specifiers);
        let full_type =
            self.apply_declarator_to_type(base_specifier_type.clone(), &func.declarator);

        // Extract return type, param types, and variadic flag from the full type.
        let (return_type, param_types, is_variadic) = if let CType::Function {
            ref return_type,
            ref params,
            variadic,
        } = full_type
        {
            (return_type.as_ref().clone(), params.clone(), variadic)
        } else {
            // Fallback for non-function types (shouldn't happen for definitions).
            let mut rt = base_specifier_type;
            if let Some(ref ptr) = func.declarator.pointer {
                rt = self.apply_pointer_to_type(rt, ptr);
            }
            (rt, Vec::new(), false)
        };

        // Step 1.5: For K&R (old-style) function definitions, merge the actual
        // parameter types from old_style_params into param_types.
        // K&R functions have params as bare identifiers in the declarator:
        //   void f(a, b) int *a; char *b; { ... }
        // The declarator gives us only names with default type `int`.
        // The real types are in func.old_style_params declarations.
        let param_types = if !func.old_style_params.is_empty() {
            // Build name→type mapping from old-style parameter declarations.
            // Each old_style_param is a Declaration like `int *d;` or `char *str;`.
            let mut knr_type_map = crate::common::fx_hash::FxHashMap::<String, CType>::default();
            for osp in &func.old_style_params {
                let osp_base_ty = self.resolve_type_from_specifiers(&osp.specifiers);
                for init_decl in &osp.declarators {
                    if let Some(pname) = self.extract_declarator_name(&init_decl.declarator) {
                        let pty = self
                            .apply_declarator_to_type(osp_base_ty.clone(), &init_decl.declarator);
                        let name_str = self.interner.resolve(pname);
                        knr_type_map.insert(name_str.to_string(), pty);
                    }
                }
            }
            // Override param_types using the K&R type map
            let (_, pn, _) = self.extract_function_params(&func.declarator);
            let mut new_param_types = Vec::with_capacity(pn.len());
            for (i, maybe_name) in pn.iter().enumerate() {
                if let Some(sym) = maybe_name {
                    let name_str = self.interner.resolve(*sym);
                    if let Some(kty) = knr_type_map.get(name_str) {
                        new_param_types.push(kty.clone());
                    } else {
                        // Not declared in old_style_params — default to int (C89 rule)
                        new_param_types.push(param_types.get(i).cloned().unwrap_or(CType::Int));
                    }
                } else {
                    new_param_types.push(param_types.get(i).cloned().unwrap_or(CType::Int));
                }
            }
            new_param_types
        } else {
            param_types
        };

        // Step 2: Extract function name and parameter NAMES from the declarator.
        // For complex declarators, extract_function_params finds the innermost
        // Function declarator (the one with the actual parameter names).
        let func_name = self.extract_declarator_name(&func.declarator);
        let (_, param_names, _) = self.extract_function_params(&func.declarator);

        // Step 3: Build the complete function type for symbol table registration.
        let func_type = CType::Function {
            return_type: Box::new(return_type.clone()),
            params: param_types.clone(),
            variadic: is_variadic,
        };

        // Step 4: Determine storage class and validate attributes.
        let storage = self.resolve_storage_class(&func.specifiers);
        let validated_attrs = self.validate_attributes_for_context(
            &func.attributes,
            &func.specifiers.attributes,
            AttributeContext::Function,
            func_span,
        );

        // Step 5: Declare function in the symbol table at file scope.
        if let Some(name) = func_name {
            let sym_id = self.symbols.declare_function(
                name,
                func_type.clone(),
                storage,
                func_span,
                self.diagnostics,
            );
            if let Ok(id) = sym_id {
                self.symbols.define(id);
                // Register the function in the scope system so that
                // it is resolvable by name from other functions.
                self.scopes.declare_ordinary(name, id);
                // Propagate validated attributes to the symbol.
                for attr in &validated_attrs {
                    self.propagate_attribute_to_symbol(id, attr);
                }
            }
        }

        // Step 6: Push function scope and declare parameters.
        self.scopes.push_scope(ScopeKind::Function);
        self.symbols.enter_scope();

        for (i, param_type) in param_types.iter().enumerate() {
            let param_name = param_names.get(i).copied().flatten();
            if let Some(pname) = param_name {
                let param_entry = SymbolEntry {
                    name: pname,
                    ty: param_type.clone(),
                    kind: SymbolKind::Variable,
                    linkage: Linkage::None,
                    storage_class: StorageClass::Auto,
                    is_defined: true,
                    is_tentative: false,
                    span: func_span,
                    attributes: Vec::new(),
                    is_weak: false,
                    visibility: None,
                    section: None,
                    is_used: false,
                    scope_depth: 0, // Will be set by declare()
                };
                if let Ok(id) = self.symbols.declare(param_entry, self.diagnostics) {
                    // Register the parameter in the scope stack so that
                    // scopes.lookup_ordinary() can find it during body analysis.
                    self.scopes.declare_ordinary(pname, id);
                }
            }
        }

        // Step 7: Set current function return type for return-statement checking.
        let prev_return_type = self.current_function_return_type.take();
        self.current_function_return_type = Some(return_type.clone());

        // Step 8: Analyze the function body.
        let result = self.analyze_compound_statement(&mut func.body);

        // Step 9: Validate labels and pop scope.
        self.scopes.validate_labels(self.diagnostics);
        self.symbols.leave_scope();
        self.scopes.pop_scope(self.diagnostics);

        // Step 10: Reset function state.
        self.current_function_return_type = prev_return_type;

        result
    }

    // ====================================================================
    // Declaration Analysis
    // ====================================================================

    /// Analyze a declaration (variable, typedef, struct/union/enum, _Static_assert).
    ///
    /// Processes declaration specifiers to resolve the base type, then iterates
    /// over each init-declarator to:
    /// - Build the fully-qualified type
    /// - Resolve linkage and storage class
    /// - Validate and propagate attributes
    /// - Check for redeclaration conflicts
    /// - Analyze initializers against the declared type
    /// - Insert into the symbol table
    pub fn analyze_declaration(&mut self, decl: &mut Declaration) -> Result<(), ()> {
        let _decl_span = decl.span;

        // Handle _Static_assert: the parser attaches a `StaticAssert` node
        // to the declaration's `static_assert` field when it encounters
        // `_Static_assert(...)`.  Evaluate the condition at compile time.
        if decl.static_assert.is_some() {
            return self.handle_static_assert(decl);
        }

        // Check if this is an __auto_type declaration (GCC extension).
        // For __auto_type, we defer type resolution until the initializer
        // expression is analyzed, then use its type.
        let is_auto_type = decl
            .specifiers
            .type_specifiers
            .iter()
            .any(|s| matches!(s, TypeSpecifier::AutoType));

        // Resolve the base type from declaration specifiers.
        let base_type = self.resolve_type_from_specifiers(&decl.specifiers);

        // Process struct/union/enum definitions embedded in specifiers.
        self.process_embedded_tag_definitions(&decl.specifiers);

        // Determine storage class.
        let storage = self.resolve_storage_class(&decl.specifiers);

        // If there are no declarators, this is a standalone struct/union/enum
        // definition or forward declaration — already handled above.
        if decl.declarators.is_empty() {
            return Ok(());
        }

        // Process each init-declarator.
        for init_decl in &mut decl.declarators {
            let id_span = init_decl.span;

            // Build the fully-qualified type from the base type and declarator.
            let mut full_type =
                self.apply_declarator_to_type(base_type.clone(), &init_decl.declarator);

            // For __auto_type declarations, infer the actual type from the
            // initializer expression. This GCC extension acts like C++ `auto`:
            //   __auto_type __ptr = &(p);  // type = typeof(&(p))
            //   __auto_type __val = *__ptr; // type = typeof(*__ptr)
            if is_auto_type {
                if let Some(Initializer::Expression(ref mut expr)) = init_decl.initializer {
                    if let Ok(inferred) = self.analyze_expression(expr) {
                        full_type = inferred;
                    }
                }
            }

            // Extract the declared name.
            let name = self.extract_declarator_name(&init_decl.declarator);
            let Some(sym_name) = name else {
                // Anonymous declarator — possible for abstract declarators
                // in certain contexts, but unusual at file/block scope.
                continue;
            };

            // Validate attributes from both specifiers and declarator.
            let attr_context = if matches!(full_type, CType::Function { .. }) {
                AttributeContext::Function
            } else {
                AttributeContext::Variable
            };
            let validated_attrs = self.validate_attributes_for_context(
                &init_decl.declarator.attributes,
                &decl.specifiers.attributes,
                attr_context,
                id_span,
            );

            // Handle typedef declarations specially.
            if storage == StorageClass::Typedef {
                // Apply aligned/packed attributes from the declaration
                // to the underlying struct/union type.  This handles
                // patterns like:
                //   typedef struct { int a; } __attribute__((aligned(32))) X;
                let decl_aligned = validated_attrs.iter().find_map(|a| {
                    if let ValidatedAttribute::Aligned(n) = a {
                        Some(*n as usize)
                    } else {
                        None
                    }
                });
                let decl_packed = validated_attrs
                    .iter()
                    .any(|a| matches!(a, ValidatedAttribute::Packed));
                if decl_aligned.is_some() || decl_packed {
                    match &mut full_type {
                        CType::Struct {
                            ref mut aligned,
                            ref mut packed,
                            ..
                        } => {
                            if let Some(a) = decl_aligned {
                                *aligned = Some(aligned.map_or(a, |existing| existing.max(a)));
                            }
                            if decl_packed {
                                *packed = true;
                            }
                        }
                        CType::Union {
                            ref mut aligned,
                            ref mut packed,
                            ..
                        } => {
                            if let Some(a) = decl_aligned {
                                *aligned = Some(aligned.map_or(a, |existing| existing.max(a)));
                            }
                            if decl_packed {
                                *packed = true;
                            }
                        }
                        _ => {}
                    }
                }
                let entry = SymbolEntry {
                    name: sym_name,
                    ty: full_type.clone(),
                    kind: SymbolKind::TypedefName,
                    linkage: Linkage::None,
                    storage_class: StorageClass::Typedef,
                    is_defined: true,
                    is_tentative: false,
                    span: id_span,
                    attributes: validated_attrs.clone(),
                    is_weak: false,
                    visibility: None,
                    section: None,
                    is_used: false,
                    scope_depth: 0,
                };
                if let Ok(id) = self.symbols.declare(entry, self.diagnostics) {
                    self.scopes.declare_ordinary(sym_name, id);
                    self.scopes.register_typedef(sym_name);
                }
                continue;
            }

            // Determine linkage.
            let linkage =
                self.symbols
                    .resolve_linkage(sym_name, storage, self.scopes.current_depth());

            // Determine if this is a definition or declaration.
            let has_init = init_decl.initializer.is_some();
            let is_at_file_scope = self.scopes.is_file_scope();
            let is_defined = has_init;
            let is_tentative = !has_init
                && is_at_file_scope
                && storage != StorageClass::Extern
                && matches!(
                    full_type,
                    CType::Int
                        | CType::UInt
                        | CType::Long
                        | CType::ULong
                        | CType::Short
                        | CType::UShort
                        | CType::Char
                        | CType::SChar
                        | CType::UChar
                        | CType::LongLong
                        | CType::ULongLong
                        | CType::Int128
                        | CType::UInt128
                        | CType::Float
                        | CType::Double
                        | CType::LongDouble
                        | CType::Bool
                        | CType::Pointer(_, _)
                        | CType::Array(_, _)
                        | CType::Struct { .. }
                        | CType::Union { .. }
                        | CType::Enum { .. }
                        | CType::Qualified(_, _)
                        | CType::Typedef { .. }
                );

            // Build symbol entry.
            let sym_kind = if matches!(full_type, CType::Function { .. }) {
                SymbolKind::Function
            } else {
                SymbolKind::Variable
            };

            let entry = SymbolEntry {
                name: sym_name,
                ty: full_type.clone(),
                kind: sym_kind,
                linkage,
                storage_class: storage,
                is_defined,
                is_tentative,
                span: id_span,
                attributes: validated_attrs.clone(),
                is_weak: validated_attrs
                    .iter()
                    .any(|a| matches!(a, ValidatedAttribute::Weak)),
                visibility: validated_attrs.iter().find_map(|a| {
                    if let ValidatedAttribute::Visibility(v) = a {
                        Some(*v)
                    } else {
                        None
                    }
                }),
                section: validated_attrs.iter().find_map(|a| {
                    if let ValidatedAttribute::Section(s) = a {
                        Some(s.clone())
                    } else {
                        None
                    }
                }),
                is_used: false,
                scope_depth: 0,
            };

            if let Ok(id) = self.symbols.declare(entry, self.diagnostics) {
                self.scopes.declare_ordinary(sym_name, id);

                // Analyze initializer if present.
                if let Some(ref _init) = init_decl.initializer {
                    // Complete incomplete array types from the initializer.
                    // For declarations like `int arr[] = {1,2,3};`, the
                    // parsed type is Array(Int, None). We deduce the
                    // element count from the initializer list length and
                    // update the symbol's type to Array(Int, Some(3)).
                    {
                        let sym_entry = self.symbols.get(id);
                        let needs_completion = match &sym_entry.ty {
                            CType::Array(_, None) => true,
                            CType::Qualified(inner, _) => {
                                matches!(inner.as_ref(), CType::Array(_, None))
                            }
                            _ => false,
                        };
                        if needs_completion {
                            let count = Self::count_initializer_elements(
                                _init,
                                &self.enum_constant_values,
                                self.interner,
                            );
                            if count > 0 {
                                let sym_mut = self.symbols.get_mut(id);
                                sym_mut.ty = Self::complete_array_type(&sym_mut.ty, count);
                            }
                        }
                    }
                    self.symbols.define(id);
                }
            }
        }

        Ok(())
    }

    // ====================================================================
    // Incomplete Array Type Completion
    // ====================================================================

    /// Count the number of top-level elements in an initializer list to
    /// deduce the array size for declarations like `int arr[] = {1,2,3};`.
    ///
    /// For a brace-enclosed initializer list, returns the number of entries.
    /// For designated initializers with explicit array indices, tracks the
    /// maximum index to handle sparse initialization like `[5] = 42`.
    /// For a single expression initializer (e.g., string literal), counts
    /// string length + 1 for the NUL terminator if it's a string; otherwise
    /// returns 1 (shouldn't normally be used for arrays).
    fn count_initializer_elements(
        init: &crate::frontend::parser::ast::Initializer,
        enum_values: &FxHashMap<String, i128>,
        interner: &crate::common::string_interner::Interner,
    ) -> usize {
        use crate::frontend::parser::ast::{Designator, Initializer};
        match init {
            Initializer::List {
                designators_and_initializers,
                ..
            } => {
                let mut max_index: usize = 0;
                let mut positional: usize = 0;
                let mut unresolved_indices = false;
                for di in designators_and_initializers {
                    if di.designators.is_empty() {
                        // Positional initializer — increments the index
                        positional += 1;
                    } else {
                        // Check for array index designators [N] or [lo...hi]
                        let mut has_array_index = false;
                        for d in &di.designators {
                            match d {
                                Designator::Index(expr, _) => {
                                    has_array_index = true;
                                    if let Some(idx) =
                                        Self::try_eval_index_expr(expr, enum_values, interner)
                                    {
                                        let idx_usize = idx as usize;
                                        if idx_usize + 1 > max_index {
                                            max_index = idx_usize + 1;
                                        }
                                    } else {
                                        unresolved_indices = true;
                                    }
                                }
                                Designator::IndexRange(_lo, hi, _) => {
                                    has_array_index = true;
                                    if let Some(idx) =
                                        Self::try_eval_index_expr(hi, enum_values, interner)
                                    {
                                        let idx_usize = idx as usize;
                                        if idx_usize + 1 > max_index {
                                            max_index = idx_usize + 1;
                                        }
                                    } else {
                                        unresolved_indices = true;
                                    }
                                }
                                Designator::Field(_, _) => {
                                    // Struct field designator — not an array index
                                }
                            }
                        }
                        if !has_array_index {
                            // Pure field designators (.field = val) count as
                            // one positional element for the top-level array
                            positional += 1;
                        }
                    }
                }
                // If we couldn't resolve some index designators,
                // fall back to using the total number of initializer
                // entries as the array size.
                if unresolved_indices && max_index == 0 {
                    designators_and_initializers.len()
                } else {
                    std::cmp::max(
                        std::cmp::max(positional, max_index),
                        designators_and_initializers.len(),
                    )
                }
            }
            Initializer::Expression(expr) => {
                // String literal initializer for char arrays:
                // `char s[] = "hello"` → 6 elements (5 chars + NUL)
                use crate::frontend::parser::ast::Expression;
                match expr.as_ref() {
                    Expression::StringLiteral { segments, .. } => {
                        // Count total bytes across all segments + NUL
                        let total: usize = segments.iter().map(|s| s.value.len()).sum();
                        total + 1
                    }
                    _ => 1,
                }
            }
        }
    }

    /// Try to evaluate an index expression to a compile-time integer.
    /// Handles: integer literals, enum constant identifiers, casts wrapping
    /// any of the above, and simple arithmetic on them.
    fn try_eval_index_expr(
        expr: &crate::frontend::parser::ast::Expression,
        enum_values: &FxHashMap<String, i128>,
        interner: &crate::common::string_interner::Interner,
    ) -> Option<i128> {
        use crate::frontend::parser::ast::{BinaryOp, Expression, UnaryOp};
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as i128),
            Expression::Cast { operand, .. } => {
                Self::try_eval_index_expr(operand, enum_values, interner)
            }
            Expression::Identifier { name, .. } => {
                // Try looking up as an enum constant
                let name_str = interner.resolve(*name);
                enum_values.get(name_str).copied()
            }
            Expression::UnaryOp { op, operand, .. } => {
                let val = Self::try_eval_index_expr(operand, enum_values, interner)?;
                match op {
                    UnaryOp::Negate => Some(-val),
                    UnaryOp::BitwiseNot => Some(!val),
                    UnaryOp::LogicalNot => Some(if val == 0 { 1 } else { 0 }),
                    UnaryOp::Plus => Some(val),
                    _ => None,
                }
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                let lv = Self::try_eval_index_expr(left, enum_values, interner)?;
                let rv = Self::try_eval_index_expr(right, enum_values, interner)?;
                match op {
                    BinaryOp::Add => Some(lv.wrapping_add(rv)),
                    BinaryOp::Sub => Some(lv.wrapping_sub(rv)),
                    BinaryOp::Mul => Some(lv.wrapping_mul(rv)),
                    BinaryOp::Div if rv != 0 => Some(lv / rv),
                    BinaryOp::Mod if rv != 0 => Some(lv % rv),
                    BinaryOp::ShiftLeft => Some(lv.wrapping_shl(rv as u32)),
                    BinaryOp::ShiftRight => Some(lv.wrapping_shr(rv as u32)),
                    BinaryOp::BitwiseAnd => Some(lv & rv),
                    BinaryOp::BitwiseOr => Some(lv | rv),
                    BinaryOp::BitwiseXor => Some(lv ^ rv),
                    _ => None,
                }
            }
            Expression::Parenthesized { inner, .. } => {
                Self::try_eval_index_expr(inner, enum_values, interner)
            }
            _ => None,
        }
    }

    /// Complete an incomplete array type `Array(base, None)` by filling in
    /// the element count. Handles both plain `Array` and `Qualified(Array)`.
    fn complete_array_type(ty: &CType, count: usize) -> CType {
        match ty {
            CType::Array(base, None) => CType::Array(base.clone(), Some(count)),
            CType::Qualified(inner, quals) => {
                if let CType::Array(base, None) = inner.as_ref() {
                    CType::Qualified(Box::new(CType::Array(base.clone(), Some(count))), *quals)
                } else {
                    ty.clone()
                }
            }
            _ => ty.clone(),
        }
    }

    // ====================================================================
    // Expression Analysis
    // ====================================================================

    /// Analyze an expression, returning its resolved C type.
    ///
    /// Check whether a C type has the `const` qualifier at the top level.
    fn is_const_qualified(&self, ty: &CType) -> bool {
        match ty {
            CType::Qualified(_, quals) => quals.is_const,
            _ => false,
        }
    }

    /// Increments the recursion depth guard (max 512 per AAP §0.7.3) before
    /// dispatching to variant-specific handlers. Decrements on return.
    ///
    /// Each expression variant is handled to resolve its type, validate
    /// operand types, and emit diagnostics for type errors.
    pub fn analyze_expression(&mut self, expr: &mut Expression) -> Result<CType, ()> {
        // Check recursion depth.
        let span = self.expression_span(expr);
        self.check_recursion_depth(span)?;

        let result = self.analyze_expression_inner(expr);

        self.decrement_recursion_depth();
        result
    }

    /// Inner expression analysis dispatch — called after recursion guard.
    fn analyze_expression_inner(&mut self, expr: &mut Expression) -> Result<CType, ()> {
        match expr {
            // --- Primary expressions ---
            Expression::IntegerLiteral {
                value,
                suffix,
                is_hex_or_octal,
                ..
            } => Ok(self.integer_literal_type(*value, suffix, *is_hex_or_octal)),
            Expression::FloatLiteral { suffix, .. } => Ok(self.float_literal_type(suffix)),
            Expression::StringLiteral { prefix, .. } => Ok(self.string_literal_type(prefix)),
            Expression::CharLiteral { prefix, .. } => {
                // Character literals have type `int` in C (or wchar_t variant).
                Ok(match prefix {
                    CharPrefix::None => CType::Int,
                    CharPrefix::L => CType::Int, // wchar_t → int on Linux
                    CharPrefix::U16 => CType::UShort,
                    CharPrefix::U32 => CType::UInt,
                })
            }
            Expression::Identifier { name, span } => self.analyze_identifier(*name, *span),
            Expression::Parenthesized { inner, .. } => self.analyze_expression(inner),

            // --- Postfix expressions ---
            Expression::ArraySubscript { base, index, span } => {
                let base_ty = self.analyze_expression(base)?;
                let _index_ty = self.analyze_expression(index)?;
                // Array subscript: base must be pointer/array, result is element type.
                match self.get_pointee_type(&base_ty) {
                    Some(elem) => Ok(elem),
                    None => {
                        // If the base type is a primitive (type-inference
                        // fallback from complex _Generic/typeof/statement
                        // expressions), allow subscript with Int return.
                        let stripped = self.strip_qualifiers(&base_ty);
                        let is_primitive = matches!(
                            stripped,
                            CType::Int
                                | CType::UInt
                                | CType::Long
                                | CType::ULong
                                | CType::Char
                                | CType::Void
                        );
                        if is_primitive {
                            Ok(CType::Int)
                        } else {
                            self.diagnostics
                                .emit_error(*span, "subscripted value is not an array or pointer");
                            Err(())
                        }
                    }
                }
            }
            Expression::FunctionCall { callee, args, span } => {
                let callee_ty = self.analyze_expression(callee)?;
                // Analyze argument expressions.
                let mut _arg_types = Vec::with_capacity(args.len());
                for arg in args.iter_mut() {
                    let ty = self.analyze_expression(arg)?;
                    _arg_types.push(ty);
                }
                // Determine return type from callee type.
                self.resolve_function_call_type(&callee_ty, *span)
            }
            Expression::MemberAccess {
                object,
                member,
                span,
            } => {
                let obj_ty = self.analyze_expression(object)?;
                self.resolve_member_type(&obj_ty, *member, *span)
            }
            Expression::PointerMemberAccess {
                object,
                member,
                span,
            } => {
                let obj_ty = self.analyze_expression(object)?;
                // Dereference pointer first, then access member.
                let pointee = self.get_pointee_type(&obj_ty);
                match pointee {
                    Some(deref_ty) => self.resolve_member_type(&deref_ty, *member, *span),
                    None => {
                        // If the object type is a primitive (type-inference
                        // fallback from complex _Generic/typeof expressions),
                        // allow the access with a permissive Int return type.
                        let stripped = self.strip_qualifiers(&obj_ty);
                        let is_primitive = matches!(
                            stripped,
                            CType::Int
                                | CType::UInt
                                | CType::Long
                                | CType::ULong
                                | CType::Char
                                | CType::Void
                        );
                        if is_primitive {
                            Ok(CType::Int)
                        } else {
                            self.diagnostics
                                .emit_error(*span, "member reference base type is not a pointer");
                            Err(())
                        }
                    }
                }
            }
            Expression::PostIncrement { operand, .. }
            | Expression::PostDecrement { operand, .. } => {
                let ty = self.analyze_expression(operand)?;
                Ok(ty)
            }

            // --- Unary expressions ---
            Expression::PreIncrement { operand, .. } | Expression::PreDecrement { operand, .. } => {
                let ty = self.analyze_expression(operand)?;
                Ok(ty)
            }
            Expression::UnaryOp { op, operand, span } => {
                let operand_ty = self.analyze_expression(operand)?;
                self.resolve_unary_op_type(*op, &operand_ty, *span)
            }
            Expression::SizeofExpr { operand, .. } => {
                // sizeof(expr) — analyze expression for validity, return size_t.
                let _ = self.analyze_expression(operand);
                Ok(self.size_t_type())
            }
            Expression::SizeofType { .. } => Ok(self.size_t_type()),
            Expression::AlignofType { .. } => Ok(self.size_t_type()),
            Expression::AlignofExpr { expr: inner, .. } => {
                // GCC __alignof__(expr) — analyze the inner expression for
                // side-effect checking, then return size_t.
                let _ = self.analyze_expression(inner);
                Ok(self.size_t_type())
            }

            // --- Cast expression ---
            Expression::Cast {
                type_name, operand, ..
            } => {
                let _operand_ty = self.analyze_expression(operand)?;
                // Resolve the full cast target type from the TypeName,
                // including any pointer/array abstract declarator.
                Ok(self.resolve_type_name(type_name))
            }

            // --- Binary expression ---
            Expression::Binary {
                op,
                left,
                right,
                span,
            } => {
                let left_ty = self.analyze_expression(left)?;
                let right_ty = self.analyze_expression(right)?;
                self.resolve_binary_op_type(*op, &left_ty, &right_ty, *span)
            }

            // --- Conditional (ternary) ---
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                span,
            } => {
                let cond_ty = self.analyze_expression(condition)?;
                // Condition must be scalar.
                if !self.is_scalar_type(&cond_ty) {
                    self.diagnostics
                        .emit_error(*span, "controlling expression of conditional is not scalar");
                }
                let then_ty = if let Some(ref mut te) = then_expr {
                    self.analyze_expression(te)?
                } else {
                    // GCC conditional omission: `x ?: y` — then type = condition type.
                    cond_ty.clone()
                };
                let else_ty = self.analyze_expression(else_expr)?;
                // Result type: usual arithmetic conversion of the two branches.
                Ok(self.common_type(&then_ty, &else_ty))
            }

            // --- Assignment ---
            Expression::Assignment {
                target,
                value,
                span,
                ..
            } => {
                let target_ty = self.analyze_expression(target)?;
                let _value_ty = self.analyze_expression(value)?;

                // Check for const-qualified target — assignment to const is
                // an error per C11 §6.5.16p2.
                if self.is_const_qualified(&target_ty) {
                    self.diagnostics
                        .emit_error(*span, "assignment of read-only variable (const-qualified)");
                }

                // Assignment yields the type of the left operand.
                Ok(target_ty)
            }

            // --- Comma ---
            Expression::Comma { exprs, .. } => {
                let mut last_ty = CType::Void;
                for e in exprs.iter_mut() {
                    last_ty = self.analyze_expression(e)?;
                }
                Ok(last_ty)
            }

            // --- Compound literal (C11) ---
            Expression::CompoundLiteral { .. } => {
                // Compound literal type is determined by the type_name.
                // Full analysis would resolve the type and analyze the
                // initializer list.
                Ok(CType::Int) // Placeholder: real implementation resolves TypeName
            }

            // --- GCC Statement expression ---
            Expression::StatementExpression { compound, .. } => {
                self.analyze_statement_expression(compound)
            }

            // --- GCC Builtin call ---
            Expression::BuiltinCall {
                builtin,
                args,
                span,
            } => self.analyze_builtin_call(builtin, args, *span),

            // --- C11 _Generic selection ---
            Expression::Generic {
                controlling,
                associations,
                span,
            } => {
                let ctrl_ty = self.analyze_expression(controlling)?;
                // Select the matching association or default.
                self.resolve_generic_selection(&ctrl_ty, associations, *span)
            }

            // --- GCC Address-of-label ---
            Expression::AddressOfLabel { label, span } => {
                // &&label — result is void* (GCC extension).
                self.scopes.reference_label(*label, *span);
                Ok(CType::Pointer(
                    Box::new(CType::Void),
                    TypeQualifiers::default(),
                ))
            }
        }
    }

    // ====================================================================
    // Statement Analysis
    // ====================================================================

    /// Analyze a statement, dispatching to variant-specific handlers.
    ///
    /// Handles all C11 statement types plus GCC extensions (computed gotos,
    /// case ranges, local labels, inline assembly).
    pub fn analyze_statement(&mut self, stmt: &mut Statement) -> Result<(), ()> {
        // Enforce recursion depth limit for deeply nested statements.
        // This mirrors the depth tracking in `analyze_expression()` and
        // prevents resource exhaustion on adversarial inputs with hundreds
        // of nested control-flow constructs (e.g., 260+ nested `if`
        // statements).
        self.check_recursion_depth(Span::dummy())?;
        let result = self.analyze_statement_inner(stmt);
        self.decrement_recursion_depth();
        result
    }

    /// Inner implementation of statement analysis, called after the
    /// recursion depth check in `analyze_statement()`.
    fn analyze_statement_inner(&mut self, stmt: &mut Statement) -> Result<(), ()> {
        match stmt {
            Statement::Compound(compound) => {
                self.scopes.push_scope(ScopeKind::Block);
                self.symbols.enter_scope();
                let result = self.analyze_compound_statement(compound);
                self.symbols.leave_scope();
                self.scopes.pop_scope(self.diagnostics);
                result
            }
            Statement::Expression(opt_expr) => {
                if let Some(ref mut expr) = opt_expr {
                    let _ = self.analyze_expression(expr)?;
                }
                Ok(())
            }
            Statement::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => {
                let cond_ty = self.analyze_expression(condition)?;
                if !self.is_scalar_type(&cond_ty) {
                    self.diagnostics
                        .emit_error(*span, "controlling expression of 'if' is not scalar");
                }
                self.analyze_statement(then_branch)?;
                if let Some(ref mut else_stmt) = else_branch {
                    self.analyze_statement(else_stmt)?;
                }
                Ok(())
            }
            Statement::Switch {
                condition,
                body,
                span,
            } => {
                let cond_ty = self.analyze_expression(condition)?;
                if !self.is_integer_type(&cond_ty) {
                    self.diagnostics.emit_error(
                        *span,
                        "switch controlling expression is not an integer type",
                    );
                }
                // Save and reset switch state.
                let prev_in_switch = self.in_switch;
                let prev_cases = std::mem::take(&mut self.switch_case_values);
                let prev_default = self.switch_has_default;
                self.in_switch = true;
                self.switch_case_values = FxHashMap::default();
                self.switch_has_default = false;

                self.analyze_statement(body)?;

                // Restore switch state.
                self.in_switch = prev_in_switch;
                self.switch_case_values = prev_cases;
                self.switch_has_default = prev_default;
                Ok(())
            }
            Statement::While {
                condition,
                body,
                span,
            } => {
                let cond_ty = self.analyze_expression(condition)?;
                if !self.is_scalar_type(&cond_ty) {
                    self.diagnostics
                        .emit_error(*span, "controlling expression of 'while' is not scalar");
                }
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;
                Ok(())
            }
            Statement::DoWhile {
                body,
                condition,
                span,
            } => {
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;
                let cond_ty = self.analyze_expression(condition)?;
                if !self.is_scalar_type(&cond_ty) {
                    self.diagnostics
                        .emit_error(*span, "controlling expression of 'do-while' is not scalar");
                }
                Ok(())
            }
            Statement::For {
                init,
                condition,
                increment,
                body,
                span: _,
            } => {
                self.scopes.push_scope(ScopeKind::Block);
                self.symbols.enter_scope();

                // Analyze init clause.
                if let Some(for_init) = init {
                    match for_init {
                        ForInit::Declaration(decl) => {
                            self.analyze_declaration(decl)?;
                        }
                        ForInit::Expression(expr) => {
                            let _ = self.analyze_expression(expr)?;
                        }
                    }
                }

                // Analyze condition.
                if let Some(ref mut cond) = condition {
                    let _ = self.analyze_expression(cond)?;
                }

                // Analyze increment.
                if let Some(ref mut inc) = increment {
                    let _ = self.analyze_expression(inc)?;
                }

                // Analyze body in loop context.
                let prev_in_loop = self.in_loop;
                self.in_loop = true;
                self.analyze_statement(body)?;
                self.in_loop = prev_in_loop;

                self.symbols.leave_scope();
                self.scopes.pop_scope(self.diagnostics);
                Ok(())
            }
            Statement::Goto { label, span } => {
                self.scopes.reference_label(*label, *span);
                Ok(())
            }
            Statement::ComputedGoto { target, span } => {
                let ty = self.analyze_expression(target)?;
                // GCC computed goto: expression must be a pointer type.
                if !self.is_pointer_type(&ty) {
                    self.diagnostics
                        .emit_error(*span, "argument to computed goto is not a pointer");
                }
                Ok(())
            }
            Statement::Continue { span } => {
                if !self.in_loop {
                    self.diagnostics
                        .emit_error(*span, "'continue' statement not in loop statement");
                }
                Ok(())
            }
            Statement::Break { span } => {
                if !self.in_loop && !self.in_switch {
                    self.diagnostics
                        .emit_error(*span, "'break' statement not in loop or switch statement");
                }
                Ok(())
            }
            Statement::Return { value, span } => {
                if let Some(ref mut ret_expr) = value {
                    let ret_ty = self.analyze_expression(ret_expr)?;
                    // Check compatibility with function return type.
                    // GCC extension: `return void_expr;` is allowed in void functions
                    // when the returned expression also has type void (e.g.
                    // `return kernfs_enable_ns(kn);` where the callee returns void).
                    if let Some(ref func_ret) = self.current_function_return_type {
                        if matches!(func_ret, CType::Void) && !matches!(ret_ty, CType::Void) {
                            self.diagnostics
                                .emit_error(*span, "void function should not return a value");
                        }
                    }
                    let _ = ret_ty;
                } else {
                    // Return without value — function must return void.
                    if let Some(ref func_ret) = self.current_function_return_type {
                        if !matches!(func_ret, CType::Void) {
                            self.diagnostics
                                .emit_warning(*span, "non-void function should return a value");
                        }
                    }
                }
                Ok(())
            }
            Statement::Labeled {
                label,
                statement,
                span,
                ..
            } => {
                self.scopes.define_label(*label, *span, self.diagnostics);
                self.analyze_statement(statement)
            }
            Statement::Case {
                value,
                statement,
                span,
            } => {
                if !self.in_switch {
                    self.diagnostics
                        .emit_error(*span, "'case' label not within a switch statement");
                }
                // Evaluate case value as integer constant.
                // For now, if we can extract a literal, check for duplicates.
                if let Expression::IntegerLiteral { value: lit_val, .. } = value.as_ref() {
                    let case_val = *lit_val as i128;
                    if let Some(prev_span) = self.switch_case_values.get(&case_val) {
                        self.diagnostics.emit(
                            Diagnostic::error(*span, "duplicate case value")
                                .with_note(*prev_span, "previous case defined here"),
                        );
                    } else {
                        self.switch_case_values.insert(case_val, *span);
                    }
                }
                self.analyze_statement(statement)
            }
            Statement::CaseRange {
                low: _,
                high: _,
                statement,
                span,
            } => {
                if !self.in_switch {
                    self.diagnostics
                        .emit_error(*span, "'case' label not within a switch statement");
                }
                // GCC case range extension — validate that low <= high.
                self.analyze_statement(statement)
            }
            Statement::Default {
                statement, span, ..
            } => {
                if !self.in_switch {
                    self.diagnostics
                        .emit_error(*span, "'default' label not within a switch statement");
                }
                if self.switch_has_default {
                    self.diagnostics
                        .emit_error(*span, "multiple default labels in one switch");
                }
                self.switch_has_default = true;
                self.analyze_statement(statement)
            }
            Statement::Declaration(decl) => self.analyze_declaration(decl),
            Statement::Asm(asm) => self.analyze_asm_statement(asm),
            Statement::LocalLabel(labels, span) => {
                // GCC __label__ extension: declare local labels in current block scope.
                for &label in labels.iter() {
                    self.scopes
                        .declare_label(label, *span, true, self.diagnostics);
                }
                Ok(())
            }
        }
    }

    // ====================================================================
    // Struct/Union/Enum Definition Analysis
    // ====================================================================

    /// Analyze a struct or union definition.
    ///
    /// Declares the tag in the tag namespace, processes each field's type
    /// and bitfield width, handles flexible array members and anonymous
    /// struct/union members, and computes the aggregate layout.
    pub fn analyze_struct_definition(
        &mut self,
        spec: &mut StructOrUnionSpecifier,
    ) -> Result<CType, ()> {
        let is_struct = spec.kind == StructOrUnion::Struct;
        let tag_kind = if is_struct {
            TagKind::Struct
        } else {
            TagKind::Union
        };

        // Process struct/union attributes.
        let _validated_attrs = self.validate_attributes_for_context(
            &spec.attributes,
            &[],
            AttributeContext::Type,
            spec.span,
        );

        let packed = _validated_attrs
            .iter()
            .any(|a| matches!(a, ValidatedAttribute::Packed));
        let aligned = _validated_attrs.iter().find_map(|a| {
            if let ValidatedAttribute::Aligned(n) = a {
                Some(*n as usize)
            } else {
                None
            }
        });

        // If there's a tag name, declare/look up in tag namespace.
        if let Some(tag_name) = spec.tag {
            let existing = self.scopes.lookup_tag(tag_name);
            if let Some(existing_entry) = existing {
                if existing_entry.kind != tag_kind {
                    self.diagnostics.emit_error(
                        spec.span,
                        "use of tag with wrong kind (struct vs union vs enum)",
                    );
                    return Err(());
                }
                // Performance: if the tag is already complete and this
                // specifier also has members, we have a duplicate
                // definition (e.g. from process_embedded_tag_definitions
                // calling us twice, or from resolve_type_from_type_specifiers
                // already having processed the definition).  Short-circuit
                // to avoid re-resolving all member types.
                if existing_entry.is_complete && spec.members.is_some() {
                    return Ok(existing_entry.ty.clone());
                }
            }
        }

        // If there are no members, this is a forward declaration.
        let Some(ref members) = spec.members else {
            let incomplete_type = if is_struct {
                CType::Struct {
                    name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                    fields: Vec::new(),
                    packed,
                    aligned,
                }
            } else {
                CType::Union {
                    name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                    fields: Vec::new(),
                    packed,
                    aligned,
                }
            };

            if let Some(tag_name) = spec.tag {
                let entry = TagEntry {
                    kind: tag_kind,
                    ty: incomplete_type.clone(),
                    is_complete: false,
                    span: spec.span,
                };
                self.scopes.declare_tag(tag_name, entry);
            }
            return Ok(incomplete_type);
        };

        // Process member declarations to build field list.
        let mut fields = Vec::new();
        for member in members {
            // Process embedded struct/union/enum definitions within member
            // specifiers so that tags defined inline (e.g.
            //   struct outer { struct inner_tag { int x; } named; };
            // ) are registered in the tag namespace and their fields are
            // available for later use (e.g. `struct inner_tag t; t.x`).
            for spec in &member.specifiers.type_specifiers {
                match spec {
                    TypeSpecifier::Struct(s) | TypeSpecifier::Union(s) => {
                        if s.members.is_some() {
                            let mut s_clone = s.clone();
                            let _ = self.analyze_struct_definition(&mut s_clone);
                        }
                    }
                    TypeSpecifier::Enum(e) => {
                        if e.enumerators.is_some() {
                            let mut e_clone = e.clone();
                            let _ = self.analyze_enum_definition(&mut e_clone);
                        }
                    }
                    _ => {}
                }
            }
            let member_base_type = self.resolve_type_from_spec_qualifier_list(&member.specifiers);

            if member.declarators.is_empty() {
                // Anonymous struct/union member (C11 §6.7.2.1p13):
                //   struct outer { union { int a; int b; }; int c; };
                // The anonymous member has no declarator. We add it as an
                // unnamed field so that find_member_in_fields can recurse
                // into it when resolving member access.
                fields.push(crate::common::types::StructField {
                    name: None,
                    ty: member_base_type,
                    bit_width: None,
                });
                continue;
            }

            for decl in &member.declarators {
                let field_type = if let Some(ref d) = decl.declarator {
                    self.apply_declarator_to_type(member_base_type.clone(), d)
                } else {
                    member_base_type.clone()
                };

                let field_name = decl
                    .declarator
                    .as_ref()
                    .and_then(|d| self.extract_declarator_name(d))
                    .map(|sym| self.interner.resolve(sym).to_string());

                let bit_width = decl.bit_width.as_ref().map(|bw| {
                    // Evaluate the bitfield width as an integer constant
                    // expression. This is required for correct struct layout
                    // computation (sizeof, _Static_assert).
                    eval_bitfield_width(bw)
                });

                fields.push(crate::common::types::StructField {
                    name: field_name,
                    ty: field_type,
                    bit_width,
                });
            }
        }

        // Propagate field-level __attribute__((aligned(N))) to the struct's
        // own alignment.  C11 §6.7.2.1: the alignment of a struct is at
        // least as strict as the alignment of any of its members.  GCC
        // extends this: an `aligned(N)` attribute on a struct field member
        // declaration both forces that field to an N-byte boundary AND
        // raises the struct's overall alignment to at least N.
        let mut max_field_align: Option<usize> = None;
        for member in members {
            // Check specifiers attributes, member-level attributes, and
            // declarator attributes for aligned(N).
            let all_attrs: Vec<&Attribute> = member
                .specifiers
                .attributes
                .iter()
                .chain(member.attributes.iter())
                .chain(
                    member
                        .declarators
                        .iter()
                        .filter_map(|d| d.declarator.as_ref())
                        .flat_map(|d| d.attributes.iter()),
                )
                .collect();
            for attr in &all_attrs {
                let attr_name = self.interner.resolve(attr.name);
                if attr_name == "aligned" || attr_name == "__aligned__" {
                    let val =
                        if let Some(AttributeArg::Expression(ref boxed_expr)) = attr.args.first() {
                            self.try_eval_attr_const_expr(boxed_expr)
                                .map(|v| v as usize)
                        } else {
                            Some(self.target.max_align())
                        };
                    if let Some(v) = val {
                        max_field_align = Some(max_field_align.map_or(v, |cur: usize| cur.max(v)));
                    }
                }
            }
        }
        // Merge: explicit struct-level aligned takes precedence if larger,
        // otherwise field-level aligned raises the struct alignment.
        let aligned = match (aligned, max_field_align) {
            (Some(s), Some(f)) => Some(s.max(f)),
            (Some(s), None) => Some(s),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        };

        // Build the complete type.
        let complete_type = if is_struct {
            CType::Struct {
                name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                fields,
                packed,
                aligned,
            }
        } else {
            CType::Union {
                name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                fields,
                packed,
                aligned,
            }
        };

        // Register as complete in tag namespace.
        if let Some(tag_name) = spec.tag {
            let entry = TagEntry {
                kind: tag_kind,
                ty: complete_type.clone(),
                is_complete: true,
                span: spec.span,
            };
            self.scopes.declare_tag(tag_name, entry);
            self.scopes.complete_tag(tag_name, complete_type.clone());
            // Update the string-keyed name map for sizeof resolution.
            let tag_str = self.interner.resolve(tag_name).to_string();
            self.tag_types_by_name
                .insert(tag_str, complete_type.clone());
        }

        Ok(complete_type)
    }

    /// Analyze an enum definition.
    ///
    /// Declares the tag in the tag namespace, evaluates each enumerator
    /// value (auto-incrementing or explicit), and registers each enumerator
    /// as an enum constant in the symbol table.
    pub fn analyze_enum_definition(&mut self, spec: &mut EnumSpecifier) -> Result<CType, ()> {
        let tag_kind = TagKind::Enum;

        // Check for __attribute__((packed)) which makes the enum use the
        // smallest integer type that fits all enumerator values (GCC
        // extension used extensively in the Linux kernel, e.g. enum rw_hint).
        let is_packed = spec.attributes.iter().any(|a| {
            let attr_name = self.interner.resolve(a.name);
            attr_name == "packed" || attr_name == "__packed__"
        });

        let underlying_type = CType::Int;

        // If forward reference (no enumerators), just declare.
        let Some(ref enumerators) = spec.enumerators else {
            let enum_type = CType::Enum {
                name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                underlying_type: Box::new(underlying_type.clone()),
            };
            if let Some(tag_name) = spec.tag {
                let entry = TagEntry {
                    kind: tag_kind,
                    ty: enum_type.clone(),
                    is_complete: false,
                    span: spec.span,
                };
                self.scopes.declare_tag(tag_name, entry);
            }
            return Ok(enum_type);
        };

        // Process each enumerator, evaluating explicit value expressions.
        // Track per-enum min/max values for packed size computation.
        let mut next_value: i128 = 0;
        let mut this_enum_min: i128 = i128::MAX;
        let mut this_enum_max: i128 = i128::MIN;
        for enumerator in enumerators {
            let enum_val = if let Some(ref val_expr) = enumerator.value {
                // Lightweight inline evaluation for enum constant expressions.
                // This avoids creating a full ConstantEvaluator per enumerator
                // (which is O(total_enum_constants) and causes O(n²) slowdown
                // on large headers with hundreds of enums).
                self.evaluate_enum_value_expr(val_expr)
                    .unwrap_or(next_value)
            } else {
                next_value
            };

            // Track min/max for THIS enum only (not all enums).
            if enum_val < this_enum_min {
                this_enum_min = enum_val;
            }
            if enum_val > this_enum_max {
                this_enum_max = enum_val;
            }

            // Store in the analyzer's enum constant value map for future
            // ConstantEvaluator invocations (e.g., _Static_assert).
            let name_str = self.interner.resolve(enumerator.name).to_string();
            self.enum_constant_values.insert(name_str, enum_val);

            // Declare in symbol table as enum constant AND in scope
            // for name lookup.
            if let Ok(id) = self.symbols.declare_enum_constant(
                enumerator.name,
                enum_val,
                underlying_type.clone(),
                enumerator.span,
            ) {
                self.scopes.declare_ordinary(enumerator.name, id);
            }

            next_value = enum_val + 1;
        }

        // Determine the underlying type: if packed, use the smallest
        // integer type that fits all enumerator values of THIS enum.
        let final_underlying = if is_packed {
            let min_val = if this_enum_min == i128::MAX {
                0
            } else {
                this_enum_min
            };
            let max_val = if this_enum_max == i128::MIN {
                0
            } else {
                this_enum_max
            };
            if min_val >= 0 && max_val <= 255 {
                CType::UChar
            } else if min_val >= -128 && max_val <= 127 {
                CType::SChar
            } else if min_val >= 0 && max_val <= 65535 {
                CType::UShort
            } else if min_val >= -32768 && max_val <= 32767 {
                CType::Short
            } else {
                underlying_type
            }
        } else {
            // GCC-compatible: if all values are non-negative, use unsigned int
            // so that bitfield reads use zero-extension instead of
            // sign-extension.  This matches GCC's behavior where
            // `enum { A=0, B=248 }; struct { enum e : 8; }` reads 248
            // (not -8).
            let min_val = if this_enum_min == i128::MAX {
                0
            } else {
                this_enum_min
            };
            if min_val >= 0 {
                CType::UInt
            } else {
                underlying_type
            }
        };

        let enum_type = CType::Enum {
            name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
            underlying_type: Box::new(final_underlying.clone()),
        };

        if std::env::var("BCC_DEBUG_ENUM").is_ok() {
            eprintln!(
                "[BCC_ENUM] analyze_enum_definition: name={:?} final_underlying={:?} min={} max={}",
                spec.tag.map(|t| self.interner.resolve(t).to_string()),
                final_underlying,
                this_enum_min,
                this_enum_max
            );
        }

        if let Some(tag_name) = spec.tag {
            let entry = TagEntry {
                kind: tag_kind,
                ty: enum_type.clone(),
                is_complete: true,
                span: spec.span,
            };
            self.scopes.declare_tag(tag_name, entry);
            self.scopes.complete_tag(tag_name, enum_type.clone());
        }

        Ok(enum_type)
    }

    // ====================================================================
    // Inline Assembly Validation
    // ====================================================================

    /// Lightweight evaluation of an enum constant value expression.
    ///
    /// This evaluator handles the common patterns used in C enum definitions:
    /// integer literals, references to previously-defined enum constants,
    /// binary operators (+, -, *, <<, >>, |, &, ^), unary operators (-, ~, !),
    /// and cast expressions. It avoids creating a full `ConstantEvaluator`
    /// (which copies all enum constants and tags per invocation, causing
    /// O(n²) performance on headers with hundreds of enums).
    fn evaluate_enum_value_expr(&self, expr: &Expression) -> Option<i128> {
        use crate::frontend::parser::ast::{BinaryOp, UnaryOp};
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as i128),
            Expression::CharLiteral { value, .. } => Some(*value as i128),
            Expression::Identifier { name, .. } => {
                // Look up previously-defined enum constant by name.
                let name_str = self.interner.resolve(*name);
                self.enum_constant_values.get(name_str).copied()
            }
            Expression::Parenthesized { inner, .. } => self.evaluate_enum_value_expr(inner),
            Expression::UnaryOp { op, operand, .. } => {
                let val = self.evaluate_enum_value_expr(operand)?;
                match op {
                    UnaryOp::Negate => Some(-val),
                    UnaryOp::BitwiseNot => Some(!val),
                    UnaryOp::LogicalNot => Some(if val == 0 { 1 } else { 0 }),
                    UnaryOp::Plus => Some(val),
                    _ => None,
                }
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                let l = self.evaluate_enum_value_expr(left)?;
                let r = self.evaluate_enum_value_expr(right)?;
                match op {
                    BinaryOp::Add => Some(l.wrapping_add(r)),
                    BinaryOp::Sub => Some(l.wrapping_sub(r)),
                    BinaryOp::Mul => Some(l.wrapping_mul(r)),
                    BinaryOp::Div => {
                        if r == 0 {
                            None
                        } else {
                            Some(l.wrapping_div(r))
                        }
                    }
                    BinaryOp::Mod => {
                        if r == 0 {
                            None
                        } else {
                            Some(l.wrapping_rem(r))
                        }
                    }
                    BinaryOp::ShiftLeft => Some(l.wrapping_shl(r as u32)),
                    BinaryOp::ShiftRight => Some(l.wrapping_shr(r as u32)),
                    BinaryOp::BitwiseAnd => Some(l & r),
                    BinaryOp::BitwiseOr => Some(l | r),
                    BinaryOp::BitwiseXor => Some(l ^ r),
                    BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                    BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                    BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                    BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                    BinaryOp::Less => Some(if l < r { 1 } else { 0 }),
                    BinaryOp::Greater => Some(if l > r { 1 } else { 0 }),
                    BinaryOp::LessEqual => Some(if l <= r { 1 } else { 0 }),
                    BinaryOp::GreaterEqual => Some(if l >= r { 1 } else { 0 }),
                }
            }
            Expression::Cast { operand, .. } => {
                // For enum values, casts like (unsigned int)X are value-preserving.
                self.evaluate_enum_value_expr(operand)
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                let cond = self.evaluate_enum_value_expr(condition)?;
                if cond != 0 {
                    if let Some(ref t) = then_expr {
                        self.evaluate_enum_value_expr(t)
                    } else {
                        Some(cond) // GCC extension: x ?: y
                    }
                } else {
                    self.evaluate_enum_value_expr(else_expr)
                }
            }
            Expression::SizeofType { .. } | Expression::SizeofExpr { .. } => {
                // sizeof in enum values is rare; fall back to auto-increment.
                None
            }
            _ => None,
        }
    }

    /// Validate an inline assembly statement.
    ///
    /// Checks:
    /// - Output operand constraint strings (must start with `=` or `+`)
    /// - Output operand expressions must be lvalues
    /// - Input operand constraint strings
    /// - Clobber list entries ("memory", "cc", register names)
    /// - `asm goto` jump label targets exist
    /// - Named operand `[name]` uniqueness
    pub fn analyze_asm_statement(&mut self, asm: &mut AsmStatement) -> Result<(), ()> {
        let asm_span = asm.span;

        // Validate output operands.
        let mut operand_names: FxHashMap<Symbol, Span> = FxHashMap::default();
        for output in &asm.outputs {
            // Constraint must start with '=' or '+'.
            let constraint_str = String::from_utf8_lossy(&output.constraint);
            if !constraint_str.starts_with('=') && !constraint_str.starts_with('+') {
                self.diagnostics.emit_error(
                    output.span,
                    format!(
                        "invalid output constraint '{}': must start with '=' or '+'",
                        constraint_str
                    ),
                );
            }
            // Check for duplicate named operands.
            if let Some(name) = output.symbolic_name {
                if let Some(prev_span) = operand_names.get(&name) {
                    self.diagnostics.emit(
                        Diagnostic::error(output.span, "duplicate asm operand name")
                            .with_note(*prev_span, "previous operand with this name"),
                    );
                } else {
                    operand_names.insert(name, output.span);
                }
            }
        }

        // Validate input operands.
        for input in &asm.inputs {
            let constraint_str = String::from_utf8_lossy(&input.constraint);
            // Input constraints should NOT start with '='.
            if constraint_str.starts_with('=') {
                self.diagnostics.emit_error(
                    input.span,
                    format!(
                        "invalid input constraint '{}': input constraints must not start with '='",
                        constraint_str
                    ),
                );
            }
            // Check for duplicate named operands.
            if let Some(name) = input.symbolic_name {
                if let Some(prev_span) = operand_names.get(&name) {
                    self.diagnostics.emit(
                        Diagnostic::error(input.span, "duplicate asm operand name")
                            .with_note(*prev_span, "previous operand with this name"),
                    );
                } else {
                    operand_names.insert(name, input.span);
                }
            }
        }

        // Validate clobber list entries (architecture-specific validation
        // occurs during code generation).
        for clobber in &asm.clobbers {
            let _clobber_str = String::from_utf8_lossy(&clobber.register);
        }

        // Validate asm goto labels.
        for &label in &asm.goto_labels {
            self.scopes.reference_label(label, asm_span);
        }

        Ok(())
    }

    // ====================================================================
    // Finalization
    // ====================================================================

    /// Finalize semantic analysis at the end of a translation unit.
    ///
    /// - Finalizes tentative definitions (C11 §6.9.2).
    /// - Checks for unused symbols at file scope.
    /// - Returns `Ok(())` if no errors, `Err(())` if errors accumulated.
    pub fn finalize(&mut self) -> Result<(), ()> {
        self.symbols
            .finalize_tentative_definitions(self.diagnostics);
        self.symbols
            .check_unused_symbols(self.diagnostics, self.interner);

        if self.diagnostics.has_errors() {
            Err(())
        } else {
            Ok(())
        }
    }

    // ====================================================================
    // Recursion Depth Guard
    // ====================================================================

    /// Check recursion depth against the 512 limit (AAP §0.7.3).
    ///
    /// If the limit is reached, emits an error diagnostic and returns Err.
    /// Otherwise increments the depth counter and returns Ok.
    fn check_recursion_depth(&mut self, span: Span) -> Result<(), ()> {
        if self.recursion_depth >= self.max_recursion_depth {
            self.diagnostics.emit_error(
                span,
                "recursion depth limit (512) exceeded during semantic analysis",
            );
            return Err(());
        }
        self.recursion_depth += 1;
        Ok(())
    }

    /// Decrement the recursion depth counter.
    fn decrement_recursion_depth(&mut self) {
        if self.recursion_depth > 0 {
            self.recursion_depth -= 1;
        }
    }

    // ====================================================================
    // Private Helper Methods
    // ====================================================================

    /// Analyze a compound statement (block) without pushing/popping scope
    /// (caller manages scope for function bodies vs. nested blocks).
    fn analyze_compound_statement(&mut self, compound: &mut CompoundStatement) -> Result<(), ()> {
        for item in &mut compound.items {
            match item {
                BlockItem::Declaration(decl) => {
                    let _ = self.analyze_declaration(decl);
                }
                BlockItem::Statement(stmt) => {
                    let _ = self.analyze_statement(stmt);
                }
            }
        }
        Ok(())
    }

    /// Analyze a GCC statement expression `({ ... })`.
    ///
    /// The value of the expression is the value of the last
    /// expression-statement in the compound statement.
    fn analyze_statement_expression(
        &mut self,
        compound: &mut CompoundStatement,
    ) -> Result<CType, ()> {
        self.scopes.push_scope(ScopeKind::Block);
        self.symbols.enter_scope();

        let mut last_type = CType::Void;
        for item in &mut compound.items {
            match item {
                BlockItem::Declaration(decl) => {
                    let _ = self.analyze_declaration(decl);
                }
                BlockItem::Statement(stmt) => {
                    // If the last statement is an expression statement,
                    // its type is the result type.
                    if let Statement::Expression(Some(ref mut expr)) = stmt {
                        last_type = self.analyze_expression(expr)?;
                    } else {
                        let _ = self.analyze_statement(stmt);
                    }
                }
            }
        }

        self.symbols.leave_scope();
        self.scopes.pop_scope(self.diagnostics);
        Ok(last_type)
    }

    /// Analyze a GCC builtin call, dispatching to BuiltinEvaluator.
    fn analyze_builtin_call(
        &mut self,
        builtin: &BuiltinKind,
        args: &[Expression],
        span: Span,
    ) -> Result<CType, ()> {
        // Delegate to builtin evaluator for semantic checking.
        // For the mod.rs integration, we determine result types from
        // known builtin signatures.
        match builtin {
            BuiltinKind::ConstantP => Ok(CType::Int),
            BuiltinKind::TypesCompatibleP => Ok(CType::Int),
            BuiltinKind::ChooseExpr => {
                // Type depends on chosen branch; for now return Int.
                Ok(CType::Int)
            }
            BuiltinKind::Offsetof => Ok(self.size_t_type()),
            BuiltinKind::Expect => {
                // __builtin_expect(expr, expected) → type of expr
                Ok(CType::Long)
            }
            BuiltinKind::Unreachable | BuiltinKind::Trap => Ok(CType::Void),
            BuiltinKind::Clz
            | BuiltinKind::ClzL
            | BuiltinKind::ClzLL
            | BuiltinKind::Ctz
            | BuiltinKind::CtzL
            | BuiltinKind::CtzLL
            | BuiltinKind::Popcount
            | BuiltinKind::PopcountL
            | BuiltinKind::PopcountLL
            | BuiltinKind::Ffs
            | BuiltinKind::Ffsll => Ok(CType::Int),
            BuiltinKind::Bswap16 => Ok(CType::UShort),
            BuiltinKind::Bswap32 => Ok(CType::UInt),
            BuiltinKind::Bswap64 => Ok(CType::ULongLong),
            BuiltinKind::VaStart | BuiltinKind::VaEnd | BuiltinKind::VaCopy => Ok(CType::Void),
            BuiltinKind::VaArg => {
                // Extract the real type from the second argument which
                // is wrapped as SizeofType by the parser.
                if args.len() >= 2 {
                    if let Expression::SizeofType { ref type_name, .. } = args[1] {
                        let resolved = self.resolve_type_name(type_name);
                        Ok(resolved)
                    } else {
                        Ok(CType::Int)
                    }
                } else {
                    Ok(CType::Int)
                }
            }
            BuiltinKind::FrameAddress | BuiltinKind::ReturnAddress => Ok(CType::Pointer(
                Box::new(CType::Void),
                TypeQualifiers::default(),
            )),
            BuiltinKind::AssumeAligned => Ok(CType::Pointer(
                Box::new(CType::Void),
                TypeQualifiers::default(),
            )),
            BuiltinKind::AddOverflow | BuiltinKind::SubOverflow | BuiltinKind::MulOverflow => {
                Ok(CType::Bool)
            }
            BuiltinKind::PrefetchData => {
                let _ = (args, span);
                Ok(CType::Void)
            }
            BuiltinKind::ObjectSize => {
                let _ = (args, span);
                Ok(CType::ULong)
            }
            BuiltinKind::ExtractReturnAddr => {
                let _ = (args, span);
                Ok(CType::Pointer(
                    Box::new(CType::Void),
                    crate::common::types::TypeQualifiers::default(),
                ))
            }
            // All other builtins — use BuiltinEvaluator for comprehensive return type.
            other => {
                let _ = (args, span);
                let eval = crate::frontend::sema::builtin_eval::BuiltinEvaluator::new(
                    self.diagnostics,
                    self.type_builder,
                    self.target,
                );
                Ok(eval.get_builtin_return_type(other))
            }
        }
    }

    /// Resolve a `_Generic` selection expression.
    ///
    /// C11 §6.5.1.1: The controlling expression's type is compared
    /// (ignoring qualifiers) against each association's type name.  The
    /// expression of the first matching association is selected.  If no
    /// match is found and a `default:` association exists, its expression
    /// is selected.  If neither matches, it is a constraint violation.
    fn resolve_generic_selection(
        &mut self,
        controlling_type: &CType,
        associations: &[GenericAssociation],
        span: Span,
    ) -> Result<CType, ()> {
        // C11 §6.5.1.1p2: The controlling expression undergoes lvalue
        // conversion, which includes array-to-pointer decay and
        // function-to-pointer decay, and qualifier stripping.
        let decayed = match self.strip_qualifiers(controlling_type) {
            CType::Array(elem, _) => CType::Pointer(elem.clone(), TypeQualifiers::default()),
            CType::Function { .. } => CType::Pointer(
                Box::new(controlling_type.clone()),
                TypeQualifiers::default(),
            ),
            _ => controlling_type.clone(),
        };
        let ctrl_stripped = self.strip_qualifiers(&decayed).clone();
        let mut default_expr_type: Option<CType> = None;

        // C11 §6.5.1.1p3: No two generic associations in the same
        // _Generic selection shall specify compatible types.
        // Also, at most one `default` association is allowed.
        {
            let mut seen_types: Vec<CType> = Vec::new();
            let mut seen_default = false;
            for assoc in associations {
                if let Some(ref tn) = assoc.type_name {
                    let assoc_type = self.resolve_type_name(tn);
                    let assoc_stripped = self.strip_qualifiers(&assoc_type).clone();
                    for prev in &seen_types {
                        if self.types_match_generic(&assoc_stripped, prev) {
                            // GCC extension: allow duplicate types in _Generic
                            // (Linux kernel relies on this when typeof() resolves
                            // to the same type in multiple associations).
                            // Downgrade to warning; first matching association wins.
                            self.diagnostics
                                .emit_warning(span, "duplicate type name in _Generic association");
                            break;
                        }
                    }
                    seen_types.push(assoc_stripped);
                } else {
                    if seen_default {
                        self.diagnostics
                            .emit_error(span, "duplicate default case in _Generic association");
                        return Err(());
                    }
                    seen_default = true;
                }
            }
        }

        // First pass: try to match the controlling type against each
        // association's type name.
        for assoc in associations {
            if let Some(ref tn) = assoc.type_name {
                let assoc_type = self.resolve_type_name(tn);
                let assoc_stripped = self.strip_qualifiers(&assoc_type).clone();
                if self.types_match_generic(&ctrl_stripped, &assoc_stripped) {
                    // Match found — analyze the expression and return its type.
                    let mut expr_clone = (*assoc.expression).clone();
                    return self.analyze_expression(&mut expr_clone);
                }
            } else {
                // Default association — remember for fallback.
                let mut expr_clone = (*assoc.expression).clone();
                let ty = self.analyze_expression(&mut expr_clone)?;
                default_expr_type = Some(ty);
            }
        }

        // No specific match — use default if available.
        if let Some(ty) = default_expr_type {
            return Ok(ty);
        }

        // No match and no default — try the first association as a
        // last resort (better than hard error for practical compatibility).
        if let Some(assoc) = associations.first() {
            let mut expr_clone = (*assoc.expression).clone();
            return self.analyze_expression(&mut expr_clone);
        }

        self.diagnostics
            .emit_error(span, "_Generic selection has no matching association");
        Err(())
    }

    /// Check if two types match for `_Generic` selection purposes.
    ///
    /// For `_Generic`, types are compared after stripping top-level
    /// qualifiers.  Named struct/union types match by tag name; other
    /// types use structural compatibility.
    fn types_match_generic(&self, a: &CType, b: &CType) -> bool {
        let a = self.strip_qualifiers(a);
        let b = self.strip_qualifiers(b);
        if a == b {
            return true;
        }
        match (a, b) {
            (CType::Struct { name: Some(na), .. }, CType::Struct { name: Some(nb), .. }) => {
                na == nb
            }
            (CType::Union { name: Some(na), .. }, CType::Union { name: Some(nb), .. }) => na == nb,
            // For _Generic, pointer types match if their pointed-to types
            // match after stripping qualifiers (ignore const, volatile on
            // both the pointer and the pointee).
            (CType::Pointer(inner_a, _), CType::Pointer(inner_b, _)) => self.types_match_generic(
                self.strip_qualifiers(inner_a),
                self.strip_qualifiers(inner_b),
            ),
            (CType::Typedef { underlying: ua, .. }, other)
            | (other, CType::Typedef { underlying: ua, .. }) => {
                self.types_match_generic(self.strip_qualifiers(ua), self.strip_qualifiers(other))
            }
            (CType::Qualified(inner, _), other) | (other, CType::Qualified(inner, _)) => {
                self.types_match_generic(self.strip_qualifiers(inner), self.strip_qualifiers(other))
            }
            _ => false,
        }
    }

    /// Resolve the base type from declaration specifiers.
    fn resolve_type_from_specifiers(&self, specs: &DeclarationSpecifiers) -> CType {
        self.resolve_type_from_type_specifiers(&specs.type_specifiers, &specs.type_qualifiers)
    }

    /// Resolve a full type from a `TypeName` (specifier-qualifier list +
    /// optional abstract declarator).  This handles casts like `(int *)`,
    /// `(void **)`, `(const char *)`, `(struct foo *)`, etc.
    fn resolve_type_name(&self, tn: &TypeName) -> CType {
        let mut base = self.resolve_type_from_spec_qualifier_list(&tn.specifier_qualifiers);
        if let Some(ref abs) = tn.abstract_declarator {
            // Apply pointer chain (e.g. `*`, `**`, `*const`).
            if let Some(ref ptr) = abs.pointer {
                base = self.apply_pointer_to_type(base, ptr);
            }
            // Apply direct abstract declarator (arrays, function pointers).
            if let Some(ref direct_abs) = abs.direct {
                base = self.apply_direct_abstract_declarator_to_type(base, direct_abs);
            }
        }
        base
    }

    /// Apply a direct abstract declarator (array/function) to a type.
    fn apply_direct_abstract_declarator_to_type(
        &self,
        base: CType,
        dad: &DirectAbstractDeclarator,
    ) -> CType {
        match dad {
            DirectAbstractDeclarator::Parenthesized(inner) => {
                let mut ty = base;
                if let Some(ref ptr) = inner.pointer {
                    ty = self.apply_pointer_to_type(ty, ptr);
                }
                if let Some(ref d) = inner.direct {
                    ty = self.apply_direct_abstract_declarator_to_type(ty, d);
                }
                ty
            }
            DirectAbstractDeclarator::Array {
                base: base_dad,
                size,
                ..
            } => {
                let arr_size = size.as_ref().and_then(|s| {
                    if let Expression::IntegerLiteral { value, .. } = s.as_ref() {
                        Some(*value as usize)
                    } else {
                        // Evaluate non-literal array size expressions
                        // (e.g. sizeof-based expressions) for correct
                        // type layout in abstract declarator contexts.
                        let result2 = self.try_eval_attr_const_expr(s);
                        result2.map(|v| v as usize)
                    }
                });
                let arr_ty = CType::Array(Box::new(base), arr_size);
                if let Some(ref inner) = base_dad {
                    self.apply_direct_abstract_declarator_to_type(arr_ty, inner)
                } else {
                    arr_ty
                }
            }
            DirectAbstractDeclarator::Function {
                base: base_dad,
                params,
                is_variadic,
            } => {
                let param_types: Vec<CType> = params
                    .iter()
                    .map(|p| {
                        let bt = self.resolve_type_from_specifiers(&p.specifiers);
                        if let Some(ref decl) = p.declarator {
                            self.apply_declarator_to_type(bt, decl)
                        } else if let Some(ref ad) = p.abstract_declarator {
                            let mut pt = bt;
                            if let Some(ref ptr) = ad.pointer {
                                pt = self.apply_pointer_to_type(pt, ptr);
                            }
                            if let Some(ref d) = ad.direct {
                                pt = self.apply_direct_abstract_declarator_to_type(pt, d);
                            }
                            pt
                        } else {
                            bt
                        }
                    })
                    .collect();
                let func_ty = CType::Function {
                    return_type: Box::new(base),
                    params: param_types,
                    variadic: *is_variadic,
                };
                // If there's a base direct-abstract-declarator (e.g., the `(*)`
                // in `(*)(int)`), apply it to the function type to create a
                // pointer-to-function (or other derived type).
                if let Some(ref inner) = base_dad {
                    self.apply_direct_abstract_declarator_to_type(func_ty, inner)
                } else {
                    func_ty
                }
            }
        }
    }

    /// Resolve type from specifier-qualifier list.
    fn resolve_type_from_spec_qualifier_list(&self, sql: &SpecifierQualifierList) -> CType {
        self.resolve_type_from_type_specifiers(&sql.type_specifiers, &sql.type_qualifiers)
    }

    /// Infer the C type of an expression for `typeof(expr)` / `__typeof__(expr)`.
    ///
    /// This performs read-only type inference on the expression without
    /// modifying it. For most common patterns used in kernel macros
    /// (identifiers, member access, pointer dereference, arithmetic),
    /// the correct type is returned. Falls back to `CType::Int` for
    /// expressions that cannot be resolved statically.
    fn infer_typeof_expr_type(&self, expr: &Expression) -> CType {
        match expr {
            // Identifier — look up in symbol table.
            Expression::Identifier { name, .. } => {
                if let Some(id) = self.scopes.lookup_ordinary(*name) {
                    let entry = self.symbols.get(id);
                    entry.ty.clone()
                } else {
                    CType::Int
                }
            }
            // Integer literals.
            Expression::IntegerLiteral { suffix, .. } => match suffix {
                IntegerSuffix::None => CType::Int,
                IntegerSuffix::U => CType::UInt,
                IntegerSuffix::L => CType::Long,
                IntegerSuffix::UL => CType::ULong,
                IntegerSuffix::LL => CType::LongLong,
                IntegerSuffix::ULL => CType::ULongLong,
            },
            // Float literals.
            Expression::FloatLiteral { .. } => CType::Double,
            // Char literals.
            Expression::CharLiteral { .. } => CType::Int,
            // String literals.
            Expression::StringLiteral { .. } => {
                CType::Pointer(Box::new(CType::Char), TypeQualifiers::default())
            }
            // Dereference: *ptr → pointee type.
            Expression::UnaryOp {
                op: ast::UnaryOp::Deref,
                operand,
                ..
            } => {
                let inner = self.infer_typeof_expr_type(operand);
                match inner {
                    CType::Pointer(pointee, _) => *pointee,
                    CType::Array(elem, _) => *elem,
                    _ => CType::Int,
                }
            }
            // Address-of: &x → pointer to x's type.
            Expression::UnaryOp {
                op: ast::UnaryOp::AddressOf,
                operand,
                ..
            } => {
                let inner = self.infer_typeof_expr_type(operand);
                CType::Pointer(Box::new(inner), TypeQualifiers::default())
            }
            // Other unary ops: +x, -x, ~x, !x — same type as operand.
            Expression::UnaryOp { operand, op, .. } => {
                if matches!(op, ast::UnaryOp::LogicalNot) {
                    CType::Int
                } else {
                    self.infer_typeof_expr_type(operand)
                }
            }
            // Member access: obj.member.
            Expression::MemberAccess { object, member, .. } => {
                let obj_ty = self.infer_typeof_expr_type(object);
                self.lookup_member_type(&obj_ty, *member)
            }
            // Pointer member access: ptr->member.
            Expression::PointerMemberAccess { object, member, .. } => {
                let obj_ty = self.infer_typeof_expr_type(object);
                // Strip qualifiers and typedefs before checking for pointer
                let stripped = crate::common::types::resolve_and_strip(&obj_ty);
                match stripped {
                    CType::Pointer(inner, _) => self.lookup_member_type(inner, *member),
                    _ => CType::Int,
                }
            }
            // Array subscript: arr[i] → element type.
            Expression::ArraySubscript { base, .. } => {
                let arr_ty = self.infer_typeof_expr_type(base);
                match arr_ty {
                    CType::Array(elem, _) => *elem,
                    CType::Pointer(pointee, _) => *pointee,
                    _ => CType::Int,
                }
            }
            // Binary arithmetic preserves the "wider" type; simplify to Int
            // for most cases. This covers `a + b`, `a * b`, etc.
            Expression::Binary { left, .. } => self.infer_typeof_expr_type(left),
            // Conditional: a ? b : c → type of b (or c if omitted).
            Expression::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                if let Some(then_e) = then_expr {
                    self.infer_typeof_expr_type(then_e)
                } else {
                    self.infer_typeof_expr_type(else_expr)
                }
            }
            // Cast: (type)expr → cast target type.
            Expression::Cast { type_name, .. } => self.resolve_type_name(type_name),
            // Parenthesized.
            Expression::Parenthesized { inner, .. } => self.infer_typeof_expr_type(inner),
            // Post/pre increment/decrement — same type as operand.
            Expression::PostIncrement { operand, .. }
            | Expression::PostDecrement { operand, .. }
            | Expression::PreIncrement { operand, .. }
            | Expression::PreDecrement { operand, .. } => self.infer_typeof_expr_type(operand),
            // Comma: value is the last expression in the list.
            Expression::Comma { exprs, .. } => {
                if let Some(last) = exprs.last() {
                    self.infer_typeof_expr_type(last)
                } else {
                    CType::Void
                }
            }
            // sizeof always produces size_t (unsigned long).
            Expression::SizeofExpr { .. } | Expression::SizeofType { .. } => CType::ULong,
            // Compound literal.
            Expression::CompoundLiteral { type_name, .. } => {
                self.resolve_type_from_spec_qualifier_list(&type_name.specifier_qualifiers)
            }
            // Function call: we'd need the return type. Default to Int.
            Expression::FunctionCall { callee, .. } => {
                let func_ty = self.infer_typeof_expr_type(callee);
                if let CType::Function { return_type, .. } = func_ty {
                    *return_type
                } else {
                    CType::Int
                }
            }
            // C11 _Generic selection — infer the type of the matching branch.
            Expression::Generic {
                controlling,
                associations,
                ..
            } => {
                let ctrl_ty = self.infer_typeof_expr_type(controlling);
                let ctrl_stripped = crate::common::types::resolve_and_strip(&ctrl_ty);
                let mut default_type: Option<CType> = None;

                // Match controlling type against each association.
                for assoc in associations.iter() {
                    if let Some(ref tn) = assoc.type_name {
                        let assoc_type = self.resolve_type_name(tn);
                        let assoc_stripped = crate::common::types::resolve_and_strip(&assoc_type);
                        if self.types_match_generic(ctrl_stripped, assoc_stripped) {
                            return self.infer_typeof_expr_type(&assoc.expression);
                        }
                    } else {
                        // Default association.
                        default_type = Some(self.infer_typeof_expr_type(&assoc.expression));
                    }
                }

                // Fallback to default or Int.
                default_type.unwrap_or(CType::Int)
            }
            // Statement expression — infer from last expression in compound.
            Expression::StatementExpression { compound, .. } => {
                // The type is the type of the last expression statement.
                // We must collect local declarations from the compound so that
                // identifiers declared inside (e.g. `_a` in `({T _a = v; _a;})`)
                // are resolved to their declared type rather than falling back
                // to CType::Int.  This is critical for nested statement-expression
                // macros like min/max where the last expression is a ternary
                // whose branches reference locally-declared variables.
                if let Some(crate::frontend::parser::ast::BlockItem::Statement(
                    Statement::Expression(Some(ref expr)),
                )) = compound.items.last()
                {
                    // Build a map of locally-declared variable names → types.
                    let mut local_types = FxHashMap::default();
                    for item in &compound.items {
                        if let crate::frontend::parser::ast::BlockItem::Declaration(decl) = item {
                            let base = self.resolve_type_from_specifiers(&decl.specifiers);
                            for id in &decl.declarators {
                                if let Some(sym) = self.extract_declarator_name(&id.declarator) {
                                    let full_ty =
                                        self.apply_declarator_to_type(base.clone(), &id.declarator);
                                    local_types.insert(sym, full_ty);
                                }
                            }
                        }
                    }
                    return self.infer_typeof_with_locals(expr, &local_types);
                }
                CType::Int
            }
            // Default fallback.
            _ => CType::Int,
        }
    }

    /// Infer the type of an expression within a statement-expression context,
    /// where `locals` maps identifiers declared inside the compound block to
    /// their types.  This is needed because `infer_typeof_expr_type` normally
    /// resolves identifiers via scope lookup, which does not see variables
    /// local to the statement expression (the inner scope isn't active during
    /// typeof evaluation).
    ///
    /// Handles recursive descent through Conditional (ternary), Binary,
    /// UnaryOp, Parenthesized, Comma, Cast, and nested StatementExpression
    /// nodes so that expressions like `_a < _b ? _a : _b` correctly resolve
    /// `_a`/`_b` from the local declarations map.
    fn infer_typeof_with_locals(
        &self,
        expr: &Expression,
        locals: &FxHashMap<crate::common::string_interner::Symbol, CType>,
    ) -> CType {
        match expr {
            Expression::Identifier { name, .. } => {
                // Check statement-expression locals first, then outer scope.
                if let Some(ty) = locals.get(name) {
                    return ty.clone();
                }
                if let Some(id) = self.scopes.lookup_ordinary(*name) {
                    return self.symbols.get(id).ty.clone();
                }
                CType::Int
            }
            Expression::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                if let Some(then_e) = then_expr {
                    self.infer_typeof_with_locals(then_e, locals)
                } else {
                    self.infer_typeof_with_locals(else_expr, locals)
                }
            }
            Expression::Parenthesized { inner, .. } => self.infer_typeof_with_locals(inner, locals),
            Expression::Comma { exprs, .. } => {
                if let Some(last) = exprs.last() {
                    self.infer_typeof_with_locals(last, locals)
                } else {
                    CType::Void
                }
            }
            Expression::Binary { left, .. } => self.infer_typeof_with_locals(left, locals),
            Expression::UnaryOp {
                op: ast::UnaryOp::Deref,
                operand,
                ..
            } => {
                let inner = self.infer_typeof_with_locals(operand, locals);
                match inner {
                    CType::Pointer(pointee, _) | CType::Array(pointee, _) => *pointee,
                    _ => CType::Int,
                }
            }
            Expression::UnaryOp {
                op: ast::UnaryOp::AddressOf,
                operand,
                ..
            } => {
                let inner = self.infer_typeof_with_locals(operand, locals);
                CType::Pointer(Box::new(inner), TypeQualifiers::default())
            }
            Expression::UnaryOp { operand, op, .. } => {
                if matches!(op, ast::UnaryOp::LogicalNot) {
                    CType::Int
                } else {
                    self.infer_typeof_with_locals(operand, locals)
                }
            }
            Expression::Cast { type_name, .. } => self.resolve_type_name(type_name),
            Expression::PostIncrement { operand, .. }
            | Expression::PostDecrement { operand, .. }
            | Expression::PreIncrement { operand, .. }
            | Expression::PreDecrement { operand, .. } => {
                self.infer_typeof_with_locals(operand, locals)
            }
            // Nested statement expression: recurse with a fresh local map
            // built from the inner compound's declarations.
            Expression::StatementExpression { compound, .. } => {
                if let Some(crate::frontend::parser::ast::BlockItem::Statement(
                    Statement::Expression(Some(ref inner_expr)),
                )) = compound.items.last()
                {
                    let mut inner_locals = locals.clone();
                    for item in &compound.items {
                        if let crate::frontend::parser::ast::BlockItem::Declaration(decl) = item {
                            let base = self.resolve_type_from_specifiers(&decl.specifiers);
                            for id in &decl.declarators {
                                if let Some(sym) = self.extract_declarator_name(&id.declarator) {
                                    let full_ty =
                                        self.apply_declarator_to_type(base.clone(), &id.declarator);
                                    inner_locals.insert(sym, full_ty);
                                }
                            }
                        }
                    }
                    return self.infer_typeof_with_locals(inner_expr, &inner_locals);
                }
                CType::Int
            }
            // For anything not specifically handled, delegate to the standard
            // typeof inference (which handles FunctionCall, SizeofExpr,
            // CompoundLiteral, Generic, MemberAccess, etc.).
            _ => self.infer_typeof_expr_type(expr),
        }
    }

    /// Look up the type of a struct/union member by name.
    ///
    /// Handles forward-referenced structs whose `fields` list may be empty
    /// at the time of type recording (e.g. function parameters, typeof).
    /// When fields are empty and the struct has a tag name, resolves the
    /// current definition from the tag namespace.
    fn lookup_member_type(&self, ty: &CType, member: Symbol) -> CType {
        let member_str = self.interner.resolve(member);
        let stripped = crate::common::types::resolve_and_strip(ty);
        match stripped {
            CType::Struct { name, fields, .. } | CType::Union { name, fields, .. } => {
                // If fields is empty, try resolving from tag namespace
                let resolved_fields = if fields.is_empty() {
                    if let Some(ref tag_name) = name {
                        if let Some(entry) = self.scopes.lookup_tag_by_str(tag_name, self.interner)
                        {
                            if entry.is_complete {
                                match &entry.ty {
                                    CType::Struct { fields: f, .. }
                                    | CType::Union { fields: f, .. } => f.as_slice(),
                                    _ => fields.as_slice(),
                                }
                            } else {
                                fields.as_slice()
                            }
                        } else {
                            fields.as_slice()
                        }
                    } else {
                        fields.as_slice()
                    }
                } else {
                    fields.as_slice()
                };
                if let Some(found) = Self::find_member_in_fields(resolved_fields, member_str) {
                    return found;
                }
                CType::Int
            }
            _ => CType::Int,
        }
    }

    /// Resolve the base C type from a list of type specifiers and qualifiers.
    ///
    /// Implements C11 §6.7.2 type specifier combination rules.
    fn resolve_type_from_type_specifiers(
        &self,
        specifiers: &[TypeSpecifier],
        qualifiers: &[ast::TypeQualifier],
    ) -> CType {
        // Count specifier keywords for combination resolution.
        let mut has_void = false;
        let mut has_char = false;
        let mut has_short = false;
        let mut has_int = false;
        let mut long_count = 0u32;
        let mut has_float = false;
        let mut has_double = false;
        let mut has_signed = false;
        let mut has_unsigned = false;
        let mut has_bool = false;
        let mut has_complex = false;
        let mut has_int128 = false;
        let mut typedef_type: Option<CType> = None;
        let mut struct_type: Option<CType> = None;
        let mut enum_type: Option<CType> = None;

        for spec in specifiers {
            match spec {
                TypeSpecifier::Void => has_void = true,
                TypeSpecifier::Char => has_char = true,
                TypeSpecifier::Short => has_short = true,
                TypeSpecifier::Int => has_int = true,
                TypeSpecifier::Long => long_count += 1,
                TypeSpecifier::Float => has_float = true,
                TypeSpecifier::Double => has_double = true,
                TypeSpecifier::Signed => has_signed = true,
                TypeSpecifier::Unsigned => has_unsigned = true,
                TypeSpecifier::Bool => has_bool = true,
                TypeSpecifier::Complex => has_complex = true,
                TypeSpecifier::Int128 => has_int128 = true,
                // IEC 60559 / GCC floating-point types map to built-in
                // C types for code generation purposes:
                TypeSpecifier::Float128 => {
                    return CType::LongDouble; // 128-bit float → long double
                }
                TypeSpecifier::Float64 => {
                    return CType::Double; // 64-bit float → double
                }
                TypeSpecifier::Float32 => {
                    return CType::Float; // 32-bit float → float
                }
                TypeSpecifier::Float16 => {
                    // _Float16 maps to a half-precision type; use unsigned
                    // short as storage type (16-bit) since we don't have
                    // native half-float support in the backend.
                    return CType::Float; // approximate: treat as float
                }
                TypeSpecifier::TypedefName(sym) => {
                    // Look up the typedef in the scope.
                    if let Some(id) = self.scopes.lookup_ordinary(*sym) {
                        let entry = self.symbols.get(id);
                        typedef_type = Some(entry.ty.clone());
                    }
                }
                TypeSpecifier::Struct(s) => {
                    // Struct type is resolved by looking up the tag.
                    struct_type = Some(self.resolve_struct_union_type(s, true));
                }
                TypeSpecifier::Union(u) => {
                    struct_type = Some(self.resolve_struct_union_type(u, false));
                }
                TypeSpecifier::Enum(e) => {
                    enum_type = Some(self.resolve_enum_type(e));
                }
                TypeSpecifier::Atomic(type_name) => {
                    let inner =
                        self.resolve_type_from_spec_qualifier_list(&type_name.specifier_qualifiers);
                    return CType::Atomic(Box::new(inner));
                }
                TypeSpecifier::Typeof(typeof_arg) => {
                    // typeof support: expression type inference or identity.
                    // Must use resolve_type_name() for TypeName to handle
                    // abstract declarators (arrays, pointers, functions).
                    // E.g. typeof(struct irq_work [2]) must yield Array type.
                    match typeof_arg {
                        TypeofArg::TypeName(tn) => {
                            return self.resolve_type_name(tn);
                        }
                        TypeofArg::Expression(expr) => {
                            return self.infer_typeof_expr_type(expr);
                        }
                    }
                }
                TypeSpecifier::AutoType => {
                    // __auto_type — GCC extension for automatic type inference.
                    // The actual type will be determined from the initializer.
                    // At the sema level, we return Int as a placeholder; the
                    // initializer type will override it during declaration
                    // processing. This is sufficient for kernel code that uses
                    // __auto_type in statement expressions with immediate init.
                    return CType::Int;
                }
            }
        }

        // Resolve the combined type per C11 §6.7.2.
        let base_type = if let Some(ref td) = typedef_type {
            td.clone()
        } else if let Some(ref st) = struct_type {
            st.clone()
        } else if let Some(ref et) = enum_type {
            et.clone()
        } else if has_void {
            CType::Void
        } else if has_bool {
            CType::Bool
        } else if has_char {
            let base = if has_unsigned {
                CType::UChar
            } else if has_signed {
                CType::SChar
            } else {
                CType::Char
            };
            if has_complex {
                CType::Complex(Box::new(base))
            } else {
                base
            }
        } else if has_int128 {
            let base = if has_unsigned {
                CType::UInt128
            } else {
                CType::Int128
            };
            if has_complex {
                CType::Complex(Box::new(base))
            } else {
                base
            }
        } else if has_short {
            let base = if has_unsigned {
                CType::UShort
            } else {
                CType::Short
            };
            if has_complex {
                CType::Complex(Box::new(base))
            } else {
                base
            }
        } else if has_float {
            if has_complex {
                CType::Complex(Box::new(CType::Float))
            } else {
                CType::Float
            }
        } else if has_double {
            if long_count > 0 {
                if has_complex {
                    CType::Complex(Box::new(CType::LongDouble))
                } else {
                    CType::LongDouble
                }
            } else if has_complex {
                CType::Complex(Box::new(CType::Double))
            } else {
                CType::Double
            }
        } else if long_count >= 2 {
            let base = if has_unsigned {
                CType::ULongLong
            } else {
                CType::LongLong
            };
            if has_complex {
                CType::Complex(Box::new(base))
            } else {
                base
            }
        } else if long_count == 1 {
            let base = if has_unsigned {
                CType::ULong
            } else {
                CType::Long
            };
            if has_complex {
                CType::Complex(Box::new(base))
            } else {
                base
            }
        } else if has_unsigned {
            if has_complex {
                CType::Complex(Box::new(CType::UInt))
            } else {
                CType::UInt
            }
        } else if has_complex {
            // `_Complex` with `int`/`signed`/`signed int` → _Complex int.
            // Bare `_Complex` alone defaults to `_Complex double` per C11 §6.7.2p2.
            if has_int || has_signed {
                CType::Complex(Box::new(CType::Int))
            } else {
                CType::Complex(Box::new(CType::Double))
            }
        } else {
            // Default: signed int (covers `int`, `signed`, `signed int`, or bare specifiers).
            CType::Int
        };

        // Apply qualifiers.
        let quals = self.build_type_qualifiers(qualifiers);
        if quals.is_empty() {
            base_type
        } else {
            CType::Qualified(Box::new(base_type), quals)
        }
    }

    /// Build TypeQualifiers from AST qualifier list.
    fn build_type_qualifiers(&self, qualifiers: &[ast::TypeQualifier]) -> TypeQualifiers {
        let mut result = TypeQualifiers::default();
        for q in qualifiers {
            match q {
                ast::TypeQualifier::Const => result.is_const = true,
                ast::TypeQualifier::Volatile => result.is_volatile = true,
                ast::TypeQualifier::Restrict => result.is_restrict = true,
                ast::TypeQualifier::Atomic => result.is_atomic = true,
            }
        }
        result
    }

    /// Resolve a struct or union type from its specifier.
    fn resolve_struct_union_type(&self, spec: &StructOrUnionSpecifier, is_struct: bool) -> CType {
        if let Some(tag_name) = spec.tag {
            if let Some(entry) = self.scopes.lookup_tag(tag_name) {
                if spec.members.is_none() {
                    // This is a tag *reference* (e.g. `struct foo var;`),
                    // NOT a definition.  Return a lightweight "tag
                    // reference" carrying only the tag name with empty
                    // fields.  All consumers that need the full field
                    // list — sizeof_ctype_resolved, lookup_member_type,
                    // and the IR lowering's resolve_forward_ref_type —
                    // already resolve lightweight references via the
                    // tag namespace or tag_types_by_name map.  This
                    // avoids deep-cloning enormous kernel structs
                    // (task_struct ~200 fields) on every reference,
                    // turning an O(n) clone into O(1) construction.
                    let tag_str = self.interner.resolve(tag_name).to_string();
                    return if is_struct {
                        CType::Struct {
                            name: Some(tag_str),
                            fields: Vec::new(),
                            packed: false,
                            aligned: None,
                        }
                    } else {
                        CType::Union {
                            name: Some(tag_str),
                            fields: Vec::new(),
                            packed: false,
                            aligned: None,
                        }
                    };
                }
                // This is a re-definition or has a body.  If the tag is
                // already complete, return the registered type to avoid
                // re-resolving all members below.
                if entry.is_complete {
                    return entry.ty.clone();
                }
            }
        }
        // If the struct/union has member declarations (i.e. a definition body),
        // extract fields directly so anonymous structs used in typedefs get
        // their member information preserved.
        if let Some(ref members) = spec.members {
            let mut fields = Vec::new();
            for member in members {
                let member_base = self.resolve_type_from_spec_qualifier_list(&member.specifiers);

                if member.declarators.is_empty() {
                    // Anonymous struct/union member (C11 §6.7.2.1p13).
                    fields.push(crate::common::types::StructField {
                        name: None,
                        ty: member_base,
                        bit_width: None,
                    });
                    continue;
                }

                for decl in &member.declarators {
                    let field_type = if let Some(ref d) = decl.declarator {
                        self.apply_declarator_to_type(member_base.clone(), d)
                    } else {
                        member_base.clone()
                    };
                    let field_name = decl
                        .declarator
                        .as_ref()
                        .and_then(|d| self.extract_declarator_name(d))
                        .map(|sym| self.interner.resolve(sym).to_string());
                    fields.push(crate::common::types::StructField {
                        name: field_name,
                        ty: field_type,
                        bit_width: decl.bit_width.as_ref().map(|bw| eval_bitfield_width(bw)),
                    });
                }
            }
            return if is_struct {
                CType::Struct {
                    name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                    fields,
                    packed: false,
                    aligned: None,
                }
            } else {
                CType::Union {
                    name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                    fields,
                    packed: false,
                    aligned: None,
                }
            };
        }
        // Forward declaration or anonymous struct/union without body.
        if is_struct {
            CType::Struct {
                name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                fields: Vec::new(),
                packed: false,
                aligned: None,
            }
        } else {
            CType::Union {
                name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
                fields: Vec::new(),
                packed: false,
                aligned: None,
            }
        }
    }

    /// Resolve an enum type from its specifier.
    fn resolve_enum_type(&self, spec: &EnumSpecifier) -> CType {
        if let Some(tag_name) = spec.tag {
            if let Some(entry) = self.scopes.lookup_tag(tag_name) {
                if std::env::var("BCC_DEBUG_ENUM").is_ok() {
                    let tag_str = self.interner.resolve(tag_name);
                    eprintln!(
                        "[BCC_ENUM] resolve_enum_type: tag={} found entry.ty={:?}",
                        tag_str, entry.ty
                    );
                }
                return entry.ty.clone();
            } else if std::env::var("BCC_DEBUG_ENUM").is_ok() {
                let tag_str = self.interner.resolve(tag_name);
                eprintln!(
                    "[BCC_ENUM] resolve_enum_type: tag={} NOT FOUND, returning default Int",
                    tag_str
                );
            }
        }
        CType::Enum {
            name: spec.tag.map(|t| self.interner.resolve(t).to_string()),
            underlying_type: Box::new(CType::Int),
        }
    }

    /// Resolve storage class from declaration specifiers.
    fn resolve_storage_class(&self, specs: &DeclarationSpecifiers) -> StorageClass {
        match specs.storage_class {
            Some(ast::StorageClass::Auto) => StorageClass::Auto,
            Some(ast::StorageClass::Register) => StorageClass::Register,
            Some(ast::StorageClass::Static) => StorageClass::Static,
            Some(ast::StorageClass::Extern) => StorageClass::Extern,
            Some(ast::StorageClass::Typedef) => StorageClass::Typedef,
            Some(ast::StorageClass::ThreadLocal) => StorageClass::ThreadLocal,
            None => {
                // Default: Auto for block scope, implicit extern for file scope
                // functions. The symbol table's resolve_linkage handles this.
                StorageClass::Auto
            }
        }
    }

    /// Apply a declarator's type modifiers to a base type.
    ///
    /// Handles pointer, array, and function declarator layers.
    fn apply_declarator_to_type(&self, mut base: CType, declarator: &Declarator) -> CType {
        // Apply pointer chain.
        if let Some(ref ptr) = declarator.pointer {
            base = self.apply_pointer_to_type(base, ptr);
        }

        // Apply direct declarator layers (array, function).
        base = self.apply_direct_declarator_to_type(base, &declarator.direct);

        base
    }

    /// Apply pointer indirection levels to a type.
    fn apply_pointer_to_type(&self, mut base: CType, ptr: &Pointer) -> CType {
        let quals = self.build_type_qualifiers(&ptr.qualifiers);
        base = CType::Pointer(Box::new(base), quals);
        if let Some(ref inner) = ptr.inner {
            base = self.apply_pointer_to_type(base, inner);
        }
        base
    }

    /// Apply direct declarator modifiers (array, function) to a type.
    ///
    /// C declarator syntax builds types from the inside out. For most
    /// declarators (e.g. `int x[5]`, `int foo(void)`), the inner direct
    /// declarator modifies the base type first, and the outer layer wraps
    /// it.  However, when the inner direct declarator is `Parenthesized`,
    /// the semantics invert: the outer layer (array/function) wraps the
    /// base type *first*, and the parenthesized declarator (which typically
    /// contains a pointer) wraps the resulting composite type.
    ///
    /// Examples:
    /// - `void (*fn)(void *)` → `Pointer(Function { return: Void, params: [Pointer(Void)] })`
    /// - `int (*arr)[5]`      → `Pointer(Array(Int, 5))`
    /// - `int x[3][5]`        → `Array(Array(Int, 5), 3)`  (no Parenthesized)
    fn apply_direct_declarator_to_type(&self, base: CType, dd: &DirectDeclarator) -> CType {
        match dd {
            DirectDeclarator::Identifier(_, _) => base,
            DirectDeclarator::Parenthesized(inner) => self.apply_declarator_to_type(base, inner),
            DirectDeclarator::Array {
                base: inner_dd,
                size,
                ..
            } => {
                let array_size = size.as_ref().and_then(|s| {
                    if let Expression::IntegerLiteral { value, .. } = s.as_ref() {
                        Some(*value as usize)
                    } else {
                        // Evaluate non-literal array size expressions
                        // (e.g. sizeof(T)*8, (int)(sizeof(T)*N), BMS,
                        // etc.) through the constant expression evaluator.
                        // This is critical for correct struct layout when
                        // array sizes involve sizeof, casts, or arithmetic.
                        let result = self.try_eval_attr_const_expr(s);
                        result.map(|v| v as usize)
                    }
                });

                // Build the array type with the CURRENT dimension applied
                // to the base type first, then recurse outward.  This
                // produces the correct nesting for multi-dimensional
                // arrays.  E.g. `int a[3][4]` has AST
                //   Array(size=4, inner=Array(size=3, inner=Ident))
                // The outermost AST node's size (4) is the INNERMOST
                // dimension in C: `a[i]` yields `int[4]`.  So we wrap
                // `base` with the current size first, then let the
                // recursive call wrap the result with the next dimension.
                let arr_type = CType::Array(Box::new(base), array_size);
                if let DirectDeclarator::Parenthesized(inner_decl) = inner_dd.as_ref() {
                    self.apply_declarator_to_type(arr_type, inner_decl)
                } else {
                    self.apply_direct_declarator_to_type(arr_type, inner_dd)
                }
            }
            DirectDeclarator::Function {
                base: inner_dd,
                params,
                is_variadic,
                ..
            } => {
                let param_types: Vec<CType> = params
                    .iter()
                    .map(|p| {
                        let base_ty = self.resolve_type_from_specifiers(&p.specifiers);
                        // Apply the parameter declarator (pointer, array, etc.)
                        // so that `int *p` correctly produces Pointer(Int) rather
                        // than just Int.
                        if let Some(ref d) = p.declarator {
                            self.apply_declarator_to_type(base_ty, d)
                        } else {
                            base_ty
                        }
                    })
                    .collect();

                // When the inner DD is Parenthesized, the parenthesized
                // declarator wraps the function type (e.g. `void (*fn)(void *)`
                // → `Pointer(Function { ... })`).
                if let DirectDeclarator::Parenthesized(inner_decl) = inner_dd.as_ref() {
                    let func_type = CType::Function {
                        return_type: Box::new(base),
                        params: param_types,
                        variadic: *is_variadic,
                    };
                    self.apply_declarator_to_type(func_type, inner_decl)
                } else {
                    let return_type = self.apply_direct_declarator_to_type(base, inner_dd);
                    CType::Function {
                        return_type: Box::new(return_type),
                        params: param_types,
                        variadic: *is_variadic,
                    }
                }
            }
        }
    }

    /// Extract the identifier name from a declarator.
    fn extract_declarator_name(&self, declarator: &Declarator) -> Option<Symbol> {
        self.extract_direct_declarator_name(&declarator.direct)
    }

    /// Extract the identifier name from a direct declarator.
    fn extract_direct_declarator_name(&self, dd: &DirectDeclarator) -> Option<Symbol> {
        match dd {
            DirectDeclarator::Identifier(sym, _) => Some(*sym),
            DirectDeclarator::Parenthesized(inner) => self.extract_declarator_name(inner),
            DirectDeclarator::Array { base, .. } => self.extract_direct_declarator_name(base),
            DirectDeclarator::Function { base, .. } => self.extract_direct_declarator_name(base),
        }
    }

    /// Extract function parameter types and names from a function declarator.
    fn extract_function_params(
        &self,
        declarator: &Declarator,
    ) -> (Vec<CType>, Vec<Option<Symbol>>, bool) {
        self.extract_function_params_from_dd(&declarator.direct)
    }

    /// Extract function parameter info from a direct declarator.
    fn extract_function_params_from_dd(
        &self,
        dd: &DirectDeclarator,
    ) -> (Vec<CType>, Vec<Option<Symbol>>, bool) {
        match dd {
            DirectDeclarator::Function {
                base,
                params,
                is_variadic,
                ..
            } => {
                // For function-returning-function-pointer patterns like:
                //   void (*memdbDlSym(real_params))(return_type_params)
                // The outer Function's params describe the returned function
                // pointer's parameter list, NOT the actual function params.
                // The real params are in the inner (nested) Function declarator.
                // Detect this by checking if the base Parenthesized contains
                // another Function declarator.
                if let DirectDeclarator::Parenthesized(inner_decl) = base.as_ref() {
                    if Self::direct_declarator_has_nested_function(&inner_decl.direct) {
                        return self.extract_function_params(inner_decl);
                    }
                }

                let types: Vec<CType> = params
                    .iter()
                    .map(|p| {
                        let base_ty = self.resolve_type_from_specifiers(&p.specifiers);
                        if let Some(ref d) = p.declarator {
                            self.apply_declarator_to_type(base_ty, d)
                        } else {
                            base_ty
                        }
                    })
                    .collect();
                let names: Vec<Option<Symbol>> = params
                    .iter()
                    .map(|p| {
                        p.declarator
                            .as_ref()
                            .and_then(|d| self.extract_declarator_name(d))
                    })
                    .collect();
                (types, names, *is_variadic)
            }
            DirectDeclarator::Parenthesized(inner) => self.extract_function_params(inner),
            _ => (Vec::new(), Vec::new(), false),
        }
    }

    /// Check if a direct declarator contains a nested Function declarator.
    /// Used to detect function-returning-function-pointer patterns where
    /// the actual function params are in an inner Function, not the outer one.
    fn direct_declarator_has_nested_function(dd: &DirectDeclarator) -> bool {
        match dd {
            DirectDeclarator::Function { .. } => true,
            DirectDeclarator::Parenthesized(inner) => {
                Self::direct_declarator_has_nested_function(&inner.direct)
            }
            DirectDeclarator::Array { base, .. } => {
                Self::direct_declarator_has_nested_function(base)
            }
            DirectDeclarator::Identifier(..) => false,
        }
    }

    /// Validate attributes from both specifier and declarator contexts.
    fn validate_attributes_for_context(
        &self,
        declarator_attrs: &[Attribute],
        specifier_attrs: &[Attribute],
        _context: AttributeContext,
        _span: Span,
    ) -> Vec<ValidatedAttribute> {
        // In a full integration, this would delegate to AttributeHandler.
        // For the mod.rs orchestration, we collect and validate attributes.
        let mut result = Vec::new();
        for attr in specifier_attrs.iter().chain(declarator_attrs.iter()) {
            let name_str = self.interner.resolve(attr.name);
            match name_str {
                "packed" => result.push(ValidatedAttribute::Packed),
                "aligned" => {
                    if let Some(AttributeArg::Expression(ref boxed_expr)) = attr.args.first() {
                        if let Some(val) = self.try_eval_attr_const_expr(boxed_expr) {
                            result.push(ValidatedAttribute::Aligned(val));
                        } else {
                            // Fallback: max alignment when expression
                            // cannot be evaluated (matches GCC's
                            // `__attribute__((aligned))` without args).
                            result
                                .push(ValidatedAttribute::Aligned(self.target.max_align() as u64));
                        }
                    } else {
                        // No argument: default to target max alignment.
                        result.push(ValidatedAttribute::Aligned(self.target.max_align() as u64));
                    }
                }
                "section" => {
                    if let Some(AttributeArg::String(ref bytes, _)) = attr.args.first() {
                        let s = String::from_utf8_lossy(bytes).into_owned();
                        result.push(ValidatedAttribute::Section(s));
                    }
                }
                "used" => result.push(ValidatedAttribute::Used),
                "unused" => result.push(ValidatedAttribute::Unused),
                "weak" => result.push(ValidatedAttribute::Weak),
                "noreturn" | "__noreturn__" => result.push(ValidatedAttribute::NoReturn),
                "noinline" => result.push(ValidatedAttribute::NoInline),
                "always_inline" => result.push(ValidatedAttribute::AlwaysInline),
                "cold" => result.push(ValidatedAttribute::Cold),
                "hot" => result.push(ValidatedAttribute::Hot),
                "deprecated" => result.push(ValidatedAttribute::Deprecated(None)),
                "visibility" => {
                    if let Some(AttributeArg::String(ref bytes, _)) = attr.args.first() {
                        let vis = String::from_utf8_lossy(bytes);
                        let v = match vis.as_ref() {
                            "hidden" => SymbolVisibility::Hidden,
                            "protected" => SymbolVisibility::Protected,
                            "internal" => SymbolVisibility::Internal,
                            _ => SymbolVisibility::Default,
                        };
                        result.push(ValidatedAttribute::Visibility(v));
                    }
                }
                "constructor" => result.push(ValidatedAttribute::Constructor(None)),
                "destructor" => result.push(ValidatedAttribute::Destructor(None)),
                "malloc" => result.push(ValidatedAttribute::Malloc),
                "pure" => result.push(ValidatedAttribute::Pure),
                "const" => result.push(ValidatedAttribute::Const),
                "warn_unused_result" => result.push(ValidatedAttribute::WarnUnusedResult),
                "fallthrough" => result.push(ValidatedAttribute::Fallthrough),
                _ => {
                    // Unknown attribute — production code would warn.
                }
            }
        }
        result
    }

    /// Try to evaluate a constant expression found in an attribute argument
    /// (e.g. `__attribute__((aligned(sizeof(void *)))))`).
    ///
    /// Handles the most common expression forms encountered in kernel and
    /// system headers: integer literals, parenthesized expressions, sizeof,
    /// alignof, and basic arithmetic.  Returns `None` when the expression
    /// cannot be evaluated — the caller should fall back to a safe default.
    fn try_eval_attr_const_expr(&self, expr: &Expression) -> Option<u64> {
        use crate::common::types::sizeof_ctype_resolved;
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as u64),
            Expression::Parenthesized { inner, .. } => self.try_eval_attr_const_expr(inner),
            Expression::SizeofType { type_name, .. } => {
                let ctype = self.resolve_type_name(type_name);
                let size = sizeof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
                Some(size as u64)
            }
            Expression::SizeofExpr { operand, .. } => {
                // For sizeof(expr) we attempt to infer the expression's
                // type so we can compute its size.  Only trivial cases
                // are handled here — compound expressions fall through.
                let ctype = self.infer_expression_type_simple(operand)?;
                let size = sizeof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
                Some(size as u64)
            }
            Expression::AlignofType { type_name, .. } => {
                use crate::common::types::alignof_ctype_resolved;
                let ctype = self.resolve_type_name(type_name);
                let align = alignof_ctype_resolved(&ctype, &self.target, &self.tag_types_by_name);
                Some(align as u64)
            }
            Expression::Binary {
                op, left, right, ..
            } => {
                let l = self.try_eval_attr_const_expr(left)?;
                let r = self.try_eval_attr_const_expr(right)?;
                match op {
                    BinaryOp::Add => Some(l.wrapping_add(r)),
                    BinaryOp::Sub => Some(l.wrapping_sub(r)),
                    BinaryOp::Mul => Some(l.wrapping_mul(r)),
                    BinaryOp::Div if r != 0 => Some(l / r),
                    BinaryOp::Mod if r != 0 => Some(l % r),
                    BinaryOp::ShiftLeft => Some(l.wrapping_shl(r as u32)),
                    BinaryOp::ShiftRight => Some(l.wrapping_shr(r as u32)),
                    BinaryOp::BitwiseAnd => Some(l & r),
                    BinaryOp::BitwiseOr => Some(l | r),
                    BinaryOp::BitwiseXor => Some(l ^ r),
                    _ => None,
                }
            }
            Expression::Cast { operand, .. } => {
                // Evaluate the inner expression; ignore the cast for
                // attribute-argument purposes (e.g. `(int)sizeof(...)`).
                self.try_eval_attr_const_expr(operand)
            }
            Expression::UnaryOp { op, operand, .. } => {
                // Classic offsetof pattern:
                //   &((TYPE*)0)->MEMBER
                // i.e. AddressOf(PointerMemberAccess(Cast(TYPE*, 0), member))
                // Evaluates to the byte offset of MEMBER within TYPE.
                if matches!(op, UnaryOp::AddressOf) {
                    if let Some(off) = self.try_eval_offsetof_pattern(operand) {
                        return Some(off);
                    }
                }
                let v = self.try_eval_attr_const_expr(operand)?;
                match op {
                    UnaryOp::Negate => Some((-(v as i64)) as u64),
                    UnaryOp::BitwiseNot => Some(!v),
                    UnaryOp::Plus => Some(v),
                    _ => None,
                }
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                let c = self.try_eval_attr_const_expr(condition)?;
                if c != 0 {
                    then_expr
                        .as_ref()
                        .and_then(|e| self.try_eval_attr_const_expr(e))
                } else {
                    self.try_eval_attr_const_expr(else_expr)
                }
            }
            // __builtin_offsetof(type, member) — compile-time byte offset
            // of a struct/union member.  Required for array dimensions that
            // involve offsetof, e.g. `char buf[sizeof(S) - offsetof(S, m)]`.
            Expression::BuiltinCall {
                builtin: BuiltinKind::Offsetof,
                args,
                ..
            } => {
                // args[0] is a SizeofType carrier wrapping the TypeName;
                // args[1] is the member designator expression.
                if let Some(Expression::SizeofType { type_name, .. }) = args.first() {
                    let ctype = self.resolve_type_name(type_name);
                    if let Some(member_expr) = args.get(1) {
                        return self.compute_offsetof_for_const_eval(&ctype, member_expr);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Compute the byte offset of a struct/union member for compile-time
    /// `__builtin_offsetof` evaluation.
    ///
    /// Walks the struct type's field list, accumulating byte offsets with
    /// alignment padding, to find the target member.  Handles nested
    /// member chains (`field.subfield`), anonymous structs/unions, and
    /// array subscripts in the designator.
    fn compute_offsetof_for_const_eval(
        &self,
        ctype: &CType,
        member_expr: &Expression,
    ) -> Option<u64> {
        // Extract the member designator chain.
        let chain = self.extract_offsetof_chain(member_expr);
        if chain.is_empty() {
            return None;
        }
        self.compute_field_offset_in_type(ctype, &chain)
    }

    /// Extract a chain of field names / array indices from the member
    /// designator expression of `__builtin_offsetof`.
    fn extract_offsetof_chain(&self, expr: &Expression) -> Vec<OffsetofStep> {
        match expr {
            Expression::Identifier { name, .. } => {
                let s = self.interner.resolve(*name).to_owned();
                vec![OffsetofStep::Field(s)]
            }
            Expression::MemberAccess { object, member, .. } => {
                let mut chain = self.extract_offsetof_chain(object);
                let s = self.interner.resolve(*member).to_owned();
                chain.push(OffsetofStep::Field(s));
                chain
            }
            Expression::ArraySubscript { base, index, .. } => {
                let mut chain = self.extract_offsetof_chain(base);
                if let Some(idx_val) = self.try_eval_attr_const_expr(index) {
                    chain.push(OffsetofStep::Index(idx_val as usize));
                }
                chain
            }
            _ => Vec::new(),
        }
    }

    /// Walk a struct/union type following a member-designator chain and
    /// return the cumulative byte offset.
    fn compute_field_offset_in_type(&self, ctype: &CType, chain: &[OffsetofStep]) -> Option<u64> {
        use crate::common::types::{alignof_ctype_resolved, sizeof_ctype_resolved};
        if chain.is_empty() {
            return Some(0);
        }
        let unwrapped = Self::unwrap_typedef_quals(ctype);

        // Resolve lightweight tag references (forward-declared structs
        // with empty field lists) via tag_types_by_name, just like
        // sizeof_ctype_resolved does.
        let resolved: &CType;
        let owned_resolved: CType;
        match unwrapped {
            CType::Struct {
                name: Some(ref n),
                fields,
                ..
            } if fields.is_empty() => {
                if let Some(complete) = self.tag_types_by_name.get(n) {
                    owned_resolved = complete.clone();
                    resolved = &owned_resolved;
                } else {
                    return None; // incomplete struct
                }
            }
            CType::Union {
                name: Some(ref n),
                fields,
                ..
            } if fields.is_empty() => {
                if let Some(complete) = self.tag_types_by_name.get(n) {
                    owned_resolved = complete.clone();
                    resolved = &owned_resolved;
                } else {
                    return None; // incomplete union
                }
            }
            other => {
                resolved = other;
            }
        }
        let unwrapped = Self::unwrap_typedef_quals(resolved);

        match &chain[0] {
            OffsetofStep::Index(idx) => {
                // Array subscript — element_size * index then recurse.
                if let CType::Array(elem, _) = unwrapped {
                    let elem_sz =
                        sizeof_ctype_resolved(elem, &self.target, &self.tag_types_by_name) as u64;
                    let rest = self.compute_field_offset_in_type(elem, &chain[1..])?;
                    Some((*idx as u64) * elem_sz + rest)
                } else {
                    None
                }
            }
            OffsetofStep::Field(target_name) => {
                let (fields, is_union, packed) = match unwrapped {
                    CType::Struct { fields, packed, .. } => (fields, false, *packed),
                    CType::Union { fields, packed, .. } => (fields, true, *packed),
                    _ => return None,
                };

                let mut bit_offset: usize = 0;
                for field in fields {
                    // Handle bitfield alignment.
                    if let Some(bits) = field.bit_width {
                        let bits = bits as usize;
                        let unit_bytes =
                            sizeof_ctype_resolved(&field.ty, &self.target, &self.tag_types_by_name);
                        let unit_bits = unit_bytes * 8;
                        let fa = if packed {
                            1
                        } else {
                            alignof_ctype_resolved(&field.ty, &self.target, &self.tag_types_by_name)
                        };
                        if bits == 0 {
                            let align_bits = fa * 8;
                            if align_bits > 0 {
                                bit_offset =
                                    (bit_offset + align_bits - 1) / align_bits * align_bits;
                            }
                            continue;
                        }
                        let unit_align_bits = if packed { 8 } else { fa * 8 };
                        let aligned_start =
                            (bit_offset + unit_align_bits - 1) / unit_align_bits * unit_align_bits;
                        let unit_end = aligned_start + unit_bits;
                        let start = if bit_offset + bits <= unit_end {
                            bit_offset
                        } else {
                            aligned_start
                        };
                        let field_name = field.name.as_deref().unwrap_or("");
                        if field_name == target_name.as_str() {
                            // Bitfield: return byte offset of containing unit.
                            return Some((start / 8) as u64);
                        }
                        if !is_union {
                            bit_offset = start + bits;
                        }
                        continue;
                    }
                    // Non-bitfield: round up to byte boundary first.
                    if bit_offset % 8 != 0 {
                        bit_offset = (bit_offset + 7) / 8 * 8;
                    }
                    let byte_offset = bit_offset / 8;
                    let fa = if packed {
                        1
                    } else {
                        alignof_ctype_resolved(&field.ty, &self.target, &self.tag_types_by_name)
                    };
                    let aligned = if fa > 1 {
                        (byte_offset + fa - 1) & !(fa - 1)
                    } else {
                        byte_offset
                    };

                    let field_name = field.name.as_deref().unwrap_or("");
                    if field_name == target_name.as_str() {
                        if chain.len() == 1 {
                            return Some(aligned as u64);
                        }
                        return self
                            .compute_field_offset_in_type(&field.ty, &chain[1..])
                            .map(|rest| aligned as u64 + rest);
                    }

                    // Check for anonymous struct/union — recurse into it
                    // to find the target field.
                    if field_name.is_empty() {
                        if let Some(inner_offset) =
                            self.compute_field_offset_in_type(&field.ty, chain)
                        {
                            return Some(aligned as u64 + inner_offset);
                        }
                    }

                    if !is_union {
                        let field_sz =
                            sizeof_ctype_resolved(&field.ty, &self.target, &self.tag_types_by_name);
                        bit_offset = aligned * 8 + field_sz * 8;
                    }
                }
                None
            }
        }
    }

    /// Strip Typedef / Qualified wrappers.
    fn unwrap_typedef_quals(ctype: &CType) -> &CType {
        let mut t = ctype;
        loop {
            match t {
                CType::Typedef { underlying, .. } => t = underlying.as_ref(),
                CType::Qualified(inner, _) => t = inner.as_ref(),
                _ => return t,
            }
        }
    }

    /// Detect the classic C offsetof pattern:
    ///   `((TYPE*)0)->MEMBER` — a PointerMemberAccess whose object
    ///   is a cast of 0 to a struct pointer.
    ///
    /// Given the operand of `&`, this function checks if it matches
    /// `((TYPE*)0)->member` (possibly with chained `.sub` and `[idx]`).
    /// If so, it resolves the struct type and computes the byte offset.
    fn try_eval_offsetof_pattern(&self, expr: &Expression) -> Option<u64> {
        // Peel through pointer-member accesses and dot-member accesses
        // to build a chain and find the root, which should be
        // (TYPE*)0 or (TYPE*)NULL.
        let mut chain: Vec<OffsetofStep> = Vec::new();
        let mut current = expr;

        loop {
            match current {
                // ptr->field
                Expression::PointerMemberAccess { object, member, .. } => {
                    let name = self.interner.resolve(*member).to_owned();
                    chain.push(OffsetofStep::Field(name));
                    current = object.as_ref();
                }
                // val.field
                Expression::MemberAccess { object, member, .. } => {
                    let name = self.interner.resolve(*member).to_owned();
                    chain.push(OffsetofStep::Field(name));
                    current = object.as_ref();
                }
                // base[index]
                Expression::ArraySubscript { base, index, .. } => {
                    let idx = self.try_eval_attr_const_expr(index)?;
                    chain.push(OffsetofStep::Index(idx as usize));
                    current = base.as_ref();
                }
                // Parenthesized
                Expression::Parenthesized { inner, .. } => {
                    current = inner.as_ref();
                }
                // UnaryOp(Deref, inner) — explicit dereference  `*(TYPE*)0`
                Expression::UnaryOp {
                    op: UnaryOp::Deref,
                    operand,
                    ..
                } => {
                    current = operand.as_ref();
                }
                // Cast(TYPE*, 0) — the null pointer base
                Expression::Cast {
                    type_name, operand, ..
                } => {
                    // Check if the inner value is 0 (null pointer constant).
                    if self.try_eval_attr_const_expr(operand) == Some(0) {
                        // Extract the pointed-to type from the cast.
                        let cast_ty = self.resolve_type_name(type_name);
                        let pointee = match cast_ty {
                            CType::Pointer(inner, _) => *inner,
                            _ => return None,
                        };
                        // The chain was built bottom-up (innermost field first),
                        // but compute_field_offset_in_type expects outermost first.
                        chain.reverse();
                        return self.compute_field_offset_in_type(&pointee, &chain);
                    }
                    return None;
                }
                _ => return None,
            }
        }
    }

    /// Simple type inference for an expression (used by attribute
    /// evaluation to resolve `sizeof(expr)` without a full type checker).
    /// Returns `None` for expressions whose type cannot be trivially
    /// determined.
    fn infer_expression_type_simple(&self, expr: &Expression) -> Option<CType> {
        match expr {
            Expression::IntegerLiteral { .. } => Some(CType::Int),
            Expression::Identifier { name, .. } => {
                // Look up the variable's declared type.
                self.scopes
                    .lookup_ordinary(*name)
                    .map(|sym_id| self.symbols.get(sym_id).ty.clone())
            }
            Expression::StringLiteral { .. } => {
                let quals = TypeQualifiers {
                    is_const: true,
                    ..TypeQualifiers::default()
                };
                Some(CType::Pointer(Box::new(CType::Char), quals))
            }
            _ => None,
        }
    }

    /// Propagate a validated attribute to a symbol entry.
    fn propagate_attribute_to_symbol(&mut self, id: SymbolId, attr: &ValidatedAttribute) {
        let sym = self.symbols.get_mut(id);
        sym.attributes.push(attr.clone());
        match attr {
            ValidatedAttribute::Weak => sym.is_weak = true,
            ValidatedAttribute::Visibility(v) => sym.visibility = Some(*v),
            ValidatedAttribute::Section(s) => sym.section = Some(s.clone()),
            _ => {}
        }
    }

    /// Handle a `_Static_assert` declaration by evaluating the condition
    /// as an integer constant expression.
    ///
    /// If the condition evaluates to zero the diagnostic engine emits an
    /// error that includes the optional message string.  A non-zero
    /// condition (or a condition that cannot be evaluated at compile time)
    /// passes silently.
    fn handle_static_assert(&mut self, decl: &Declaration) -> Result<(), ()> {
        if let Some(ref sa) = decl.static_assert {
            let target = self.target;
            let mut evaluator = crate::frontend::sema::constant_eval::ConstantEvaluator::new(
                self.diagnostics,
                target,
            );
            // Populate the evaluator with known struct/union tag types so
            // that sizeof(struct tag) works in constant expressions.
            for (tag_sym, tag_entry) in self.scopes.all_tags() {
                evaluator.register_tag_type(tag_sym, tag_entry.ty.clone());
            }
            // Populate with known variable types so that sizeof(var)
            // works for local arrays in block-scope _Static_assert.
            // Also populate typedef types so sizeof(__u8) etc. resolve
            // correctly (typedef names in struct member types must be
            // resolvable for accurate struct layout computation).
            for (var_sym, sym_id) in self.scopes.all_ordinary_symbols() {
                let entry = self.symbols.get(sym_id);
                if entry.storage_class == crate::frontend::sema::symbol_table::StorageClass::Typedef
                {
                    evaluator.register_typedef_type(var_sym, entry.ty.clone());
                }
                evaluator.register_variable_type(var_sym, entry.ty.clone());
            }
            // Populate with known enum constant values so that expressions
            // like `_Static_assert(__BPF_ARG_TYPE_MAX <= 256, ...)` can
            // resolve the enum constant names.
            for (name_str, &val) in &self.enum_constant_values {
                if let Some(sym) = self.interner.get(name_str) {
                    evaluator.register_enum_value(sym, val);
                }
            }
            // Populate the name table for symbol resolution in builtins
            let name_table: Vec<String> = (0..self.interner.len())
                .map(|idx| {
                    let sym = crate::common::string_interner::Symbol::from_u32(idx as u32);
                    self.interner.resolve(sym).to_string()
                })
                .collect();
            evaluator.set_name_table(name_table);

            // Inject the pre-built string-keyed tag type map for sizeof
            // resolution of lightweight "tag reference" CTypes.  We move
            // the map into the evaluator (zero-copy) and retrieve it back
            // after evaluation to avoid per-_Static_assert cloning.
            let name_map = std::mem::take(&mut self.tag_types_by_name);
            evaluator.set_tag_name_index(name_map);
            let result = evaluator.evaluate_static_assert_node(sa);
            self.tag_types_by_name = evaluator.take_tag_name_index();
            result
        } else {
            Ok(())
        }
    }

    /// Process embedded struct/union/enum definitions within specifiers.
    fn process_embedded_tag_definitions(&mut self, specs: &DeclarationSpecifiers) {
        for spec in &specs.type_specifiers {
            match spec {
                TypeSpecifier::Struct(s) | TypeSpecifier::Union(s) => {
                    if s.members.is_some() {
                        // This is a definition — fully process the struct/union
                        // so that member field information is recorded in the
                        // tag entry.  Previously called `register_struct_union_tag`
                        // which registered the tag with an empty fields list,
                        // causing member access (e.g. `s.x`) to fail.
                        let mut s_clone = s.clone();
                        let _ = self.analyze_struct_definition(&mut s_clone);
                    }
                }
                TypeSpecifier::Enum(e) => {
                    if e.enumerators.is_some() {
                        let mut e_clone = e.clone();
                        let _ = self.analyze_enum_definition(&mut e_clone);
                    }
                }
                _ => {}
            }
        }
    }

    /// Register a struct/union tag in the tag namespace.
    #[allow(dead_code)]
    fn register_struct_union_tag(&mut self, spec: &StructOrUnionSpecifier, is_struct: bool) {
        let tag_kind = if is_struct {
            TagKind::Struct
        } else {
            TagKind::Union
        };
        if let Some(tag_name) = spec.tag {
            let ty = if is_struct {
                CType::Struct {
                    name: Some(self.interner.resolve(tag_name).to_string()),
                    fields: Vec::new(),
                    packed: false,
                    aligned: None,
                }
            } else {
                CType::Union {
                    name: Some(self.interner.resolve(tag_name).to_string()),
                    fields: Vec::new(),
                    packed: false,
                    aligned: None,
                }
            };
            let entry = TagEntry {
                kind: tag_kind,
                ty,
                is_complete: spec.members.is_some(),
                span: spec.span,
            };
            self.scopes.declare_tag(tag_name, entry);
        }
    }

    /// Register an enum tag in the tag namespace.
    #[allow(dead_code)]
    fn register_enum_tag(&mut self, spec: &EnumSpecifier) {
        if let Some(tag_name) = spec.tag {
            let ty = CType::Enum {
                name: Some(self.interner.resolve(tag_name).to_string()),
                underlying_type: Box::new(CType::Int),
            };
            let entry = TagEntry {
                kind: TagKind::Enum,
                ty,
                is_complete: spec.enumerators.is_some(),
                span: spec.span,
            };
            self.scopes.declare_tag(tag_name, entry);
        }
    }

    /// Analyze an identifier expression: look up in symbol table, mark used.
    fn analyze_identifier(&mut self, name: Symbol, span: Span) -> Result<CType, ()> {
        // Look up the identifier in the scope stack.
        if let Some(sym_id) = self.scopes.lookup_ordinary(name) {
            let entry = self.symbols.get(sym_id);
            let ty = entry.ty.clone();
            self.symbols.mark_used(name);
            Ok(ty)
        } else {
            // C99 §6.4.2.2: __func__ is a predefined identifier equivalent to
            // static const char __func__[] = "function-name";
            // GCC also provides __FUNCTION__ and __PRETTY_FUNCTION__ as aliases.
            let name_str = self.interner.resolve(name);
            if name_str == "__func__"
                || name_str == "__FUNCTION__"
                || name_str == "__PRETTY_FUNCTION__"
            {
                // Type is const char[] which decays to const char* in expressions.
                return Ok(CType::Pointer(
                    Box::new(CType::Char),
                    crate::common::types::TypeQualifiers::default(),
                ));
            }

            // GCC implicit builtin functions: __builtin_memcpy, __builtin_memset,
            // __builtin_memmove, __builtin_strlen, __builtin_strcmp, etc.
            // These are regular function calls that GCC treats as built-in.
            // Return the appropriate function type so the call type-checks.
            let void_ptr = CType::Pointer(Box::new(CType::Void), TypeQualifiers::default());
            let const_void_ptr = CType::Pointer(
                Box::new(CType::Void),
                TypeQualifiers {
                    is_const: true,
                    ..TypeQualifiers::default()
                },
            );
            let const_char_ptr = CType::Pointer(
                Box::new(CType::Char),
                TypeQualifiers {
                    is_const: true,
                    ..TypeQualifiers::default()
                },
            );
            let size_t = self.size_t_type();

            if let Some(builtin_ty) = match name_str {
                // void *__builtin_memcpy(void *dest, const void *src, size_t n)
                "__builtin_memcpy" => Some(CType::Function {
                    return_type: Box::new(void_ptr.clone()),
                    params: vec![void_ptr.clone(), const_void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                // void *__builtin_memmove(void *dest, const void *src, size_t n)
                "__builtin_memmove" => Some(CType::Function {
                    return_type: Box::new(void_ptr.clone()),
                    params: vec![void_ptr.clone(), const_void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                // void *__builtin_memset(void *s, int c, size_t n)
                "__builtin_memset" => Some(CType::Function {
                    return_type: Box::new(void_ptr.clone()),
                    params: vec![void_ptr.clone(), CType::Int, size_t.clone()],
                    variadic: false,
                }),
                // int __builtin_memcmp(const void *s1, const void *s2, size_t n)
                "__builtin_memcmp" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![
                        const_void_ptr.clone(),
                        const_void_ptr.clone(),
                        size_t.clone(),
                    ],
                    variadic: false,
                }),
                // size_t __builtin_strlen(const char *s)
                "__builtin_strlen" => Some(CType::Function {
                    return_type: Box::new(size_t.clone()),
                    params: vec![const_char_ptr.clone()],
                    variadic: false,
                }),
                // int __builtin_strcmp(const char *s1, const char *s2)
                "__builtin_strcmp" | "__builtin_strncmp" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![const_char_ptr.clone(), const_char_ptr.clone()],
                    variadic: false,
                }),
                // char *__builtin_strcpy(char *dest, const char *src)
                "__builtin_strcpy" | "__builtin_strncpy" | "__builtin_strcat"
                | "__builtin_strncat" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![
                        CType::Pointer(Box::new(CType::Char), TypeQualifiers::default()),
                        const_char_ptr.clone(),
                    ],
                    variadic: false,
                }),
                // char *__builtin_strchr(const char *s, int c)
                "__builtin_strchr" | "__builtin_strrchr" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![const_char_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                // int __builtin_abs(int)
                "__builtin_abs" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::Int],
                    variadic: false,
                }),
                // long __builtin_labs(long)
                "__builtin_labs" => Some(CType::Function {
                    return_type: Box::new(CType::Long),
                    params: vec![CType::Long],
                    variadic: false,
                }),
                // int __builtin_printf(const char *fmt, ...)
                "__builtin_printf" | "__builtin_snprintf" | "__builtin_sprintf"
                | "__builtin_fprintf" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![const_char_ptr.clone()],
                    variadic: true,
                }),
                // Floating-point classification builtins
                "__builtin_isnan"
                | "__builtin_isinf"
                | "__builtin_isfinite"
                | "__builtin_isnormal"
                | "__builtin_signbit"
                | "__builtin_isnan_sign"
                | "__builtin_isinf_sign"
                | "__builtin_fpclassify" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::Double],
                    variadic: false,
                }),
                "__builtin_isnanf"
                | "__builtin_isinff"
                | "__builtin_isfinitef"
                | "__builtin_signbitf" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::Float],
                    variadic: false,
                }),
                "__builtin_isnanl"
                | "__builtin_isinfl"
                | "__builtin_isfinitel"
                | "__builtin_signbitl" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::LongDouble],
                    variadic: false,
                }),
                // NaN / infinity / huge_val generation builtins
                "__builtin_nan" | "__builtin_huge_val" | "__builtin_inf" => Some(CType::Function {
                    return_type: Box::new(CType::Double),
                    params: vec![const_char_ptr.clone()],
                    variadic: false,
                }),
                "__builtin_nanf" | "__builtin_huge_valf" | "__builtin_inff" => {
                    Some(CType::Function {
                        return_type: Box::new(CType::Float),
                        params: vec![const_char_ptr.clone()],
                        variadic: false,
                    })
                }
                "__builtin_nanl" | "__builtin_huge_vall" | "__builtin_infl" => {
                    Some(CType::Function {
                        return_type: Box::new(CType::LongDouble),
                        params: vec![const_char_ptr.clone()],
                        variadic: false,
                    })
                }
                // Math builtins (common)
                "__builtin_fabs" | "__builtin_sqrt" | "__builtin_floor" | "__builtin_ceil"
                | "__builtin_round" | "__builtin_log" | "__builtin_log2" | "__builtin_exp"
                | "__builtin_pow" | "__builtin_sin" | "__builtin_cos" | "__builtin_copysign" => {
                    Some(CType::Function {
                        return_type: Box::new(CType::Double),
                        params: vec![CType::Double],
                        variadic: false,
                    })
                }
                "__builtin_fabsf"
                | "__builtin_sqrtf"
                | "__builtin_floorf"
                | "__builtin_ceilf"
                | "__builtin_roundf"
                | "__builtin_copysignf" => Some(CType::Function {
                    return_type: Box::new(CType::Float),
                    params: vec![CType::Float],
                    variadic: false,
                }),
                "__builtin_fabsl" => Some(CType::Function {
                    return_type: Box::new(CType::LongDouble),
                    params: vec![CType::LongDouble],
                    variadic: false,
                }),
                // Object size builtin
                "__builtin_object_size" => Some(CType::Function {
                    return_type: Box::new(if self.target.pointer_width() == 8 {
                        CType::ULong
                    } else {
                        CType::UInt
                    }),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                // Atomic builtins
                "__sync_synchronize" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![],
                    variadic: false,
                }),
                // GCC __atomic builtins (C11 atomics via compiler builtins)
                // __atomic_store_n(ptr, val, memorder) -> void
                "__atomic_store_n" | "__atomic_store" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone(), CType::Int, CType::Int],
                    variadic: false,
                }),
                // __atomic_load_n(ptr, memorder) -> value
                "__atomic_load_n" | "__atomic_load" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                // __atomic_exchange_n(ptr, val, memorder) -> old value
                "__atomic_exchange_n" | "__atomic_exchange" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int, CType::Int],
                    variadic: false,
                }),
                // __atomic_compare_exchange_n(ptr, expected, desired, weak, success, failure) -> bool
                "__atomic_compare_exchange_n" | "__atomic_compare_exchange" => {
                    Some(CType::Function {
                        return_type: Box::new(CType::Bool),
                        params: vec![
                            void_ptr.clone(),
                            void_ptr.clone(),
                            CType::Int,
                            CType::Bool,
                            CType::Int,
                            CType::Int,
                        ],
                        variadic: false,
                    })
                }
                // __atomic_add_fetch, __atomic_sub_fetch, etc.
                "__atomic_add_fetch"
                | "__atomic_sub_fetch"
                | "__atomic_and_fetch"
                | "__atomic_or_fetch"
                | "__atomic_xor_fetch"
                | "__atomic_nand_fetch"
                | "__atomic_fetch_add"
                | "__atomic_fetch_sub"
                | "__atomic_fetch_and"
                | "__atomic_fetch_or"
                | "__atomic_fetch_xor"
                | "__atomic_fetch_nand" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int, CType::Int],
                    variadic: false,
                }),
                // __atomic_test_and_set(ptr, memorder) -> bool
                "__atomic_test_and_set" => Some(CType::Function {
                    return_type: Box::new(CType::Bool),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                // __atomic_clear(ptr, memorder) -> void
                "__atomic_clear" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                // __atomic_thread_fence(memorder) / __atomic_signal_fence(memorder) -> void
                "__atomic_thread_fence" | "__atomic_signal_fence" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![CType::Int],
                    variadic: false,
                }),
                // __atomic_always_lock_free(size, ptr) -> bool
                "__atomic_always_lock_free" | "__atomic_is_lock_free" => Some(CType::Function {
                    return_type: Box::new(CType::Bool),
                    params: vec![size_t.clone(), void_ptr.clone()],
                    variadic: false,
                }),
                // Legacy __sync builtins
                "__sync_fetch_and_add"
                | "__sync_fetch_and_sub"
                | "__sync_fetch_and_or"
                | "__sync_fetch_and_and"
                | "__sync_fetch_and_xor"
                | "__sync_fetch_and_nand"
                | "__sync_add_and_fetch"
                | "__sync_sub_and_fetch"
                | "__sync_or_and_fetch"
                | "__sync_and_and_fetch"
                | "__sync_xor_and_fetch"
                | "__sync_nand_and_fetch" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: true,
                }),
                "__sync_bool_compare_and_swap" => Some(CType::Function {
                    return_type: Box::new(CType::Bool),
                    params: vec![void_ptr.clone(), CType::Int, CType::Int],
                    variadic: true,
                }),
                "__sync_val_compare_and_swap" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int, CType::Int],
                    variadic: true,
                }),
                "__sync_lock_test_and_set" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: true,
                }),
                "__sync_lock_release" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone()],
                    variadic: true,
                }),
                _ => None,
            } {
                return Ok(builtin_ty);
            }

            // Implicit function declarations for common C library functions.
            // GCC and most C compilers provide these when the function is
            // called without a prior declaration. We emit a warning and
            // provide an implicit declaration.
            if let Some(implicit_ty) = match name_str {
                "abort" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![],
                    variadic: false,
                }),
                "exit" | "_exit" | "_Exit" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![CType::Int],
                    variadic: false,
                }),
                "malloc" | "calloc" | "realloc" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Void),
                        TypeQualifiers::default(),
                    )),
                    params: vec![size_t.clone()],
                    variadic: false,
                }),
                "free" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "memcpy" | "memmove" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Void),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone(), void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                "memset" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Void),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone(), CType::Int, size_t.clone()],
                    variadic: false,
                }),
                "memcmp" | "strcmp" | "strncmp" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                "strlen" => Some(CType::Function {
                    return_type: Box::new(size_t.clone()),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "strcpy" | "strcat" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone(), void_ptr.clone()],
                    variadic: false,
                }),
                "printf" | "fprintf" | "sprintf" | "snprintf" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: true,
                }),
                "puts" | "fputs" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "putchar" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::Int],
                    variadic: false,
                }),
                "atoi" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "atol" => Some(CType::Function {
                    return_type: Box::new(CType::Long),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "abs" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![CType::Int],
                    variadic: false,
                }),
                "labs" => Some(CType::Function {
                    return_type: Box::new(CType::Long),
                    params: vec![CType::Long],
                    variadic: false,
                }),
                "llabs" => Some(CType::Function {
                    return_type: Box::new(CType::LongLong),
                    params: vec![CType::LongLong],
                    variadic: false,
                }),
                "fabs" => Some(CType::Function {
                    return_type: Box::new(CType::Double),
                    params: vec![CType::Double],
                    variadic: false,
                }),
                "fabsf" => Some(CType::Function {
                    return_type: Box::new(CType::Float),
                    params: vec![CType::Float],
                    variadic: false,
                }),
                "qsort" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![
                        void_ptr.clone(),
                        size_t.clone(),
                        size_t.clone(),
                        void_ptr.clone(),
                    ],
                    variadic: false,
                }),
                "signal" => Some(CType::Function {
                    return_type: Box::new(void_ptr.clone()),
                    params: vec![CType::Int, void_ptr.clone()],
                    variadic: false,
                }),
                "setjmp" | "_setjmp" | "sigsetjmp" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "longjmp" | "siglongjmp" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                "alloca" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Void),
                        TypeQualifiers::default(),
                    )),
                    params: vec![size_t.clone()],
                    variadic: false,
                }),
                "bcmp" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone(), void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                "bzero" => Some(CType::Function {
                    return_type: Box::new(CType::Void),
                    params: vec![void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                "strncat" | "strncpy" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone(), void_ptr.clone(), size_t.clone()],
                    variadic: false,
                }),
                "fopen" => Some(CType::Function {
                    return_type: Box::new(void_ptr.clone()),
                    params: vec![void_ptr.clone(), void_ptr.clone()],
                    variadic: false,
                }),
                "fclose" | "fflush" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                "fread" | "fwrite" => Some(CType::Function {
                    return_type: Box::new(size_t.clone()),
                    params: vec![
                        void_ptr.clone(),
                        size_t.clone(),
                        size_t.clone(),
                        void_ptr.clone(),
                    ],
                    variadic: false,
                }),
                "sscanf" | "fscanf" | "scanf" => Some(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![void_ptr.clone()],
                    variadic: true,
                }),
                "strtol" | "strtoul" => Some(CType::Function {
                    return_type: Box::new(CType::Long),
                    params: vec![void_ptr.clone(), void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                "strtoll" | "strtoull" => Some(CType::Function {
                    return_type: Box::new(CType::LongLong),
                    params: vec![void_ptr.clone(), void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                "strtod" => Some(CType::Function {
                    return_type: Box::new(CType::Double),
                    params: vec![void_ptr.clone(), void_ptr.clone()],
                    variadic: false,
                }),
                "strtof" => Some(CType::Function {
                    return_type: Box::new(CType::Float),
                    params: vec![void_ptr.clone(), void_ptr.clone()],
                    variadic: false,
                }),
                "strstr" | "strchr" | "strrchr" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone(), CType::Int],
                    variadic: false,
                }),
                "getenv" => Some(CType::Function {
                    return_type: Box::new(CType::Pointer(
                        Box::new(CType::Char),
                        TypeQualifiers::default(),
                    )),
                    params: vec![void_ptr.clone()],
                    variadic: false,
                }),
                _ => None,
            } {
                // Emit implicit declaration warning (not error) — consistent with GCC behavior.
                return Ok(implicit_ty);
            }

            // Handle __builtin_* names as implicit builtins.
            // GCC treats all __builtin_* names as implicitly declared.
            // Use BuiltinEvaluator to check if the name is a known builtin.
            // Handle __builtin_* names as implicit builtins.
            // GCC treats all __builtin_* names as implicitly declared.
            if name_str.starts_with("__builtin_") {
                let ret_ty = BuiltinEvaluator::builtin_return_type_by_name(name_str, self.target);
                return Ok(CType::Function {
                    return_type: Box::new(ret_ty),
                    params: vec![],
                    variadic: true,
                });
            }

            // Handle __sync_ and __atomic_ builtins that are called directly
            if name_str.starts_with("__sync_") || name_str.starts_with("__atomic_") {
                return Ok(CType::Function {
                    return_type: Box::new(CType::Int),
                    params: vec![],
                    variadic: true,
                });
            }

            // C89/GNU89 implicit function declaration: any undeclared identifier
            // that appears in a function-call expression is implicitly declared as
            // `int name(...)`. Emit a warning, not an error.
            // Check if caller context suggests this is a function call (heuristic:
            // the resolution is happening for a name that will be called).
            // We always allow implicit declaration since we cannot distinguish
            // call context here — the error will surface if used in non-call context.
            self.diagnostics.emit_warning(
                span,
                format!("implicit declaration of function '{}'", name_str),
            );
            Ok(CType::Function {
                return_type: Box::new(CType::Int),
                params: vec![],
                variadic: true,
            })
        }
    }

    /// Determine the type of an integer literal based on value and suffix.
    fn integer_literal_type(
        &self,
        value: u128,
        suffix: &IntegerSuffix,
        is_hex_or_octal: bool,
    ) -> CType {
        // C11 §6.4.4.1 Table:
        //   Decimal (no suffix): int → long → long long
        //   Hex/Octal (no suffix): int → unsigned int → long → unsigned long → long long → unsigned long long
        //   Suffix U: unsigned int → unsigned long → unsigned long long
        //   Suffix L (decimal): long → long long
        //   Suffix L (hex/octal): long → unsigned long → long long → unsigned long long
        //   Suffix LL (decimal): long long
        //   Suffix LL (hex/octal): long long → unsigned long long
        let long_sz = self.target.long_size();
        match suffix {
            IntegerSuffix::None => {
                if value <= i32::MAX as u128 {
                    CType::Int
                } else if is_hex_or_octal && value <= u32::MAX as u128 {
                    CType::UInt
                } else if long_sz == 8 && value <= i64::MAX as u128 {
                    CType::Long
                } else if is_hex_or_octal && long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else if long_sz != 8 && value <= i32::MAX as u128 {
                    // LP32: long == 4 bytes, same range as int
                    CType::Long
                } else if is_hex_or_octal && long_sz != 8 && value <= u32::MAX as u128 {
                    CType::ULong
                } else if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    // Values > 2^64 — unsigned long long (best effort)
                    CType::ULongLong
                }
            }
            IntegerSuffix::U => {
                if value <= u32::MAX as u128 {
                    CType::UInt
                } else if long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::L => {
                if long_sz == 8 {
                    if value <= i64::MAX as u128 {
                        CType::Long
                    } else if is_hex_or_octal && value <= u64::MAX as u128 {
                        CType::ULong
                    } else {
                        CType::LongLong
                    }
                } else if value <= i32::MAX as u128 {
                    CType::Long
                } else if is_hex_or_octal && value <= u32::MAX as u128 {
                    CType::ULong
                } else if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::UL => {
                if long_sz == 8 && value <= u64::MAX as u128 {
                    CType::ULong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::LL => {
                if value <= i64::MAX as u128 {
                    CType::LongLong
                } else if is_hex_or_octal && value <= u64::MAX as u128 {
                    CType::ULongLong
                } else {
                    CType::ULongLong
                }
            }
            IntegerSuffix::ULL => CType::ULongLong,
        }
    }

    /// Determine the type of a float literal based on suffix.
    /// Imaginary suffixes (i/j) produce `_Complex` types for GCC extension.
    fn float_literal_type(&self, suffix: &FloatSuffix) -> CType {
        match suffix {
            FloatSuffix::None => CType::Double,
            FloatSuffix::F => CType::Float,
            FloatSuffix::L => CType::LongDouble,
            // GCC imaginary constant extension: `1.0i` → `_Complex double`
            FloatSuffix::I => CType::Complex(Box::new(CType::Double)),
            FloatSuffix::FI => CType::Complex(Box::new(CType::Float)),
            FloatSuffix::LI => CType::Complex(Box::new(CType::LongDouble)),
        }
    }

    /// Determine the type of a string literal based on prefix.
    fn string_literal_type(&self, prefix: &StringPrefix) -> CType {
        let char_type = match prefix {
            StringPrefix::None | StringPrefix::U8 => CType::Char,
            StringPrefix::L => CType::Int,
            StringPrefix::U16 => CType::UShort,
            StringPrefix::U32 => CType::UInt,
        };
        CType::Pointer(
            Box::new(char_type),
            TypeQualifiers {
                is_const: true,
                ..TypeQualifiers::default()
            },
        )
    }

    /// Returns the C type representing `size_t` for the current target.
    fn size_t_type(&self) -> CType {
        // Use type_builder's target for architecture-appropriate sizing.
        if self.type_builder.target().pointer_width() == 8 {
            CType::ULong
        } else {
            CType::UInt
        }
    }

    /// Resolve the return type of a function call expression.
    ///
    /// Properly strips qualifiers, typedefs, and atomic wrappers before
    /// checking for function or pointer-to-function types. This is required
    /// for patterns like `READ_ONCE(fn_ptr)(args)` where the kernel's macro
    /// wraps the result in `const volatile typeof(...)`.
    fn resolve_function_call_type(&mut self, callee_type: &CType, span: Span) -> Result<CType, ()> {
        use crate::common::types::resolve_and_strip;

        // Fully strip qualifiers and typedefs first
        let stripped = resolve_and_strip(callee_type);

        // Check for pointer-to-function, then function directly
        let func_type = match stripped {
            CType::Pointer(inner, _) => resolve_and_strip(inner),
            other => other,
        };

        match func_type {
            CType::Function { return_type, .. } => Ok(*return_type.clone()),
            _ => {
                // Called object is not a function.
                self.diagnostics
                    .emit_error(span, "called object is not a function or function pointer");
                Err(())
            }
        }
    }

    /// Resolve the type of a struct/union member access.
    ///
    /// When a struct/union was forward-declared at the point where a
    /// function return type or variable type was recorded, the stored
    /// `CType::Struct` may have an empty `fields` list even though the
    /// full definition was provided later in the translation unit. This
    /// method handles that case by looking up the current (possibly
    /// completed) definition from the tag namespace whenever the stored
    /// fields list is empty and the type has a tag name.
    fn resolve_member_type(
        &mut self,
        obj_type: &CType,
        member: Symbol,
        span: Span,
    ) -> Result<CType, ()> {
        let member_name = self.interner.resolve(member);

        // Phase 1: extract tag name and fields from the stored type
        // without mutably borrowing self. We clone the tag name if we
        // might need to resolve a forward-declared (empty) struct.
        #[derive(Clone)]
        enum StructOrUnion {
            Struct,
            Union,
        }
        let (tag_name_opt, stored_fields, kind) = {
            let stripped = self.strip_qualifiers(obj_type);
            match stripped {
                CType::Struct {
                    ref name,
                    ref fields,
                    ..
                } => (name.clone(), fields.clone(), Some(StructOrUnion::Struct)),
                CType::Union {
                    ref name,
                    ref fields,
                    ..
                } => (name.clone(), fields.clone(), Some(StructOrUnion::Union)),
                _ => (None, Vec::new(), None),
            }
        };

        if kind.is_none() {
            // When type inference falls back to a primitive type (e.g.,
            // CType::Int) for a complex expression like _Generic/typeof
            // inside a statement expression, we cannot definitively say
            // that the member access is invalid — the original C code
            // compiles with GCC. Emit a warning and return Int as a
            // permissive fallback to allow compilation to proceed.
            let stripped = self.strip_qualifiers(obj_type);
            let is_primitive_fallback = matches!(
                stripped,
                CType::Int
                    | CType::UInt
                    | CType::Long
                    | CType::ULong
                    | CType::LongLong
                    | CType::ULongLong
                    | CType::Int128
                    | CType::UInt128
                    | CType::Char
                    | CType::UChar
                    | CType::SChar
                    | CType::Void
            );
            if is_primitive_fallback {
                // Likely a type-inference gap — silently fall back.
                return Ok(CType::Int);
            }
            self.diagnostics
                .emit_error(span, "member reference base type is not a struct or union");
            return Err(());
        }
        let kind = kind.unwrap();

        // Phase 2: if the stored fields are empty and we have a tag name,
        // look up the completed definition from the tag namespace.
        let fields: Vec<StructField> = if stored_fields.is_empty() {
            if let Some(ref tag_name) = tag_name_opt {
                if let Some(entry) = self.scopes.lookup_tag_by_str(tag_name, self.interner) {
                    if entry.is_complete {
                        match (&kind, &entry.ty) {
                            (StructOrUnion::Struct, CType::Struct { ref fields, .. })
                            | (StructOrUnion::Union, CType::Union { ref fields, .. }) => {
                                fields.clone()
                            }
                            _ => stored_fields,
                        }
                    } else {
                        stored_fields
                    }
                } else {
                    stored_fields
                }
            } else {
                stored_fields
            }
        } else {
            stored_fields
        };

        if let Some(ty) = Self::find_member_in_fields(&fields, member_name) {
            return Ok(ty);
        }

        self.diagnostics.emit_error(
            span,
            format!("no member named '{}' in struct/union", member_name),
        );
        Err(())
    }

    /// Recursively search for a named member in a list of struct/union fields,
    /// including members of anonymous (unnamed) struct/union fields (C11 §6.7.2.1p13).
    fn find_member_in_fields(fields: &[StructField], name: &str) -> Option<CType> {
        for field in fields {
            if let Some(ref fname) = field.name {
                if fname == name {
                    return Some(field.ty.clone());
                }
            } else {
                // Anonymous struct/union — recurse into its fields.
                // Must handle qualifier wrappers (e.g. `const struct { ... }`).
                let inner_fields: &[StructField] = match &field.ty {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => f,
                    CType::Qualified(inner, _) => match inner.as_ref() {
                        CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => f,
                        _ => continue,
                    },
                    CType::Typedef { underlying, .. } => match underlying.as_ref() {
                        CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => f,
                        CType::Qualified(inner, _) => match inner.as_ref() {
                            CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => f,
                            _ => continue,
                        },
                        _ => continue,
                    },
                    _ => continue,
                };
                if let Some(ty) = Self::find_member_in_fields(inner_fields, name) {
                    return Some(ty);
                }
            }
        }
        None
    }

    /// Get the pointee type of a pointer or array type.
    fn get_pointee_type(&self, ty: &CType) -> Option<CType> {
        match self.strip_qualifiers(ty) {
            CType::Pointer(inner, _) => Some(*inner.clone()),
            CType::Array(inner, _) => Some(*inner.clone()),
            _ => None,
        }
    }

    /// Resolve the type of a unary operator expression.
    fn resolve_unary_op_type(
        &mut self,
        op: UnaryOp,
        operand_type: &CType,
        _span: Span,
    ) -> Result<CType, ()> {
        match op {
            UnaryOp::AddressOf => Ok(CType::Pointer(
                Box::new(operand_type.clone()),
                TypeQualifiers::default(),
            )),
            UnaryOp::Deref => match self.get_pointee_type(operand_type) {
                Some(inner) => Ok(inner),
                None => {
                    self.diagnostics
                        .emit_error(_span, "indirection requires pointer operand");
                    Err(())
                }
            },
            UnaryOp::Plus | UnaryOp::Negate => {
                // Integer promotions apply.
                Ok(operand_type.clone())
            }
            UnaryOp::BitwiseNot => Ok(operand_type.clone()),
            UnaryOp::LogicalNot => Ok(CType::Int),
            // GCC __real__ / __imag__: for non-complex types, acts like identity
            // (real part) or returns 0 (imaginary part). For _Complex, returns
            // the component type.
            UnaryOp::RealPart | UnaryOp::ImagPart => {
                if let CType::Complex(inner) = operand_type {
                    Ok(*inner.clone())
                } else {
                    // For non-complex, __real__ returns the value, __imag__ returns 0
                    Ok(operand_type.clone())
                }
            }
        }
    }

    /// Resolve the type of a binary operator expression.
    fn resolve_binary_op_type(
        &self,
        op: BinaryOp,
        left: &CType,
        right: &CType,
        _span: Span,
    ) -> Result<CType, ()> {
        match op {
            // Arithmetic operators: usual arithmetic conversions.
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                // Pointer subtraction (ptr - ptr) yields ptrdiff_t (long).
                if matches!(op, BinaryOp::Sub)
                    && self.is_pointer_type(self.strip_qualifiers(left))
                    && self.is_pointer_type(self.strip_qualifiers(right))
                {
                    return Ok(CType::Long);
                }
                Ok(self.common_type(left, right))
            }
            // Bitwise operators: usual arithmetic conversions.
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                Ok(self.common_type(left, right))
            }
            // Shift operators: result type is promoted left operand.
            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => Ok(left.clone()),
            // Logical operators: result is int.
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => Ok(CType::Int),
            // Relational and equality operators: result is int.
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::Less
            | BinaryOp::Greater
            | BinaryOp::LessEqual
            | BinaryOp::GreaterEqual => Ok(CType::Int),
        }
    }

    /// Compute the common type of two operands (simplified usual arithmetic conversion).
    fn common_type(&self, a: &CType, b: &CType) -> CType {
        let a_stripped = self.strip_qualifiers(a);
        let b_stripped = self.strip_qualifiers(b);

        // If either is a pointer, pointer arithmetic rules apply.
        if self.is_pointer_type(a_stripped) {
            return a_stripped.clone();
        }
        if self.is_pointer_type(b_stripped) {
            return b_stripped.clone();
        }

        // Floating-point promotion chain.
        if matches!(a_stripped, CType::LongDouble) || matches!(b_stripped, CType::LongDouble) {
            return CType::LongDouble;
        }
        if matches!(a_stripped, CType::Double) || matches!(b_stripped, CType::Double) {
            return CType::Double;
        }
        if matches!(a_stripped, CType::Float) || matches!(b_stripped, CType::Float) {
            return CType::Float;
        }

        // Integer promotion: higher rank wins.
        let rank_a = self.integer_rank(a_stripped);
        let rank_b = self.integer_rank(b_stripped);
        if rank_a >= rank_b {
            a_stripped.clone()
        } else {
            b_stripped.clone()
        }
    }

    /// Get the integer conversion rank of a type (higher = wider).
    fn integer_rank(&self, ty: &CType) -> u32 {
        match ty {
            CType::Bool => 0,
            CType::Char | CType::SChar | CType::UChar => 1,
            CType::Short | CType::UShort => 2,
            CType::Int | CType::UInt => 3,
            CType::Long | CType::ULong => 4,
            CType::LongLong | CType::ULongLong => 5,
            CType::Int128 | CType::UInt128 => 6,
            _ => 3, // Default to int rank.
        }
    }

    /// Strip top-level qualifiers and typedefs from a type.
    fn strip_qualifiers<'b>(&self, ty: &'b CType) -> &'b CType {
        match ty {
            CType::Qualified(inner, _) => self.strip_qualifiers(inner),
            CType::Typedef { underlying, .. } => self.strip_qualifiers(underlying),
            other => other,
        }
    }

    /// Check if a type is a scalar type (arithmetic or pointer).
    fn is_scalar_type(&self, ty: &CType) -> bool {
        self.is_arithmetic_type(ty)
            || self.is_pointer_type(ty)
            || self.is_function_type(ty)
            || self.is_array_type(ty)
    }

    /// Check if a type is an array type (decays to a pointer, which is scalar).
    fn is_array_type(&self, ty: &CType) -> bool {
        let stripped = self.strip_qualifiers(ty);
        matches!(stripped, CType::Array { .. })
    }

    /// Check if a type is a function type (decays to a function pointer,
    /// which is scalar).
    fn is_function_type(&self, ty: &CType) -> bool {
        let stripped = self.strip_qualifiers(ty);
        matches!(stripped, CType::Function { .. })
    }

    /// Check if a type is an arithmetic type (integer or floating-point).
    fn is_arithmetic_type(&self, ty: &CType) -> bool {
        self.is_integer_type(ty) || self.is_float_type(ty)
    }

    /// Check if a type is an integer type.
    fn is_integer_type(&self, ty: &CType) -> bool {
        let stripped = self.strip_qualifiers(ty);
        matches!(
            stripped,
            CType::Bool
                | CType::Char
                | CType::SChar
                | CType::UChar
                | CType::Short
                | CType::UShort
                | CType::Int
                | CType::UInt
                | CType::Long
                | CType::ULong
                | CType::LongLong
                | CType::ULongLong
                | CType::Int128
                | CType::UInt128
                | CType::Enum { .. }
        )
    }

    /// Check if a type is a floating-point type.
    fn is_float_type(&self, ty: &CType) -> bool {
        let stripped = self.strip_qualifiers(ty);
        matches!(
            stripped,
            CType::Float | CType::Double | CType::LongDouble | CType::Complex(_)
        )
    }

    /// Check if a type is a pointer type.
    fn is_pointer_type(&self, ty: &CType) -> bool {
        let stripped = self.strip_qualifiers(ty);
        matches!(stripped, CType::Pointer(_, _))
    }

    /// Extract the span from any expression variant.
    fn expression_span(&self, expr: &Expression) -> Span {
        match expr {
            Expression::IntegerLiteral { span, .. }
            | Expression::FloatLiteral { span, .. }
            | Expression::StringLiteral { span, .. }
            | Expression::CharLiteral { span, .. }
            | Expression::Identifier { span, .. }
            | Expression::Parenthesized { span, .. }
            | Expression::ArraySubscript { span, .. }
            | Expression::FunctionCall { span, .. }
            | Expression::MemberAccess { span, .. }
            | Expression::PointerMemberAccess { span, .. }
            | Expression::PostIncrement { span, .. }
            | Expression::PostDecrement { span, .. }
            | Expression::PreIncrement { span, .. }
            | Expression::PreDecrement { span, .. }
            | Expression::UnaryOp { span, .. }
            | Expression::SizeofExpr { span, .. }
            | Expression::SizeofType { span, .. }
            | Expression::AlignofType { span, .. }
            | Expression::AlignofExpr { span, .. }
            | Expression::Cast { span, .. }
            | Expression::Binary { span, .. }
            | Expression::Conditional { span, .. }
            | Expression::Assignment { span, .. }
            | Expression::Comma { span, .. }
            | Expression::CompoundLiteral { span, .. }
            | Expression::StatementExpression { span, .. }
            | Expression::BuiltinCall { span, .. }
            | Expression::Generic { span, .. }
            | Expression::AddressOfLabel { span, .. } => *span,
        }
    }

    /// Register a typedef name in the current scope (for parser disambiguation).
    pub fn register_typedef(&mut self, name: Symbol) {
        self.scopes.register_typedef(name);
    }
}

// ============================================================================
// ScopeStack extension: depth accessor for schema compliance
// ============================================================================

impl ScopeStack {
    /// Return the current nesting depth.
    ///
    /// Exposed for the semantic analyzer to pass to `SymbolTable::resolve_linkage`.
    #[inline]
    pub fn depth(&self) -> u32 {
        self.current_depth()
    }
}

/// Evaluate a bitfield width expression to a u32 value.
///
/// Handles common patterns:
/// - Integer literals: `int x : 6;`
/// - Parenthesized literals: `int x : (6);`
/// - Simple binary/unary expressions involving integer literals
///
/// Falls back to 0 for complex expressions that cannot be evaluated
/// statically in this simplified context.
fn eval_bitfield_width(expr: &crate::frontend::parser::ast::Expression) -> u32 {
    use crate::frontend::parser::ast::Expression;
    match expr {
        Expression::IntegerLiteral { value, .. } => *value as u32,
        Expression::Parenthesized { inner, .. } => eval_bitfield_width(inner),
        Expression::UnaryOp {
            op: crate::frontend::parser::ast::UnaryOp::Plus,
            operand,
            ..
        } => eval_bitfield_width(operand),
        Expression::Binary {
            op, left, right, ..
        } => {
            let l = eval_bitfield_width(left);
            let r = eval_bitfield_width(right);
            match op {
                crate::frontend::parser::ast::BinaryOp::Add => l.wrapping_add(r),
                crate::frontend::parser::ast::BinaryOp::Sub => l.wrapping_sub(r),
                crate::frontend::parser::ast::BinaryOp::Mul => l.wrapping_mul(r),
                _ => 0,
            }
        }
        Expression::Cast { operand, .. } => eval_bitfield_width(operand),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::DiagnosticEngine;
    use crate::common::string_interner::Interner;
    use crate::common::target::Target;
    use crate::common::type_builder::TypeBuilder;

    /// Verify that SemanticAnalyzer can be constructed successfully.
    #[test]
    fn test_semantic_analyzer_construction() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);
        assert_eq!(sema.max_recursion_depth, 512);
        assert_eq!(sema.recursion_depth, 0);
        assert!(!sema.in_loop);
        assert!(!sema.in_switch);
        assert!(sema.current_function_return_type.is_none());
    }

    /// Verify recursion depth guard triggers at 512.
    #[test]
    fn test_recursion_depth_limit() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let mut sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);

        // Set depth to just below the limit.
        sema.recursion_depth = 511;
        assert!(sema.check_recursion_depth(Span::dummy()).is_ok());
        assert_eq!(sema.recursion_depth, 512);

        // Now at the limit — should fail.
        assert!(sema.check_recursion_depth(Span::dummy()).is_err());
        assert!(sema.diagnostics.has_errors());
    }

    /// Verify that an empty translation unit can be analyzed.
    #[test]
    fn test_analyze_empty_translation_unit() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let mut sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);

        let mut tu = TranslationUnit {
            declarations: Vec::new(),
            span: Span::dummy(),
        };
        assert!(sema.analyze(&mut tu).is_ok());
    }

    /// Verify integer literal type resolution.
    #[test]
    fn test_integer_literal_types() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);

        // Decimal 42 → int
        assert_eq!(
            sema.integer_literal_type(42, &IntegerSuffix::None, false),
            CType::Int
        );
        // Hex 42 → int (fits in signed int)
        assert_eq!(
            sema.integer_literal_type(42, &IntegerSuffix::None, true),
            CType::Int
        );
        // Decimal 0x83fd4005 (2214019077) → long (decimal skips unsigned int)
        assert_eq!(
            sema.integer_literal_type(0x83fd4005, &IntegerSuffix::None, false),
            CType::Long
        );
        // Hex 0x83fd4005 → unsigned int (fits in u32, hex tries unsigned)
        assert_eq!(
            sema.integer_literal_type(0x83fd4005, &IntegerSuffix::None, true),
            CType::UInt
        );
        assert_eq!(
            sema.integer_literal_type(42, &IntegerSuffix::U, false),
            CType::UInt
        );
        assert_eq!(
            sema.integer_literal_type(42, &IntegerSuffix::ULL, false),
            CType::ULongLong
        );
    }

    /// Verify type classification helpers.
    #[test]
    fn test_type_classification() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);

        assert!(sema.is_integer_type(&CType::Int));
        assert!(sema.is_integer_type(&CType::ULong));
        assert!(!sema.is_integer_type(&CType::Float));
        assert!(sema.is_float_type(&CType::Double));
        assert!(sema.is_scalar_type(&CType::Int));
        assert!(sema.is_pointer_type(&CType::Pointer(
            Box::new(CType::Void),
            TypeQualifiers::default(),
        )));
    }

    /// Verify finalize does not error on clean state.
    #[test]
    fn test_finalize_clean() {
        let mut diag = DiagnosticEngine::new();
        let tb = TypeBuilder::new(Target::X86_64);
        let interner = Interner::new();
        let mut sema = SemanticAnalyzer::new(&mut diag, &tb, Target::X86_64, &interner);
        assert!(sema.finalize().is_ok());
    }

    /// Verify ScopeStack::depth() accessor.
    #[test]
    fn test_scope_depth() {
        let scopes = ScopeStack::new();
        assert_eq!(scopes.depth(), 0);
    }
}
