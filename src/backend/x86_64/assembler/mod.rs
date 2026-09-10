//! x86-64 built-in assembler — encodes machine instructions into relocatable
//! ELF `.o` object files.
//!
//! This module serves as the assembler driver:
//! 1. Receives [`MachineFunction`] from x86-64 instruction selection
//! 2. Iterates machine instructions, dispatching to the encoder
//! 3. Collects relocations for external symbol references
//! 4. Produces ELF `.o` files with `.text`, `.data`, `.bss`, `.rodata`,
//!    `.symtab`, `.rela.text`
//!
//! No external assembler (`as`, `nasm`, `llvm-mc`) is invoked — everything
//! is built-in.
//!
//! # Integration
//!
//! - Receives [`MachineFunction`] from `src/backend/x86_64/codegen.rs`
//!   (via the [`ArchCodegen`](crate::backend::traits::ArchCodegen) trait)
//! - Uses [`encoder`] for instruction-level binary encoding
//! - Uses [`relocations`] for relocation type definitions and record construction
//! - Uses [`registers`](crate::backend::x86_64::registers) for register encoding
//! - Uses [`ElfWriter`](crate::backend::elf_writer_common::ElfWriter) for ELF output
//! - Uses [`SecurityConfig`](crate::backend::x86_64::security::SecurityConfig) for
//!   retpoline thunks and CET note sections
//! - Uses [`FxHashMap`](crate::common::fx_hash::FxHashMap) for label maps
//!
//! # Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references. No external crates.

pub mod encoder;
pub mod relocations;

// ---------------------------------------------------------------------------
// Re-exports — public API surface for downstream consumers
// ---------------------------------------------------------------------------

pub use self::encoder::X86_64Encoder;
pub use self::relocations::{RelocationEntry, X86_64RelocationType};

// ---------------------------------------------------------------------------
// Internal imports
// ---------------------------------------------------------------------------

use crate::backend::elf_writer_common::{
    ElfSymbol, ElfWriter, Section, ET_REL, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHT_NOBITS,
    SHT_NOTE, SHT_PROGBITS, SHT_RELA, STB_GLOBAL, STB_LOCAL, STB_WEAK, STT_FUNC, STT_NOTYPE,
    STT_SECTION, STV_DEFAULT, STV_HIDDEN,
};
use crate::backend::traits::{MachineFunction, MachineOperand};
use crate::backend::x86_64::codegen::X86Opcode;
use crate::backend::x86_64::registers::{
    hw_encoding, reg_name_64, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX,
    RSI, RSP,
};
#[cfg(test)]
use crate::backend::x86_64::security::ENDBR64_BYTES;
use crate::backend::x86_64::security::{generate_cet_note_section, SecurityConfig};
use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// ===========================================================================
// SymbolSection — section classification for assembler symbols
// ===========================================================================

/// Section in which an assembler symbol resides.
///
/// Used by [`AssemblerSymbol`] to indicate which ELF section the symbol
/// belongs to, enabling correct `st_shndx` assignment during ELF output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolSection {
    /// `.text` section — executable code.
    Text,
    /// `.data` section — initialized writable data.
    Data,
    /// `.bss` section — zero-initialized data (no file space).
    Bss,
    /// `.rodata` section — read-only data.
    Rodata,
    /// Undefined symbol — referenced but not defined in this object.
    Undefined,
    /// Absolute symbol — value is an absolute constant.
    Absolute,
}

// ===========================================================================
// SymbolBinding — ELF symbol binding
// ===========================================================================

/// ELF symbol binding attribute.
///
/// Maps directly to ELF `STB_LOCAL`, `STB_GLOBAL`, `STB_WEAK` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBinding {
    /// Local symbol — not visible outside the object file.
    Local,
    /// Global symbol — visible to all object files.
    Global,
    /// Weak symbol — like global but with lower precedence.
    Weak,
}

impl SymbolBinding {
    /// Convert to the ELF `STB_*` constant.
    #[inline]
    fn to_elf(self) -> u8 {
        match self {
            SymbolBinding::Local => STB_LOCAL,
            SymbolBinding::Global => STB_GLOBAL,
            SymbolBinding::Weak => STB_WEAK,
        }
    }
}

// ===========================================================================
// SymbolType — ELF symbol type
// ===========================================================================

/// ELF symbol type attribute.
///
/// Maps directly to ELF `STT_NOTYPE`, `STT_FUNC`, `STT_OBJECT`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    /// No type specified.
    NoType,
    /// Function entry point.
    Function,
    /// Data object (variable, array, etc.).
    Object,
    /// Section symbol.
    Section,
    /// Source file symbol.
    File,
}

impl SymbolType {
    /// Convert to the ELF `STT_*` constant.
    #[inline]
    fn to_elf(self) -> u8 {
        match self {
            SymbolType::NoType => STT_NOTYPE,
            SymbolType::Function => STT_FUNC,
            SymbolType::Object => 1, // STT_OBJECT
            SymbolType::Section => STT_SECTION,
            SymbolType::File => 4, // STT_FILE
        }
    }
}

// ===========================================================================
// SymbolVisibility — ELF symbol visibility
// ===========================================================================

/// ELF symbol visibility attribute.
///
/// Controls whether a symbol is visible from other shared objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolVisibility {
    /// Default visibility — determined by binding type.
    Default,
    /// Hidden visibility — not visible from other components.
    Hidden,
    /// Protected visibility — visible but not preemptable.
    Protected,
}

impl SymbolVisibility {
    /// Convert to the ELF `STV_*` constant.
    #[inline]
    fn to_elf(self) -> u8 {
        match self {
            SymbolVisibility::Default => STV_DEFAULT,
            SymbolVisibility::Hidden => STV_HIDDEN,
            SymbolVisibility::Protected => 3, // STV_PROTECTED
        }
    }
}

// ===========================================================================
// AssemblerSymbol — symbol entry in assembler output
// ===========================================================================

/// A symbol in the assembler output object file.
///
/// Represents a named location (function, variable, label) that will be
/// placed in the ELF `.symtab` section.
#[derive(Debug, Clone)]
pub struct AssemblerSymbol {
    /// Symbol name — stored in the `.strtab` string table.
    pub name: String,
    /// Offset within its section (for defined symbols) or 0 for undefined.
    pub offset: u64,
    /// Size of the symbol in bytes (e.g., function code length).
    pub size: u64,
    /// Which section this symbol belongs to.
    pub section: SymbolSection,
    /// Binding: local, global, or weak.
    pub binding: SymbolBinding,
    /// Symbol type: function, object, section, etc.
    pub sym_type: SymbolType,
    /// Visibility: default, hidden, or protected.
    pub visibility: SymbolVisibility,
}

// ===========================================================================
// LabelFixup — pending label reference
// ===========================================================================

/// A fixup for an unresolved label reference within the `.text` section.
///
/// When an instruction references a label that has not yet been defined,
/// a `LabelFixup` is recorded. Once the label is defined (or at the end
/// of assembly), the fixup is resolved by patching the encoded bytes.
#[derive(Debug, Clone)]
pub struct LabelFixup {
    /// Byte offset within `.text` where the fixup address must be patched.
    pub offset: usize,
    /// Size of the fixup field in bytes (typically 4 for rel32).
    pub size: u8,
    /// Whether this is a PC-relative fixup.
    pub pc_relative: bool,
    /// Addend value for the relocation computation.
    pub addend: i64,
}

// ===========================================================================
// AssemblyResult — output of the assemble() function
// ===========================================================================

/// Result of assembling a single [`MachineFunction`].
///
/// Contains the encoded `.text` section bytes, collected relocations,
/// symbol definitions, and any retpoline thunk data generated for
/// security mitigation.
#[derive(Debug, Clone)]
pub struct AssemblyResult {
    /// Encoded `.text` section bytes (machine code).
    pub text: Vec<u8>,
    /// Relocations for `.rela.text`.
    pub relocations: Vec<RelocationEntry>,
    /// Symbol definitions from this function.
    pub symbols: Vec<AssemblerSymbol>,
    /// Encoded retpoline thunk bytes `(name, bytes)` if retpoline is enabled.
    pub retpoline_thunks: Vec<(String, Vec<u8>)>,
}

// ===========================================================================
// AssemblerContext — mutable state during assembly
// ===========================================================================

/// Tracks the mutable state of the x86-64 assembler during object file
/// construction.
///
/// `AssemblerContext` accumulates encoded bytes for each ELF section,
/// collects relocations and symbols, and resolves intra-function label
/// references.  It is created fresh for each assembly operation and
/// consumed when the final output is produced.
pub struct AssemblerContext {
    /// Accumulated `.text` section bytes (encoded machine instructions).
    pub text_section: Vec<u8>,
    /// Accumulated `.data` section bytes (initialized global data).
    pub data_section: Vec<u8>,
    /// `.bss` section size (uninitialized data — no bytes stored, just size).
    pub bss_size: usize,
    /// Accumulated `.rodata` section bytes (read-only data, string literals).
    pub rodata_section: Vec<u8>,
    /// Relocations collected during encoding (for `.rela.text`).
    pub relocations: Vec<RelocationEntry>,
    /// Symbol table entries.
    pub symbols: Vec<AssemblerSymbol>,
    /// Label-to-offset mapping within `.text` section.
    label_offsets: FxHashMap<String, usize>,
    /// Unresolved label references (label name → list of patch locations).
    unresolved_labels: FxHashMap<String, Vec<LabelFixup>>,
    /// Current offset within `.text` section.
    pub current_offset: usize,
    /// Whether PIC code generation is active (`-fPIC`).
    pub pic_enabled: bool,
    /// Security configuration for retpoline thunks and CET.
    security_config: SecurityConfig,
}

impl AssemblerContext {
    /// Create a new assembler context.
    ///
    /// # Arguments
    ///
    /// * `pic_enabled` — Whether `-fPIC` code generation is active.
    /// * `security_config` — Security mitigation settings (retpoline, CET,
    ///   stack probe).
    pub fn new(pic_enabled: bool, security_config: SecurityConfig) -> Self {
        AssemblerContext {
            text_section: Vec::with_capacity(4096),
            data_section: Vec::new(),
            bss_size: 0,
            rodata_section: Vec::new(),
            relocations: Vec::new(),
            symbols: Vec::new(),
            label_offsets: FxHashMap::default(),
            unresolved_labels: FxHashMap::default(),
            current_offset: 0,
            pic_enabled,
            security_config,
        }
    }

    /// Append a single byte to the `.text` section.
    #[inline]
    pub fn emit_byte(&mut self, byte: u8) {
        self.text_section.push(byte);
        self.current_offset += 1;
    }

    /// Append a byte slice to the `.text` section.
    #[inline]
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.text_section.extend_from_slice(bytes);
        self.current_offset += bytes.len();
    }

    /// Emit a 32-bit value in little-endian order to the `.text` section.
    #[inline]
    pub fn emit_u32_le(&mut self, value: u32) {
        self.text_section.extend_from_slice(&value.to_le_bytes());
        self.current_offset += 4;
    }

    /// Emit a 64-bit value in little-endian order to the `.text` section.
    #[inline]
    pub fn emit_u64_le(&mut self, value: u64) {
        self.text_section.extend_from_slice(&value.to_le_bytes());
        self.current_offset += 8;
    }

    /// Return the current byte offset within the `.text` section.
    #[inline]
    pub fn current_text_offset(&self) -> usize {
        self.current_offset
    }

    /// Define a label at the current `.text` offset.
    ///
    /// If any forward references to this label exist, they are immediately
    /// resolved by patching the encoded bytes.
    pub fn define_label(&mut self, name: &str) {
        let offset = self.current_offset;
        self.label_offsets.insert(name.to_string(), offset);

        // Resolve any pending fixups for this label.
        if let Some(fixups) = self.unresolved_labels.remove(name) {
            for fixup in fixups {
                self.apply_fixup(&fixup, offset);
            }
        }
    }

    /// Record a reference to a label.
    ///
    /// If the label is already defined, the fixup is applied immediately.
    /// Otherwise, the fixup is deferred until [`define_label`] is called.
    pub fn reference_label(&mut self, name: &str, fixup: LabelFixup) {
        if let Some(&target_offset) = self.label_offsets.get(name) {
            // Label already defined — patch immediately.
            self.apply_fixup(&fixup, target_offset);
        } else {
            // Label not yet defined — defer.
            self.unresolved_labels
                .entry(name.to_string())
                .or_default()
                .push(fixup);
        }
    }

    /// Append a relocation entry to the collection.
    #[inline]
    pub fn add_relocation(&mut self, reloc: RelocationEntry) {
        self.relocations.push(reloc);
    }

    /// Add a symbol to the assembler's symbol table.
    #[inline]
    pub fn add_symbol(&mut self, sym: AssemblerSymbol) {
        self.symbols.push(sym);
    }

    /// Resolve all remaining pending label fixups.
    ///
    /// After all instructions have been assembled, this method checks for
    /// any labels that were referenced but never defined.  Local labels
    /// that remain unresolved produce an error; external labels generate
    /// relocation entries for the linker.
    ///
    /// # Errors
    ///
    /// Returns an error message listing all unresolved local labels.
    pub fn resolve_pending_fixups(&mut self) -> Result<(), String> {
        if self.unresolved_labels.is_empty() {
            return Ok(());
        }

        let mut diag = DiagnosticEngine::new();
        // Drain unresolved labels — external references become relocations,
        // truly-local unresolved labels are errors.
        let unresolved: Vec<(String, Vec<LabelFixup>)> = self.unresolved_labels.drain().collect();

        for (label, fixups) in unresolved {
            // Labels starting with `.L` are local — must be defined within
            // this translation unit.
            if label.starts_with(".L") {
                diag.emit_error(
                    Span::dummy(),
                    format!("unresolved local label: '{}'", label),
                );
            } else {
                // External symbol — create relocation entries.
                for fixup in &fixups {
                    let rel_type = if fixup.pc_relative {
                        X86_64RelocationType::Pc32
                    } else {
                        X86_64RelocationType::Abs32
                    };
                    self.relocations.push(RelocationEntry::new(
                        fixup.offset as u64,
                        label.clone(),
                        rel_type,
                        fixup.addend,
                        ".text".to_string(),
                    ));
                }
            }
        }

        if diag.has_errors() {
            // Collect all error messages into a single string.
            let msgs: Vec<String> = diag
                .diagnostics()
                .iter()
                .map(|d| d.message.clone())
                .collect();
            Err(msgs.join("; "))
        } else {
            Ok(())
        }
    }

    // -- Private helpers --------------------------------------------------

    /// Apply a label fixup by patching the `.text` bytes at the fixup's
    /// offset with the computed value.
    fn apply_fixup(&mut self, fixup: &LabelFixup, target_offset: usize) {
        let value = if fixup.pc_relative {
            // PC-relative: target - (fixup_site + fixup_size) + addend
            let pc = fixup.offset + fixup.size as usize;
            (target_offset as i64) - (pc as i64) + fixup.addend
        } else {
            (target_offset as i64) + fixup.addend
        };

        match fixup.size {
            1 => {
                if fixup.offset < self.text_section.len() {
                    self.text_section[fixup.offset] = value as u8;
                }
            }
            4 => {
                let bytes = (value as i32).to_le_bytes();
                let end = fixup.offset + 4;
                if end <= self.text_section.len() {
                    self.text_section[fixup.offset..end].copy_from_slice(&bytes);
                }
            }
            8 => {
                let bytes = value.to_le_bytes();
                let end = fixup.offset + 8;
                if end <= self.text_section.len() {
                    self.text_section[fixup.offset..end].copy_from_slice(&bytes);
                }
            }
            _ => {
                // Unsupported fixup size — silently ignored (should not
                // happen with correct instruction encoding).
            }
        }
    }
}

// ===========================================================================
// assemble() — main assembly entry point
// ===========================================================================

/// Assemble a [`MachineFunction`] into encoded machine code.
///
/// This is the main entry point called from the x86-64 backend's
/// [`ArchCodegen::emit_assembly()`](crate::backend::traits::ArchCodegen)
/// implementation.
///
/// # Arguments
///
/// * `mf` — The register-allocated machine function from instruction
///   selection.
/// * `security_config` — Security mitigation settings (retpoline, CET).
/// * `pic_enabled` — Whether `-fPIC` code generation is active.
///
/// # Returns
///
/// An [`AssemblyResult`] containing the encoded `.text` bytes,
/// collected relocations, symbol definitions, and retpoline thunk data.
pub fn assemble(
    mf: &MachineFunction,
    security_config: &SecurityConfig,
    pic_enabled: bool,
) -> AssemblyResult {
    let mut ctx = AssemblerContext::new(pic_enabled, *security_config);
    let func_start = ctx.current_offset;

    // Define the function symbol at the start of its code.
    ctx.add_symbol(AssemblerSymbol {
        name: mf.name.clone(),
        offset: func_start as u64,
        size: 0, // Updated after encoding completes.
        section: SymbolSection::Text,
        binding: SymbolBinding::Global,
        sym_type: SymbolType::Function,
        visibility: SymbolVisibility::Default,
    });

    // NOTE: endbr64 is already emitted by the prologue in X86_64Backend::emit_prologue()
    // when cf_protection is enabled. Adding it here again would produce a duplicate.
    // The prologue-based insertion is the correct location since it becomes part of
    // the function's MachineInstruction sequence and is visible in -S output.

    // Create the encoder starting at the current text offset.
    let mut enc = X86_64Encoder::new(ctx.current_offset);
    enc.pic_enabled = ctx.pic_enabled;

    // Iterate over all basic blocks and their instructions.
    for (block_idx, block) in mf.blocks.iter().enumerate() {
        // Define a .L{index} label for this block so that BlockLabel(idx)
        // operands in branch instructions resolve correctly.
        ctx.define_label(&format!(".L{}", block_idx));

        // Also define the human-readable label if present (for debug output
        // and any string-based references).
        if let Some(ref label) = block.label {
            ctx.define_label(label);
        }

        for inst in &block.instructions {
            // If the instruction has pre-encoded bytes (e.g., from the
            // security mitigation pass), use them directly.
            if !inst.encoded_bytes.is_empty() {
                ctx.emit_bytes(&inst.encoded_bytes);
                enc.current_offset = ctx.current_offset;
                continue;
            }

            // Handle inline assembly instructions by assembling the template
            // directly.  Template placeholders (%0, %1, …) are substituted
            // with physical register names:
            //   %0..%(num_outputs-1)  → inst.result  (output register)
            //   %num_outputs..        → inst.operands[i - num_outputs]
            if let Some(ref template) = inst.asm_template {
                let asm_bytes = assemble_inline_asm_x86_64(
                    template,
                    inst.result.as_ref(),
                    inst.asm_num_outputs,
                    &inst.operands,
                );
                ctx.emit_bytes(&asm_bytes);
                enc.current_offset = ctx.current_offset;
                continue;
            }

            // Handle InternalLabelDef pseudo-instruction: define a local
            // label at the current text offset.  Emits zero bytes.
            if inst.opcode == X86Opcode::InternalLabelDef.as_u32() {
                if let Some(MachineOperand::GlobalSymbol(ref name)) = inst.operands.first() {
                    ctx.define_label(name);
                }
                continue;
            }

            // Dispatch to the encoder for instruction-level binary encoding.
            let encoded = enc.encode_instruction(inst);

            // Append the encoded bytes to the text section.
            ctx.text_section.extend_from_slice(&encoded.bytes);
            ctx.current_offset += encoded.bytes.len();
            // (enc.current_offset is already updated by encode_instruction.)

            // Collect any relocations generated during encoding.
            for reloc in encoded.relocations {
                ctx.add_relocation(reloc);
            }
        }
    }

    // Resolve all pending label fixups (forward references).
    let _ = ctx.resolve_pending_fixups();

    // Resolve local .L labels in the relocation list by patching the text
    // section directly. The encoder emits RelocationEntry objects for
    // BlockLabel operands, but .L labels are function-local and must be
    // resolved during assembly rather than deferred to the linker.
    let mut remaining_relocs = Vec::new();
    for reloc in ctx.relocations.drain(..) {
        if reloc.symbol.starts_with(".L") {
            if let Some(&target_offset) = ctx.label_offsets.get(&reloc.symbol) {
                let fixup_offset = reloc.offset as usize;
                // Standard ELF R_X86_64_PC32 formula: S + A - P
                // S = target_offset, A = reloc.addend (typically -4), P = fixup_offset
                let value = (target_offset as i64) + reloc.addend - (fixup_offset as i64);
                let bytes = (value as i32).to_le_bytes();
                let end = fixup_offset + 4;
                if end <= ctx.text_section.len() {
                    ctx.text_section[fixup_offset..end].copy_from_slice(&bytes);
                }
            } else {
                // Label not found — keep as relocation for the linker.
                remaining_relocs.push(reloc);
            }
        } else {
            remaining_relocs.push(reloc);
        }
    }
    ctx.relocations = remaining_relocs;

    // Compute the function's total code size.
    let func_size = ctx.current_offset - func_start;
    if let Some(sym) = ctx.symbols.first_mut() {
        sym.size = func_size as u64;
    }

    // Generate retpoline thunks if enabled.
    let retpoline_thunks = if ctx.security_config.retpoline {
        assemble_retpoline_thunks()
    } else {
        Vec::new()
    };

    AssemblyResult {
        text: ctx.text_section,
        relocations: ctx.relocations,
        symbols: ctx.symbols,
        retpoline_thunks,
    }
}

// ===========================================================================
// assemble_to_object() — complete ELF .o file generation
// ===========================================================================

/// Assemble one or more functions into a complete relocatable ELF `.o` file.
///
/// This higher-level function orchestrates the assembly of multiple
/// [`MachineFunction`]s and produces a valid ELF relocatable object file
/// containing all required sections (`.text`, `.data`, `.bss`, `.rodata`,
/// `.rela.text`, `.symtab`, `.strtab`).
///
/// # Arguments
///
/// * `functions` — List of machine functions to assemble.
/// * `global_data` — Global variable data for the `.data` section.
/// * `rodata` — Read-only data (string literals, etc.) for `.rodata`.
/// * `bss_size` — Size of uninitialized data (`.bss` section).
/// * `target` — Target configuration (must be [`Target::X86_64`]).
/// * `security_config` — Security mitigation settings.
/// * `pic_enabled` — Whether `-fPIC` is active.
///
/// # Returns
///
/// Complete ELF `.o` file bytes suitable for consumption by the built-in
/// linker or external tools (`readelf`, `objdump`).
///
/// # Errors
///
/// Returns an error message if assembly or ELF construction fails.
pub fn assemble_to_object(
    functions: &[&MachineFunction],
    global_data: &[(String, Vec<u8>)],
    rodata: &[(String, Vec<u8>)],
    bss_size: usize,
    target: &Target,
    security_config: &SecurityConfig,
    pic_enabled: bool,
) -> Result<Vec<u8>, String> {
    // Validate target architecture.
    debug_assert!(
        *target == Target::X86_64,
        "x86-64 assembler invoked for non-x86-64 target"
    );

    let mut combined_text: Vec<u8> = Vec::with_capacity(4096);
    let mut combined_relocs: Vec<RelocationEntry> = Vec::new();
    let mut combined_symbols: Vec<AssemblerSymbol> = Vec::new();
    let mut retpoline_needed = false;

    // Assemble each function and accumulate results.
    for mf in functions {
        let base_offset = combined_text.len();
        let result = assemble(mf, security_config, pic_enabled);

        // Adjust relocation offsets by the function's position in the
        // combined text section.
        for mut reloc in result.relocations {
            reloc.offset += base_offset as u64;
            combined_relocs.push(reloc);
        }

        // Adjust symbol offsets.
        for mut sym in result.symbols {
            sym.offset += base_offset as u64;
            combined_symbols.push(sym);
        }

        combined_text.extend_from_slice(&result.text);

        if !result.retpoline_thunks.is_empty() {
            retpoline_needed = true;
        }
    }

    // Append retpoline thunks to the text section if any function needed them.
    if retpoline_needed && security_config.retpoline {
        let thunks = assemble_retpoline_thunks();
        for (thunk_name, thunk_bytes) in &thunks {
            let thunk_offset = combined_text.len();
            combined_text.extend_from_slice(thunk_bytes);
            combined_symbols.push(AssemblerSymbol {
                name: thunk_name.clone(),
                offset: thunk_offset as u64,
                size: thunk_bytes.len() as u64,
                section: SymbolSection::Text,
                binding: SymbolBinding::Global,
                sym_type: SymbolType::Function,
                visibility: SymbolVisibility::Hidden,
            });
        }
    }

    // Build data section from global_data entries.
    let mut data_section_bytes: Vec<u8> = Vec::new();
    for (name, data) in global_data {
        let data_offset = data_section_bytes.len();
        data_section_bytes.extend_from_slice(data);
        combined_symbols.push(AssemblerSymbol {
            name: name.clone(),
            offset: data_offset as u64,
            size: data.len() as u64,
            section: SymbolSection::Data,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
        });
    }

    // Build rodata section from rodata entries.
    let mut rodata_section_bytes: Vec<u8> = Vec::new();
    for (name, data) in rodata {
        let ro_offset = rodata_section_bytes.len();
        rodata_section_bytes.extend_from_slice(data);
        combined_symbols.push(AssemblerSymbol {
            name: name.clone(),
            offset: ro_offset as u64,
            size: data.len() as u64,
            section: SymbolSection::Rodata,
            binding: SymbolBinding::Local,
            sym_type: SymbolType::Object,
            visibility: SymbolVisibility::Default,
        });
    }

    // ---------------------------------------------------------------
    // Build the ELF .o file
    // ---------------------------------------------------------------
    let mut elf = ElfWriter::new(*target, ET_REL);

    // Section indices (1-based; index 0 is the null section).
    let text_idx = elf.add_section(Section {
        name: ".text".to_string(),
        sh_type: SHT_PROGBITS,
        sh_flags: SHF_ALLOC | SHF_EXECINSTR,
        data: combined_text,
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 16,
        sh_entsize: 0,
        logical_size: 0,
        virtual_address: 0,
        file_offset_hint: 0,
    });

    let mut data_idx: Option<usize> = None;
    if !data_section_bytes.is_empty() {
        data_idx = Some(elf.add_section(Section {
            name: ".data".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: data_section_bytes,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        }));
    }

    let mut bss_idx: Option<usize> = None;
    if bss_size > 0 {
        // SHT_NOBITS sections occupy no file space — use empty data vec
        // and set logical_size to the actual BSS size.  This avoids
        // wastefully allocating `bss_size` bytes of zeroes in memory.
        bss_idx = Some(elf.add_section(Section {
            name: ".bss".to_string(),
            sh_type: SHT_NOBITS,
            sh_flags: SHF_ALLOC | SHF_WRITE,
            data: Vec::new(),
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: bss_size as u64,
            virtual_address: 0,
            file_offset_hint: 0,
        }));
    }

    let mut rodata_idx: Option<usize> = None;
    if !rodata_section_bytes.is_empty() {
        rodata_idx = Some(elf.add_section(Section {
            name: ".rodata".to_string(),
            sh_type: SHT_PROGBITS,
            sh_flags: SHF_ALLOC,
            data: rodata_section_bytes,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        }));
    }

    // Pre-build a name→ELF-symbol-index map for relocations.
    //
    // We compute the base symbol index by counting how many section symbols
    // will be emitted before the assembled symbols.  ELF symbol indices are:
    //   0         = null symbol
    //   1         = .text section symbol (always present)
    //   +1 if .data present
    //   +1 if .rodata present
    //   then assembled symbols in order.
    let section_sym_count: u32 = 1 // .text
        + if data_idx.is_some() { 1 } else { 0 }
        + if rodata_idx.is_some() { 1 } else { 0 };
    let base_sym_index = 1 + section_sym_count; // +1 for null symbol at index 0
    let mut symbol_name_to_index: FxHashMap<String, u32> = FxHashMap::default();
    for (i, sym) in combined_symbols.iter().enumerate() {
        symbol_name_to_index.insert(sym.name.clone(), base_sym_index + i as u32);
    }

    // Add .rela.text section with all collected relocations.
    if !combined_relocs.is_empty() {
        let mut rela_data = Vec::with_capacity(combined_relocs.len() * 24);
        for reloc in &combined_relocs {
            // Look up the actual symbol table index from the name→index map
            // built during symbol emission.  If the symbol is not found (e.g.,
            // an external/undefined symbol that hasn't been added yet), fall
            // back to 0 (the null symbol) — the linker will resolve it.
            let sym_idx = symbol_name_to_index
                .get(&reloc.symbol)
                .copied()
                .unwrap_or(0);
            let elf_reloc = reloc.to_elf_relocation(sym_idx);
            // Elf64_Rela: r_offset (8 bytes) + r_info (8 bytes) + r_addend (8 bytes)
            rela_data.extend_from_slice(&elf_reloc.offset.to_le_bytes());
            let r_info = ((elf_reloc.sym_index as u64) << 32) | (elf_reloc.rel_type as u64);
            rela_data.extend_from_slice(&r_info.to_le_bytes());
            rela_data.extend_from_slice(&elf_reloc.addend.to_le_bytes());
        }
        elf.add_section(Section {
            name: ".rela.text".to_string(),
            sh_type: SHT_RELA,
            sh_flags: 0,
            data: rela_data,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 24, // sizeof(Elf64_Rela)
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });
    }

    // If CET is enabled, add .note.gnu.property section.
    if security_config.cf_protection {
        let cet_note = generate_cet_note_section();
        elf.add_section(Section {
            name: ".note.gnu.property".to_string(),
            sh_type: SHT_NOTE,
            sh_flags: SHF_ALLOC,
            data: cet_note,
            sh_link: 0,
            sh_info: 0,
            sh_addralign: 8,
            sh_entsize: 0,
            logical_size: 0,
            virtual_address: 0,
            file_offset_hint: 0,
        });
    }

    // Add .note.GNU-stack section — non-executable stack marker.
    elf.add_section(Section {
        name: ".note.GNU-stack".to_string(),
        sh_type: SHT_PROGBITS,
        sh_flags: 0, // No SHF_EXECINSTR = non-executable stack.
        data: Vec::new(),
        sh_link: 0,
        sh_info: 0,
        sh_addralign: 1,
        sh_entsize: 0,
        logical_size: 0,
        virtual_address: 0,
        file_offset_hint: 0,
    });

    // -- Section symbols --------------------------------------------------

    elf.add_symbol(ElfSymbol {
        name: String::new(),
        value: 0,
        size: 0,
        binding: STB_LOCAL,
        sym_type: STT_SECTION,
        visibility: STV_DEFAULT,
        section_index: text_idx as u16,
    });

    if let Some(idx) = data_idx {
        elf.add_symbol(ElfSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_SECTION,
            visibility: STV_DEFAULT,
            section_index: idx as u16,
        });
    }

    if let Some(idx) = rodata_idx {
        elf.add_symbol(ElfSymbol {
            name: String::new(),
            value: 0,
            size: 0,
            binding: STB_LOCAL,
            sym_type: STT_SECTION,
            visibility: STV_DEFAULT,
            section_index: idx as u16,
        });
    }

    // -- Assembled symbols ------------------------------------------------
    // Add all assembled symbols to the ELF symbol table.  The name→index
    // map (`symbol_name_to_index`) was already built above for relocations.
    for sym in combined_symbols.iter() {
        let section_index = match sym.section {
            SymbolSection::Text => text_idx as u16,
            SymbolSection::Data => data_idx.unwrap_or(0) as u16,
            SymbolSection::Bss => bss_idx.unwrap_or(0) as u16,
            SymbolSection::Rodata => rodata_idx.unwrap_or(0) as u16,
            SymbolSection::Undefined => 0,        // SHN_UNDEF
            SymbolSection::Absolute => 0xFFF1u16, // SHN_ABS
        };

        elf.add_symbol(ElfSymbol {
            name: sym.name.clone(),
            value: sym.offset,
            size: sym.size,
            binding: sym.binding.to_elf(),
            sym_type: sym.sym_type.to_elf(),
            visibility: sym.visibility.to_elf(),
            section_index,
        });
    }

    // Serialize the complete ELF binary.
    Ok(elf.write())
}

// ===========================================================================
// Retpoline Thunk Assembly
// ===========================================================================

/// Assemble retpoline thunks for all 16 x86-64 general-purpose registers.
///
/// Returns a vector of `(symbol_name, encoded_bytes)` tuples, one per GPR
/// (RAX through R15).  Each thunk implements the retpoline speculation
/// barrier pattern:
///
/// ```text
/// __x86_indirect_thunk_<reg>:
///     call .Lretpoline_call_target    ; E8 07 00 00 00
/// .Lretpoline_capture:
///     pause                           ; F3 90
///     lfence                          ; 0F AE E8
///     jmp .Lretpoline_capture         ; EB F9
/// .Lretpoline_call_target:
///     mov [rsp], <reg>                ; REX 89 <ModR/M> 24
///     ret                             ; C3
/// ```
pub fn assemble_retpoline_thunks() -> Vec<(String, Vec<u8>)> {
    let gprs: [u16; 16] = [
        RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
    ];

    let mut thunks = Vec::with_capacity(16);

    for &reg in &gprs {
        let name = format!("__x86_indirect_thunk_{}", reg_name_64(reg));
        let mut bytes = Vec::with_capacity(17);

        // call .Lretpoline_call_target (relative offset = +7)
        //   IP after call = current + 5, target = current + 12, rel32 = 7
        bytes.push(0xE8);
        bytes.extend_from_slice(&7i32.to_le_bytes());

        // .Lretpoline_capture:
        // pause = F3 90
        bytes.push(0xF3);
        bytes.push(0x90);

        // lfence = 0F AE E8
        bytes.push(0x0F);
        bytes.push(0xAE);
        bytes.push(0xE8);

        // jmp .Lretpoline_capture
        //   IP after jmp = byte 12, target = byte 5, rel8 = 5 - 12 = -7
        bytes.push(0xEB);
        bytes.push((-7i8) as u8);

        // .Lretpoline_call_target:
        // mov [rsp], <reg>
        let hw = hw_encoding(reg);
        // REX.W = 0x48; REX.R = 0x04 when hw >= 8 (extended register).
        let rex = 0x48u8 | if hw >= 8 { 0x04 } else { 0x00 };
        // ModR/M: mod=00, reg=<lower 3 bits>, r/m=100 (SIB follows).
        let modrm = ((hw & 0x07) << 3) | 0x04;
        // SIB: base=RSP(100), index=none(100), scale=0.
        let sib = 0x24u8;
        bytes.push(rex);
        bytes.push(0x89); // MOV r/m64, r64
        bytes.push(modrm);
        bytes.push(sib);

        // ret = C3
        bytes.push(0xC3);

        debug_assert_eq!(bytes.len(), 17, "retpoline thunk must be 17 bytes");
        thunks.push((name, bytes));
    }

    thunks
}

// ===========================================================================
// Inline Assembly — x86-64 AT&T Syntax Assembler
// ===========================================================================

/// Assemble an x86-64 inline assembly template into machine code bytes.
///
/// The template uses AT&T syntax with `%%reg` for literal registers, `%N`
/// placeholders already substituted with physical register names, and `$imm`
/// for immediates.  Each line (separated by `\n`, `\n\t`, or `;`) is
/// assembled independently.
///
/// # Arguments
///
/// * `template` - The inline assembly template string with `%0`, `%1`, …
///   placeholders.
/// * `result_op` - The physical register assigned to the asm result (outputs).
/// * `num_outputs` - How many output operands exist. Template `%0`..`%(num_outputs-1)`
///   map to `result_op`.
/// * `input_operands` - Physical register operands for inputs (from register
///   allocation). Template `%num_outputs`.. maps to `input_operands[i - num_outputs]`.
fn assemble_inline_asm_x86_64(
    template: &str,
    result_op: Option<&crate::backend::traits::MachineOperand>,
    num_outputs: usize,
    input_operands: &[crate::backend::traits::MachineOperand],
) -> Vec<u8> {
    let mut result = Vec::new();

    // First, protect %% escape sequences from being matched as operand refs.
    // Replace %% with a placeholder that won't conflict with %N patterns.
    let template = template.replace("%%", "\x01\x01");

    // Substitute %N operand references with physical register names or
    // memory operand syntax.  Process HIGHER indices first to avoid %1
    // matching inside %10.
    let total_operands = num_outputs + input_operands.len();
    let mut substituted = template.to_string();
    for idx in (0..total_operands).rev() {
        let placeholder = format!("%{}", idx);
        let operand_text = if idx < num_outputs {
            // Output operand → use result register or memory operand
            match result_op {
                Some(crate::backend::traits::MachineOperand::Register(r)) => x86_64_reg_name_32(*r),
                Some(crate::backend::traits::MachineOperand::Memory {
                    base, displacement, ..
                }) => format_memory_operand_att(*base, *displacement),
                _ => format!("%{}", idx),
            }
        } else {
            // Input operand → use input_operands[idx - num_outputs]
            let input_idx = idx - num_outputs;
            if input_idx < input_operands.len() {
                match &input_operands[input_idx] {
                    crate::backend::traits::MachineOperand::Register(r) => x86_64_reg_name_32(*r),
                    crate::backend::traits::MachineOperand::Immediate(imm) => {
                        format!("${}", imm)
                    }
                    crate::backend::traits::MachineOperand::Memory {
                        base, displacement, ..
                    } => format_memory_operand_att(*base, *displacement),
                    _ => format!("%{}", idx),
                }
            } else {
                format!("%{}", idx)
            }
        };
        substituted = substituted.replace(&placeholder, &operand_text);
    }

    // Restore %% escapes to single %.
    substituted = substituted.replace("\x01\x01", "%");

    // Split into individual instructions.
    for raw_line in substituted.split('\n') {
        let line = raw_line.trim().trim_end_matches('\t');
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Split on ';' for multiple instructions per line.
        for sub_line in line.split(';') {
            let sub_line = sub_line.trim();
            if sub_line.is_empty() {
                continue;
            }
            result.extend(assemble_single_x86_64_instruction(sub_line));
        }
    }

    result
}

/// Get the 32-bit register name for a physical register number.
fn x86_64_reg_name_32(reg: u16) -> String {
    match reg {
        0 => "%eax".to_string(),
        1 => "%ecx".to_string(),
        2 => "%edx".to_string(),
        3 => "%ebx".to_string(),
        4 => "%esp".to_string(),
        5 => "%ebp".to_string(),
        6 => "%esi".to_string(),
        7 => "%edi".to_string(),
        8 => "%r8d".to_string(),
        9 => "%r9d".to_string(),
        10 => "%r10d".to_string(),
        11 => "%r11d".to_string(),
        12 => "%r12d".to_string(),
        13 => "%r13d".to_string(),
        14 => "%r14d".to_string(),
        15 => "%r15d".to_string(),
        _ => format!("%r{}", reg),
    }
}

/// Assemble a single AT&T syntax x86-64 instruction into machine code bytes.
///
/// Supports a subset of x86-64 instructions commonly used in inline assembly:
/// - `movl`, `movq` — register-register and immediate-register moves
/// - `syscall` — system call instruction
/// - `nop` — no operation
/// - `int $N` — software interrupt
/// - `xorl`, `xorq` — XOR
/// - `addl`, `addq` — ADD
/// - `subl`, `subq` — SUB
/// - `ret` — return
/// - `hlt` — halt
fn assemble_single_x86_64_instruction(line: &str) -> Vec<u8> {
    let line = line.trim();

    // Parse mnemonic and operands.
    let (mnemonic, operands_str) = if let Some(pos) = line.find(|c: char| c.is_whitespace()) {
        let m = &line[..pos];
        let rest = line[pos..].trim();
        (m, rest)
    } else {
        (line, "")
    };

    match mnemonic {
        "syscall" => vec![0x0F, 0x05],
        "nop" => vec![0x90],
        "ret" | "retq" => vec![0xC3],
        "hlt" => vec![0xF4],
        "ud2" => vec![0x0F, 0x0B],
        "cld" => vec![0xFC],
        "std" => vec![0xFD],
        "cli" => vec![0xFA],
        "sti" => vec![0xFB],
        "int" => {
            // int $N
            if let Some(imm) = parse_att_immediate(operands_str) {
                vec![0xCD, imm as u8]
            } else {
                vec![0xCC] // int3 fallback
            }
        }
        "movl" => assemble_mov_l(operands_str),
        "movq" => assemble_mov_q(operands_str),
        "xorl" => assemble_alu_l(0x31, operands_str), // XOR r/m32, r32
        "xorq" => assemble_alu_q(0x31, operands_str),
        "addl" => assemble_alu_l(0x01, operands_str), // ADD r/m32, r32
        "addq" => assemble_alu_q(0x01, operands_str),
        "subl" => assemble_alu_l(0x29, operands_str), // SUB r/m32, r32
        "subq" => assemble_alu_q(0x29, operands_str),
        "andl" => assemble_alu_l(0x21, operands_str),
        "andq" => assemble_alu_q(0x21, operands_str),
        "orl" => assemble_alu_l(0x09, operands_str),
        "orq" => assemble_alu_q(0x09, operands_str),
        "cmpl" => assemble_alu_l(0x39, operands_str),
        "cmpq" => assemble_alu_q(0x39, operands_str),
        "testl" => assemble_alu_l(0x85, operands_str),
        "testq" => assemble_alu_q(0x85, operands_str),
        "pushl" | "pushq" => assemble_push(operands_str),
        "popl" | "popq" => assemble_pop(operands_str),
        _ => {
            // Unknown instruction — emit as NOP with a debug warning.
            // This prevents crashes for unsupported inline asm instructions.
            if crate::common::verbosity::verbose_enabled() {
                eprintln!(
                    "[inline-asm] WARNING: unsupported instruction '{}', emitting NOP",
                    line
                );
            }
            vec![0x90]
        }
    }
}

/// Parse an AT&T immediate operand like `$60` → `Some(60)`.
fn parse_att_immediate(s: &str) -> Option<i64> {
    let s = s.trim();
    let s = s.strip_prefix('$')?;
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<i64>().ok()
    }
}

/// Parse AT&T register name to register number (0-15).
fn parse_att_register(s: &str) -> Option<u8> {
    let s = s.trim().strip_prefix('%')?;
    match s {
        "eax" | "rax" | "al" | "ax" => Some(0),
        "ecx" | "rcx" | "cl" | "cx" => Some(1),
        "edx" | "rdx" | "dl" | "dx" => Some(2),
        "ebx" | "rbx" | "bl" | "bx" => Some(3),
        "esp" | "rsp" | "sp" => Some(4),
        "ebp" | "rbp" | "bp" => Some(5),
        "esi" | "rsi" | "si" | "sil" => Some(6),
        "edi" | "rdi" | "di" | "dil" => Some(7),
        "r8d" | "r8" | "r8b" | "r8w" => Some(8),
        "r9d" | "r9" | "r9b" | "r9w" => Some(9),
        "r10d" | "r10" | "r10b" | "r10w" => Some(10),
        "r11d" | "r11" | "r11b" | "r11w" => Some(11),
        "r12d" | "r12" | "r12b" | "r12w" => Some(12),
        "r13d" | "r13" | "r13b" | "r13w" => Some(13),
        "r14d" | "r14" | "r14b" | "r14w" => Some(14),
        "r15d" | "r15" | "r15b" | "r15w" => Some(15),
        _ => None,
    }
}

/// Format a `MachineOperand::Memory` as AT&T syntax (e.g., `-8(%rbp)`).
fn format_memory_operand_att(base: Option<u16>, displacement: i64) -> String {
    match base {
        Some(r) => {
            let base_name = reg_name_64(r);
            if displacement != 0 {
                format!("{}(%{})", displacement, base_name)
            } else {
                format!("(%{})", base_name)
            }
        }
        None => {
            // Absolute address (rare, but handle it)
            format!("{}", displacement)
        }
    }
}

/// Parse an AT&T memory operand of the form `disp(%reg)` or `(%reg)`.
///
/// Returns `(base_register_number, displacement)` on success.
fn parse_att_memory(s: &str) -> Option<(u8, i32)> {
    let s = s.trim();
    let paren_pos = s.find('(')?;
    let disp_str = &s[..paren_pos];
    let reg_part = &s[paren_pos + 1..];
    let reg_str = reg_part.trim_end_matches(')');
    let disp: i32 = if disp_str.is_empty() {
        0
    } else {
        disp_str.parse::<i32>().ok()?
    };
    let reg = parse_att_register(reg_str)?;
    Some((reg, disp))
}

/// Encode ModR/M (and optional SIB + displacement) for a memory operand
/// with the given base register and displacement.
///
/// `reg_opext` is the register field / opcode extension (/0../7) placed in
/// the reg bits of the ModR/M byte.
fn encode_modrm_memory(bytes: &mut Vec<u8>, reg_opext: u8, base_reg: u8, disp: i32) {
    let rm = base_reg & 7;

    if rm == 4 {
        // RSP/R12 (rm=100) requires a SIB byte
        if disp == 0 && rm != 5 {
            // mod=00, rm=100 (SIB follows)
            bytes.push(((reg_opext & 7) << 3) | 4);
            bytes.push(0x24); // SIB: scale=00, index=100 (none), base=100 (RSP)
        } else if (-128..=127).contains(&disp) {
            // mod=01, rm=100 (SIB follows), disp8
            bytes.push(0x40 | ((reg_opext & 7) << 3) | 4);
            bytes.push(0x24);
            bytes.push(disp as u8);
        } else {
            // mod=10, rm=100 (SIB follows), disp32
            bytes.push(0x80 | ((reg_opext & 7) << 3) | 4);
            bytes.push(0x24);
            bytes.extend_from_slice(&disp.to_le_bytes());
        }
    } else if rm == 5 && disp == 0 {
        // RBP/R13 (rm=101) with zero disp needs explicit disp8=0 (mod=01)
        // because mod=00 rm=101 means RIP-relative, not [RBP].
        bytes.push(0x40 | ((reg_opext & 7) << 3) | 5);
        bytes.push(0x00);
    } else if disp == 0 {
        // mod=00 (no displacement)
        bytes.push(((reg_opext & 7) << 3) | rm);
    } else if (-128..=127).contains(&disp) {
        // mod=01 (disp8)
        bytes.push(0x40 | ((reg_opext & 7) << 3) | rm);
        bytes.push(disp as u8);
    } else {
        // mod=10 (disp32)
        bytes.push(0x80 | ((reg_opext & 7) << 3) | rm);
        bytes.extend_from_slice(&disp.to_le_bytes());
    }
}

/// Split AT&T operands (src, dst) with comma separator.
fn split_att_operands(s: &str) -> (&str, &str) {
    if let Some(pos) = s.find(',') {
        (s[..pos].trim(), s[pos + 1..].trim())
    } else {
        (s.trim(), "")
    }
}

/// Encode ModR/M byte for two register operands.
fn modrm_rr(reg: u8, rm: u8) -> u8 {
    0xC0 | ((reg & 7) << 3) | (rm & 7)
}

/// Assemble `movl` (32-bit move) in AT&T syntax.
fn assemble_mov_l(operands: &str) -> Vec<u8> {
    let (src, dst) = split_att_operands(operands);

    // movl $imm, disp(%reg) — store immediate to memory
    if let Some(imm) = parse_att_immediate(src) {
        if let Some((base_reg, disp)) = parse_att_memory(dst) {
            let mut bytes = Vec::new();
            if base_reg >= 8 {
                bytes.push(0x41); // REX.B
            }
            bytes.push(0xC7); // MOV r/m32, imm32
            encode_modrm_memory(&mut bytes, 0, base_reg, disp);
            bytes.extend_from_slice(&(imm as u32).to_le_bytes());
            return bytes;
        }
    }

    // movl $imm, %reg — move immediate to register
    if let Some(imm) = parse_att_immediate(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let mut bytes = Vec::new();
            if dst_reg >= 8 {
                bytes.push(0x41); // REX.B
            }
            bytes.push(0xB8 + (dst_reg & 7));
            bytes.extend_from_slice(&(imm as u32).to_le_bytes());
            return bytes;
        }
    }

    // movl %reg, disp(%reg) — store register to memory
    if let Some(src_reg) = parse_att_register(src) {
        if let Some((base_reg, disp)) = parse_att_memory(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0u8;
            if src_reg >= 8 {
                rex |= 0x44; // REX.R
            }
            if base_reg >= 8 {
                rex |= 0x41; // REX.B
            }
            if rex != 0 {
                bytes.push(rex);
            }
            bytes.push(0x89); // MOV r/m32, r32
            encode_modrm_memory(&mut bytes, src_reg, base_reg, disp);
            return bytes;
        }
    }

    // movl disp(%reg), %reg — load from memory to register
    if let Some((base_reg, disp)) = parse_att_memory(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0u8;
            if dst_reg >= 8 {
                rex |= 0x44; // REX.R
            }
            if base_reg >= 8 {
                rex |= 0x41; // REX.B
            }
            if rex != 0 {
                bytes.push(rex);
            }
            bytes.push(0x8B); // MOV r32, r/m32
            encode_modrm_memory(&mut bytes, dst_reg, base_reg, disp);
            return bytes;
        }
    }

    // movl %reg, %reg — register-to-register
    if let (Some(src_reg), Some(dst_reg)) = (parse_att_register(src), parse_att_register(dst)) {
        let mut bytes = Vec::new();
        let mut rex = 0u8;
        if src_reg >= 8 {
            rex |= 0x44; // REX.R
        }
        if dst_reg >= 8 {
            rex |= 0x41; // REX.B
        }
        if rex != 0 {
            bytes.push(rex);
        }
        bytes.push(0x89); // MOV r/m32, r32
        bytes.push(modrm_rr(src_reg, dst_reg));
        return bytes;
    }

    if crate::common::verbosity::verbose_enabled() {
        eprintln!(
            "[inline-asm] WARNING: unsupported movl operands: {}",
            operands
        );
    }
    vec![0x90]
}

/// Assemble `movq` (64-bit move) in AT&T syntax.
fn assemble_mov_q(operands: &str) -> Vec<u8> {
    let (src, dst) = split_att_operands(operands);

    // movq $imm, disp(%reg) — store immediate to memory (64-bit context)
    if let Some(imm) = parse_att_immediate(src) {
        if let Some((base_reg, disp)) = parse_att_memory(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0x48u8; // REX.W
            if base_reg >= 8 {
                rex |= 0x01; // REX.B
            }
            bytes.push(rex);
            bytes.push(0xC7); // MOV r/m64, imm32 (sign-extended to 64-bit)
            encode_modrm_memory(&mut bytes, 0, base_reg, disp);
            bytes.extend_from_slice(&(imm as i32).to_le_bytes());
            return bytes;
        }
    }

    // movq $imm, %reg
    if let Some(imm) = parse_att_immediate(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0x48u8; // REX.W
            if dst_reg >= 8 {
                rex |= 0x01; // REX.B
            }
            bytes.push(rex);
            if imm >= i32::MIN as i64 && imm <= i32::MAX as i64 {
                // movq $imm32, %reg (sign-extended)
                bytes.push(0xC7);
                bytes.push(modrm_rr(0, dst_reg));
                bytes.extend_from_slice(&(imm as i32).to_le_bytes());
            } else {
                // movabsq $imm64, %reg
                bytes.push(0xB8 + (dst_reg & 7));
                bytes.extend_from_slice(&(imm as u64).to_le_bytes());
            }
            return bytes;
        }
    }

    // movq %reg, disp(%reg) — store register to memory
    if let Some(src_reg) = parse_att_register(src) {
        if let Some((base_reg, disp)) = parse_att_memory(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0x48u8; // REX.W
            if src_reg >= 8 {
                rex |= 0x04; // REX.R
            }
            if base_reg >= 8 {
                rex |= 0x01; // REX.B
            }
            bytes.push(rex);
            bytes.push(0x89); // MOV r/m64, r64
            encode_modrm_memory(&mut bytes, src_reg, base_reg, disp);
            return bytes;
        }
    }

    // movq disp(%reg), %reg — load from memory to register
    if let Some((base_reg, disp)) = parse_att_memory(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let mut bytes = Vec::new();
            let mut rex = 0x48u8; // REX.W
            if dst_reg >= 8 {
                rex |= 0x04; // REX.R
            }
            if base_reg >= 8 {
                rex |= 0x01; // REX.B
            }
            bytes.push(rex);
            bytes.push(0x8B); // MOV r64, r/m64
            encode_modrm_memory(&mut bytes, dst_reg, base_reg, disp);
            return bytes;
        }
    }

    // movq %reg, %reg — register-to-register
    if let (Some(src_reg), Some(dst_reg)) = (parse_att_register(src), parse_att_register(dst)) {
        let mut bytes = Vec::new();
        let mut rex = 0x48u8; // REX.W
        if src_reg >= 8 {
            rex |= 0x04; // REX.R
        }
        if dst_reg >= 8 {
            rex |= 0x01; // REX.B
        }
        bytes.push(rex);
        bytes.push(0x89); // MOV r/m64, r64
        bytes.push(modrm_rr(src_reg, dst_reg));
        return bytes;
    }

    if crate::common::verbosity::verbose_enabled() {
        eprintln!(
            "[inline-asm] WARNING: unsupported movq operands: {}",
            operands
        );
    }
    vec![0x90]
}

/// Assemble a 32-bit ALU instruction (addl, subl, xorl, etc.) in AT&T syntax.
fn assemble_alu_l(opcode: u8, operands: &str) -> Vec<u8> {
    let (src, dst) = split_att_operands(operands);

    // ALU $imm, %reg — use opcode group /N with 0x83 or 0x81
    if let Some(imm) = parse_att_immediate(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let alu_group = match opcode {
                0x01 => 0u8, // ADD
                0x09 => 1,   // OR
                0x21 => 4,   // AND
                0x29 => 5,   // SUB
                0x31 => 6,   // XOR
                0x39 => 7,   // CMP
                _ => 0,
            };
            let mut bytes = Vec::new();
            if dst_reg >= 8 {
                bytes.push(0x41);
            }
            if (-128..=127).contains(&imm) {
                bytes.push(0x83);
                bytes.push(modrm_rr(alu_group, dst_reg));
                bytes.push(imm as u8);
            } else {
                bytes.push(0x81);
                bytes.push(modrm_rr(alu_group, dst_reg));
                bytes.extend_from_slice(&(imm as u32).to_le_bytes());
            }
            return bytes;
        }
    }

    // ALU %reg, %reg
    if let (Some(src_reg), Some(dst_reg)) = (parse_att_register(src), parse_att_register(dst)) {
        let mut bytes = Vec::new();
        let mut rex = 0u8;
        if src_reg >= 8 {
            rex |= 0x44;
        }
        if dst_reg >= 8 {
            rex |= 0x41;
        }
        if rex != 0 {
            bytes.push(rex);
        }
        bytes.push(opcode);
        bytes.push(modrm_rr(src_reg, dst_reg));
        return bytes;
    }

    if crate::common::verbosity::verbose_enabled() {
        eprintln!(
            "[inline-asm] WARNING: unsupported ALU operands: {}",
            operands
        );
    }
    vec![0x90]
}

/// Assemble a 64-bit ALU instruction in AT&T syntax.
fn assemble_alu_q(opcode: u8, operands: &str) -> Vec<u8> {
    let (src, dst) = split_att_operands(operands);

    if let Some(imm) = parse_att_immediate(src) {
        if let Some(dst_reg) = parse_att_register(dst) {
            let alu_group = match opcode {
                0x01 => 0u8,
                0x09 => 1,
                0x21 => 4,
                0x29 => 5,
                0x31 => 6,
                0x39 => 7,
                _ => 0,
            };
            let mut bytes = Vec::new();
            let mut rex = 0x48u8;
            if dst_reg >= 8 {
                rex |= 0x01;
            }
            bytes.push(rex);
            if (-128..=127).contains(&imm) {
                bytes.push(0x83);
                bytes.push(modrm_rr(alu_group, dst_reg));
                bytes.push(imm as u8);
            } else {
                bytes.push(0x81);
                bytes.push(modrm_rr(alu_group, dst_reg));
                bytes.extend_from_slice(&(imm as u32).to_le_bytes());
            }
            return bytes;
        }
    }

    if let (Some(src_reg), Some(dst_reg)) = (parse_att_register(src), parse_att_register(dst)) {
        let mut bytes = Vec::new();
        let mut rex = 0x48u8;
        if src_reg >= 8 {
            rex |= 0x04;
        }
        if dst_reg >= 8 {
            rex |= 0x01;
        }
        bytes.push(rex);
        bytes.push(opcode);
        bytes.push(modrm_rr(src_reg, dst_reg));
        return bytes;
    }

    if crate::common::verbosity::verbose_enabled() {
        eprintln!(
            "[inline-asm] WARNING: unsupported ALU64 operands: {}",
            operands
        );
    }
    vec![0x90]
}

/// Assemble push instruction.
fn assemble_push(operands: &str) -> Vec<u8> {
    let operands = operands.trim();
    if let Some(reg) = parse_att_register(operands) {
        let mut bytes = Vec::new();
        if reg >= 8 {
            bytes.push(0x41);
        }
        bytes.push(0x50 + (reg & 7));
        bytes
    } else if let Some(imm) = parse_att_immediate(operands) {
        if (-128..=127).contains(&imm) {
            vec![0x6A, imm as u8]
        } else {
            let mut bytes = vec![0x68];
            bytes.extend_from_slice(&(imm as u32).to_le_bytes());
            bytes
        }
    } else {
        vec![0x90]
    }
}

/// Assemble pop instruction.
fn assemble_pop(operands: &str) -> Vec<u8> {
    let operands = operands.trim();
    if let Some(reg) = parse_att_register(operands) {
        let mut bytes = Vec::new();
        if reg >= 8 {
            bytes.push(0x41);
        }
        bytes.push(0x58 + (reg & 7));
        bytes
    } else {
        vec![0x90]
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::traits::{MachineBasicBlock, MachineFunction};

    /// Helper: create a minimal empty MachineFunction.
    fn make_empty_func(name: &str) -> MachineFunction {
        MachineFunction {
            name: name.to_string(),
            blocks: vec![MachineBasicBlock::new(Some("entry".to_string()))],
            frame_size: 0,
            spill_slots: Vec::new(),
            callee_saved_regs: Vec::new(),
            is_leaf: true,
            vreg_to_ir_value: crate::common::fx_hash::FxHashMap::default(),
            sse_vregs: crate::common::fx_hash::FxHashSet::default(),
            va_save_area_offset: None,
            va_fp_save_area_offset: None,
            va_control_offset: None,
            named_gpr_count: 0,
            named_fp_count: 0,
            named_memory_stack_bytes: 0,
            num_param_moves: 0,
        }
    }

    #[test]
    fn test_assembler_context_new() {
        let ctx = AssemblerContext::new(false, SecurityConfig::none());
        assert_eq!(ctx.current_offset, 0);
        assert!(ctx.text_section.is_empty());
        assert!(ctx.relocations.is_empty());
        assert!(!ctx.pic_enabled);
    }

    #[test]
    fn test_emit_byte() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_byte(0xCC);
        assert_eq!(ctx.text_section, vec![0xCC]);
        assert_eq!(ctx.current_offset, 1);
    }

    #[test]
    fn test_emit_bytes() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_bytes(&[0x48, 0x89, 0xE5]);
        assert_eq!(ctx.text_section, vec![0x48, 0x89, 0xE5]);
        assert_eq!(ctx.current_offset, 3);
    }

    #[test]
    fn test_emit_u32_le() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_u32_le(0x12345678);
        assert_eq!(ctx.text_section, vec![0x78, 0x56, 0x34, 0x12]);
        assert_eq!(ctx.current_offset, 4);
    }

    #[test]
    fn test_emit_u64_le() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_u64_le(0x0102030405060708);
        assert_eq!(
            ctx.text_section,
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
        assert_eq!(ctx.current_offset, 8);
    }

    #[test]
    fn test_label_define_and_forward_reference() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        // Emit some bytes to set the offset.
        ctx.emit_bytes(&[0x90, 0x90, 0x90, 0x90, 0x90]); // 5 NOPs
                                                         // Place 4 placeholder bytes at offset 5.
        ctx.emit_u32_le(0);
        let fixup = LabelFixup {
            offset: 5,
            size: 4,
            pc_relative: true,
            addend: 0,
        };
        ctx.reference_label("target", fixup);
        // Emit more bytes.
        ctx.emit_bytes(&[0x90, 0x90]); // offsets 9, 10
                                       // Define the label at offset 11.
        ctx.define_label("target");
        // Fixup: target(11) - (5+4) + 0 = 2
        let patched = i32::from_le_bytes([
            ctx.text_section[5],
            ctx.text_section[6],
            ctx.text_section[7],
            ctx.text_section[8],
        ]);
        assert_eq!(patched, 2);
    }

    #[test]
    fn test_label_backward_reference() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        // Define label first.
        ctx.define_label("loop_top");
        ctx.emit_bytes(&[0x90; 10]); // 10 NOPs
                                     // Now reference the already-defined label.
        ctx.emit_u32_le(0);
        let fixup = LabelFixup {
            offset: 10,
            size: 4,
            pc_relative: true,
            addend: 0,
        };
        ctx.reference_label("loop_top", fixup);
        // Fixup: target(0) - (10+4) + 0 = -14
        let patched = i32::from_le_bytes([
            ctx.text_section[10],
            ctx.text_section[11],
            ctx.text_section[12],
            ctx.text_section[13],
        ]);
        assert_eq!(patched, -14);
    }

    #[test]
    fn test_resolve_pending_fixups_error_for_local() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_u32_le(0);
        // Reference a local label that is never defined.
        let fixup = LabelFixup {
            offset: 0,
            size: 4,
            pc_relative: true,
            addend: 0,
        };
        ctx.reference_label(".Lmissing", fixup);
        let result = ctx.resolve_pending_fixups();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains(".Lmissing"));
    }

    #[test]
    fn test_resolve_pending_fixups_external_symbol() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.emit_u32_le(0);
        // Reference an external label (no `.L` prefix).
        let fixup = LabelFixup {
            offset: 0,
            size: 4,
            pc_relative: true,
            addend: -4,
        };
        ctx.reference_label("printf", fixup);
        let result = ctx.resolve_pending_fixups();
        assert!(result.is_ok());
        assert_eq!(ctx.relocations.len(), 1);
        assert_eq!(ctx.relocations[0].symbol, "printf");
    }

    #[test]
    fn test_assemble_empty_function() {
        let mf = make_empty_func("test_fn");
        let config = SecurityConfig::none();
        let result = assemble(&mf, &config, false);
        assert!(!result.symbols.is_empty());
        assert_eq!(result.symbols[0].name, "test_fn");
        assert_eq!(result.symbols[0].sym_type, SymbolType::Function);
        assert!(result.retpoline_thunks.is_empty());
    }

    #[test]
    fn test_assemble_with_cet() {
        // Build a function that already has endbr64 as its first instruction
        // (as emit_prologue would insert when cf_protection is enabled).
        // The assembler no longer injects endbr64 itself — that responsibility
        // lies with the prologue generator to avoid double emission.
        let mut mf = make_empty_func("cet_fn");
        let endbr = crate::backend::x86_64::security::emit_endbr64();
        mf.blocks[0].instructions.insert(0, endbr);

        let config = SecurityConfig {
            retpoline: false,
            cf_protection: true,
            stack_probe: false,
        };
        let result = assemble(&mf, &config, false);
        // First 4 bytes must be endbr64: F3 0F 1E FA.
        assert!(result.text.len() >= 4);
        assert_eq!(&result.text[..4], &ENDBR64_BYTES);
    }

    #[test]
    fn test_assemble_with_retpoline() {
        let mf = make_empty_func("retpoline_fn");
        let config = SecurityConfig {
            retpoline: true,
            cf_protection: false,
            stack_probe: false,
        };
        let result = assemble(&mf, &config, false);
        // Should have 16 retpoline thunks.
        assert_eq!(result.retpoline_thunks.len(), 16);
        assert_eq!(result.retpoline_thunks[0].0, "__x86_indirect_thunk_rax");
        for (_, bytes) in &result.retpoline_thunks {
            assert_eq!(bytes.len(), 17);
        }
    }

    #[test]
    fn test_retpoline_thunk_encoding() {
        let thunks = assemble_retpoline_thunks();
        assert_eq!(thunks.len(), 16);
        // Validate the RAX thunk structure.
        let (name, bytes) = &thunks[0];
        assert_eq!(name, "__x86_indirect_thunk_rax");
        assert_eq!(bytes.len(), 17);
        // call +7 = E8 07 00 00 00
        assert_eq!(bytes[0], 0xE8);
        assert_eq!(&bytes[1..5], &7i32.to_le_bytes());
        // pause = F3 90
        assert_eq!(bytes[5], 0xF3);
        assert_eq!(bytes[6], 0x90);
        // lfence = 0F AE E8
        assert_eq!(bytes[7], 0x0F);
        assert_eq!(bytes[8], 0xAE);
        assert_eq!(bytes[9], 0xE8);
        // jmp -7 = EB F9
        assert_eq!(bytes[10], 0xEB);
        assert_eq!(bytes[11], 0xF9u8);
        // mov [rsp], rax = 48 89 04 24
        assert_eq!(bytes[12], 0x48);
        assert_eq!(bytes[13], 0x89);
        assert_eq!(bytes[14], 0x04);
        assert_eq!(bytes[15], 0x24);
        // ret = C3
        assert_eq!(bytes[16], 0xC3);
    }

    #[test]
    fn test_retpoline_thunk_r10() {
        let thunks = assemble_retpoline_thunks();
        // R10 is index 10 in the array.
        let (name, bytes) = &thunks[10];
        assert_eq!(name, "__x86_indirect_thunk_r10");
        assert_eq!(bytes.len(), 17);
        // R10 has hw_encoding = 10, so REX = 0x48|0x04 = 0x4C
        // ModR/M: reg=(10&7=2) << 3 | 4 = 0x14
        assert_eq!(bytes[12], 0x4C); // REX.W + REX.R
        assert_eq!(bytes[13], 0x89);
        assert_eq!(bytes[14], 0x14); // ModR/M
        assert_eq!(bytes[15], 0x24); // SIB
        assert_eq!(bytes[16], 0xC3);
    }

    #[test]
    fn test_symbol_binding_to_elf() {
        assert_eq!(SymbolBinding::Local.to_elf(), STB_LOCAL);
        assert_eq!(SymbolBinding::Global.to_elf(), STB_GLOBAL);
        assert_eq!(SymbolBinding::Weak.to_elf(), STB_WEAK);
    }

    #[test]
    fn test_symbol_type_to_elf() {
        assert_eq!(SymbolType::NoType.to_elf(), STT_NOTYPE);
        assert_eq!(SymbolType::Function.to_elf(), STT_FUNC);
        assert_eq!(SymbolType::Section.to_elf(), STT_SECTION);
    }

    #[test]
    fn test_symbol_visibility_to_elf() {
        assert_eq!(SymbolVisibility::Default.to_elf(), STV_DEFAULT);
        assert_eq!(SymbolVisibility::Hidden.to_elf(), STV_HIDDEN);
    }

    #[test]
    fn test_symbol_section_eq() {
        assert_ne!(SymbolSection::Text, SymbolSection::Data);
        assert_ne!(SymbolSection::Bss, SymbolSection::Rodata);
        assert_ne!(SymbolSection::Undefined, SymbolSection::Absolute);
        assert_eq!(SymbolSection::Text, SymbolSection::Text);
    }

    #[test]
    fn test_add_symbol_and_relocation() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        ctx.add_symbol(AssemblerSymbol {
            name: "test_sym".to_string(),
            offset: 0,
            size: 10,
            section: SymbolSection::Text,
            binding: SymbolBinding::Global,
            sym_type: SymbolType::Function,
            visibility: SymbolVisibility::Default,
        });
        assert_eq!(ctx.symbols.len(), 1);
        assert_eq!(ctx.symbols[0].name, "test_sym");

        ctx.add_relocation(RelocationEntry::new(
            0,
            "extern_fn".to_string(),
            X86_64RelocationType::Pc32,
            -4,
            ".text".to_string(),
        ));
        assert_eq!(ctx.relocations.len(), 1);
    }

    #[test]
    fn test_assemble_to_object_empty() {
        let target = Target::X86_64;
        let config = SecurityConfig::none();
        let result = assemble_to_object(&[], &[], &[], 0, &target, &config, false);
        assert!(result.is_ok());
        let elf_bytes = result.unwrap();
        // Must start with ELF magic: 7F 45 4C 46
        assert!(elf_bytes.len() >= 4);
        assert_eq!(&elf_bytes[0..4], &[0x7F, b'E', b'L', b'F']);
    }

    #[test]
    fn test_assemble_to_object_with_function() {
        let mf = make_empty_func("main");
        let target = Target::X86_64;
        let config = SecurityConfig::none();
        let result = assemble_to_object(&[&mf], &[], &[], 0, &target, &config, false);
        assert!(result.is_ok());
        let elf_bytes = result.unwrap();
        assert_eq!(&elf_bytes[0..4], &[0x7F, b'E', b'L', b'F']);
        // ELFCLASS64
        assert_eq!(elf_bytes[4], 2);
        // ELFDATA2LSB
        assert_eq!(elf_bytes[5], 1);
        // e_machine at offset 18: EM_X86_64 = 62
        let e_machine = u16::from_le_bytes([elf_bytes[18], elf_bytes[19]]);
        assert_eq!(e_machine, 62);
    }

    #[test]
    fn test_assemble_to_object_with_data_and_rodata() {
        let mf = make_empty_func("func");
        let target = Target::X86_64;
        let config = SecurityConfig::none();
        let global = vec![("my_global".to_string(), vec![1u8, 2, 3, 4])];
        let ro = vec![(".Lstr0".to_string(), b"Hello\0".to_vec())];
        let result = assemble_to_object(&[&mf], &global, &ro, 128, &target, &config, false);
        assert!(result.is_ok());
        let elf_bytes = result.unwrap();
        assert_eq!(&elf_bytes[0..4], &[0x7F, b'E', b'L', b'F']);
    }

    #[test]
    fn test_current_text_offset() {
        let mut ctx = AssemblerContext::new(false, SecurityConfig::none());
        assert_eq!(ctx.current_text_offset(), 0);
        ctx.emit_byte(0x90);
        assert_eq!(ctx.current_text_offset(), 1);
        ctx.emit_bytes(&[1, 2, 3]);
        assert_eq!(ctx.current_text_offset(), 4);
    }

    #[test]
    fn test_pic_enabled_flag() {
        let ctx = AssemblerContext::new(true, SecurityConfig::none());
        assert!(ctx.pic_enabled);
        let ctx2 = AssemblerContext::new(false, SecurityConfig::none());
        assert!(!ctx2.pic_enabled);
    }
}
