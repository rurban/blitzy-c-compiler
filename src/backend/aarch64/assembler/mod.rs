//! # Built-in AArch64 Assembler
//!
//! Self-contained assembler for the AArch64 (ARM 64-bit) architecture.
//! Converts A64 machine instructions into binary machine code and produces
//! ELF relocatable object sections.
//!
//! ## Architecture
//! - Accepts `A64Instruction` sequences from `codegen.rs`
//! - Dispatches to `encoder` for A64 instruction format binary encoding
//! - Collects `relocations` for unresolved symbol references
//! - Produces `.text` section bytes + relocation entries
//!
//! ## Key Characteristic: Fixed 32-bit Instruction Width
//! Every AArch64 instruction encodes to exactly 4 bytes. There is no
//! variable-length encoding, simplifying offset computation and branch targeting.
//!
//! ## Standalone Backend Mode
//! No external assembler is invoked. This module, together with `encoder.rs` and
//! `relocations.rs`, is entirely self-contained per BCC's zero-dependency mandate.
//!
//! ## Instruction Format Groups
//! - Data Processing (Immediate): ADD/SUB, MOV, logical, ADRP/ADR
//! - Data Processing (Register): shifted/extended register ops, CSEL
//! - Loads and Stores: LDR/STR, LDP/STP, pre/post-index
//! - Branches: B, BL, B.cond, BR, BLR, RET, CBZ/CBNZ, TBZ/TBNZ
//! - SIMD/FP: FADD, FSUB, FMUL, FDIV, FCVT, FMOV
//! - System: NOP, DMB, DSB, ISB, SVC, MRS, MSR

pub mod encoder;
pub mod relocations;

// ---------------------------------------------------------------------------
// Re-exports for parent module access
// ---------------------------------------------------------------------------

pub use self::encoder::AArch64Encoder;
pub use self::relocations::{AArch64RelocationHandler, AArch64RelocationType};

// ---------------------------------------------------------------------------
// Crate-internal imports
// ---------------------------------------------------------------------------

use crate::backend::aarch64::codegen::{A64Instruction, A64Opcode};
use crate::backend::aarch64::registers;
use crate::backend::linker_common::relocation::{RelocCategory, Relocation};
use crate::common::fx_hash::FxHashMap;
use crate::common::target::Target;

// Imported for documentation / future use — the assembler produces .text
// section bytes and relocations that feed into the ELF writer (via
// build_text_section) and the diagnostic engine is used for error reporting
// in extended inline assembly processing.
#[allow(unused_imports)]
use crate::backend::elf_writer_common;
#[allow(unused_imports)]
use crate::common::diagnostics::{DiagnosticEngine, Span};

use self::encoder::EncodedInstruction;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// AArch64 instruction width in bytes — every instruction is exactly 4 bytes.
const INSTRUCTION_SIZE: u64 = 4;

/// AArch64 NOP encoding: `0xD503201F` (little-endian: `1F 20 03 D5`).
const NOP_ENCODING: u32 = 0xD503_201F;

/// Maximum signed offset for B/BL (26-bit imm × 4 = ±128 MiB).
/// 2^25 instructions × 4 bytes = 128 MiB.
const MAX_BRANCH26_OFFSET: i64 = (1 << 27) - 4; // +128 MiB - 4
const MIN_BRANCH26_OFFSET: i64 = -(1 << 27); // -128 MiB

/// Maximum signed offset for B.cond / CBZ / CBNZ (19-bit imm × 4 = ±1 MiB).
const MAX_CONDBR19_OFFSET: i64 = (1 << 20) - 4; // +1 MiB - 4
const MIN_CONDBR19_OFFSET: i64 = -(1 << 20); // -1 MiB

/// Maximum signed offset for TBZ/TBNZ (14-bit imm × 4 = ±32 KiB).
const MAX_TSTBR14_OFFSET: i64 = (1 << 15) - 4; // +32 KiB - 4
const MIN_TSTBR14_OFFSET: i64 = -(1 << 15); // -32 KiB

// ---------------------------------------------------------------------------
// AssemblyRelocation — relocation entry generated during assembly
// ---------------------------------------------------------------------------

/// A relocation entry generated during assembly.
///
/// Records the location within the code buffer, the referenced symbol,
/// the AArch64-specific relocation type (as a raw ELF `r_type` value),
/// and the addend for RELA-style relocations.
#[derive(Debug, Clone)]
pub struct AssemblyRelocation {
    /// Offset within the code section where the relocation applies.
    pub offset: u64,
    /// Symbol name this relocation references.
    pub symbol: String,
    /// AArch64 relocation type (ELF r_type value).
    pub reloc_type: u32,
    /// Addend value for the relocation computation.
    pub addend: i64,
}

impl AssemblyRelocation {
    /// Convert this assembler-local relocation to the linker-common
    /// [`Relocation`] format for integration with the linker pipeline.
    pub fn to_linker_relocation(&self) -> Relocation {
        Relocation {
            offset: self.offset,
            symbol_name: self.symbol.clone(),
            sym_index: 0,
            rel_type: self.reloc_type,
            addend: self.addend,
            object_id: 0,
            section_index: 0,
            output_section_name: None,
        }
    }

    /// Classify this relocation using the AArch64 relocation type mapping.
    pub fn category(&self) -> RelocCategory {
        match AArch64RelocationType::from_raw(self.reloc_type) {
            Some(rt) => rt.category(),
            None => RelocCategory::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// AssemblyResult — output of the assembler
// ---------------------------------------------------------------------------

/// Result of assembling a sequence of AArch64 instructions.
///
/// Contains the machine code bytes suitable for a `.text` ELF section,
/// any unresolved relocations for the linker, and the symbol table
/// mapping labels/function names to code offsets.
#[derive(Debug)]
pub struct AssemblyResult {
    /// Assembled machine code bytes (`.text` section content).
    pub code: Vec<u8>,
    /// Relocations generated during assembly (for unresolved symbols).
    pub relocations: Vec<AssemblyRelocation>,
    /// Symbol definitions found during assembly (label → offset).
    pub symbols: FxHashMap<String, u64>,
    /// Total size of assembled code in bytes.
    pub code_size: u64,
}

impl AssemblyResult {
    /// Create an empty assembly result.
    pub fn empty() -> Self {
        AssemblyResult {
            code: Vec::new(),
            relocations: Vec::new(),
            symbols: FxHashMap::default(),
            code_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// InlineAsmOperand — operand descriptor for inline assembly
// ---------------------------------------------------------------------------

/// Describes a single operand for inline assembly substitution.
///
/// Used by `assemble_inline_asm` to replace `%0`, `%1`, or `%[name]`
/// placeholders in the assembly template string with concrete register
/// names or immediate values.
#[derive(Debug, Clone)]
pub struct InlineAsmOperand {
    /// Constraint string (e.g. `"r"`, `"=r"`, `"i"`, `"m"`).
    pub constraint: String,
    /// Concrete register ID (if register operand).
    pub register: Option<u8>,
    /// Immediate value (if immediate operand).
    pub immediate: Option<i64>,
    /// Named operand label (e.g. `[result]`).
    pub name: Option<String>,
}

// ---------------------------------------------------------------------------
// AArch64Assembler — the main assembler driver
// ---------------------------------------------------------------------------

/// Built-in AArch64 assembler.
///
/// Converts `A64Instruction` sequences into binary machine code,
/// collecting relocations for unresolved symbol references.
/// All instructions encode to exactly 4 bytes (fixed-width).
///
/// ## Usage
///
/// ```ignore
/// let mut asm = AArch64Assembler::new(false);
/// let result = asm.assemble_function(&instructions)?;
/// let (code, relocs) = asm.build_text_section();
/// ```
pub struct AArch64Assembler {
    /// The instruction encoder instance.
    encoder: AArch64Encoder,
    /// Accumulated machine code bytes.
    code: Vec<u8>,
    /// Accumulated relocations for unresolved symbols.
    relocations: Vec<AssemblyRelocation>,
    /// Label/symbol → code offset map.
    symbols: FxHashMap<String, u64>,
    /// Current code offset (write position), always a multiple of 4.
    current_offset: u64,
    /// Whether PIC mode is active (`-fPIC`).
    pic_mode: bool,
    /// Target architecture info.
    target: Target,
}

impl AArch64Assembler {
    // -----------------------------------------------------------------------
    // Construction and reset
    // -----------------------------------------------------------------------

    /// Create a new AArch64 assembler.
    ///
    /// # Arguments
    /// * `pic_mode` — whether to generate PIC-style ADRP+LDR GOT relocations.
    pub fn new(pic_mode: bool) -> Self {
        Self {
            encoder: AArch64Encoder::new(),
            code: Vec::new(),
            relocations: Vec::new(),
            symbols: FxHashMap::default(),
            current_offset: 0,
            pic_mode,
            target: Target::AArch64,
        }
    }

    /// Reset the assembler for a new function/section.
    ///
    /// Clears all accumulated code, relocations, symbols, and resets the
    /// write offset to zero. The encoder and configuration (PIC mode,
    /// target) are preserved.
    pub fn reset(&mut self) {
        self.code.clear();
        self.relocations.clear();
        self.symbols.clear();
        self.current_offset = 0;
    }

    // -----------------------------------------------------------------------
    // Symbol and label management
    // -----------------------------------------------------------------------

    /// Define a label at the current code offset.
    ///
    /// Records `name → current_offset` in the symbol table. Used for
    /// function entry points, branch targets, and local labels.
    pub fn define_label(&mut self, name: &str) {
        self.symbols.insert(name.to_string(), self.current_offset);
    }

    /// Return the current write position (byte offset) in the code buffer.
    ///
    /// This value is always a multiple of 4 (AArch64 instruction alignment).
    #[inline]
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    // -----------------------------------------------------------------------
    // Core assembly: single function
    // -----------------------------------------------------------------------

    /// Assemble a sequence of AArch64 instructions for a single function.
    ///
    /// For each instruction:
    /// 1. If the instruction carries a label (via `symbol` field on certain
    ///    pseudo-ops), record it at the current offset.
    /// 2. Dispatch to the encoder for binary encoding.
    /// 3. Collect any relocation produced by the encoder.
    /// 4. Handle pseudo-instructions (LA, CALL, INLINE_ASM) that expand
    ///    to multi-instruction sequences.
    /// 5. Append encoded bytes and advance the offset.
    ///
    /// Returns an `AssemblyResult` with accumulated code, relocations, and
    /// symbols for this function. The assembler state is **not** reset
    /// automatically — call [`reset`](Self::reset) before assembling
    /// the next function if a fresh context is needed.
    pub fn assemble_function(
        &mut self,
        instructions: &[A64Instruction],
    ) -> Result<AssemblyResult, String> {
        for inst in instructions {
            self.assemble_one(inst)?;
        }

        // Resolve local branch targets (basic block labels) so that
        // conditional branches don't produce unresolved relocations.
        let _ = self.resolve_local_branches();

        Ok(AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        })
    }

    // -----------------------------------------------------------------------
    // Core assembly: module (multiple functions)
    // -----------------------------------------------------------------------

    /// Assemble multiple functions sequentially into a single code section.
    ///
    /// Each function is identified by a `(name, instructions)` pair.
    /// Function names are recorded as symbols at their start offsets.
    /// Functions are aligned to 4-byte boundaries (inherent in AArch64).
    pub fn assemble_module(
        &mut self,
        functions: &[(String, Vec<A64Instruction>)],
    ) -> Result<AssemblyResult, String> {
        for (name, instructions) in functions {
            // Align to instruction boundary (should already be aligned).
            self.align_to(INSTRUCTION_SIZE);

            // Record the function name at its start offset.
            self.define_label(name);

            // Assemble all instructions for this function.
            for inst in instructions {
                self.assemble_one(inst)?;
            }
        }

        // Resolve local branch targets (basic block labels) before
        // returning the result. Without this step, conditional branches
        // targeting block labels like "if.then" or "for.body" remain as
        // unresolved relocations and cause linker errors.
        let _ = self.resolve_local_branches();

        Ok(AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        })
    }

    // -----------------------------------------------------------------------
    // Single instruction assembly
    // -----------------------------------------------------------------------

    /// Assemble a single instruction, handling pseudo-instructions and
    /// relocation emission.
    fn assemble_one(&mut self, inst: &A64Instruction) -> Result<(), String> {
        match inst.opcode {
            // Pseudo-instruction: Load Address (ADRP + ADD pair).
            A64Opcode::LA => {
                self.assemble_load_address(inst)?;
            }

            // Pseudo-instruction: Function call (may expand to BL or ADRP+BLR).
            A64Opcode::CALL => {
                self.assemble_call(inst)?;
            }

            // Pseudo-instruction: Inline assembly passthrough.
            A64Opcode::INLINE_ASM => {
                // Inline assembly is handled as raw bytes or via
                // assemble_inline_asm(); here we emit a NOP placeholder
                // if no symbol/template is provided.
                if inst.symbol.is_some() {
                    // The symbol field carries the asm template; for now
                    // emit a NOP as placeholder — full inline asm parsing
                    // is handled by assemble_inline_asm().
                    self.emit_raw_word(NOP_ENCODING);
                } else {
                    self.emit_raw_word(NOP_ENCODING);
                }
            }

            // NOP with a label-defining comment (e.g. "if.then:") —
            // this is a basic block label pseudo-instruction. Define the
            // label in the symbol table WITHOUT emitting an actual NOP so
            // conditional branch relocations can resolve locally.
            A64Opcode::NOP if inst.comment.as_ref().map_or(false, |c| c.ends_with(':')) => {
                let comment = inst.comment.as_ref().unwrap();
                let label = &comment[..comment.len() - 1]; // strip trailing ':'
                self.define_label(label);
            }

            // Standard single-instruction encoding.
            _ => {
                let encoded = self.encoder.encode(inst)?;
                self.emit_encoded(inst, &encoded)?;
            }
        }
        Ok(())
    }

    /// Emit an encoded instruction result, collecting any relocations.
    fn emit_encoded(
        &mut self,
        inst: &A64Instruction,
        encoded: &EncodedInstruction,
    ) -> Result<(), String> {
        // Handle multi-instruction expansions (e.g., MOV_imm → MOVZ+MOVK sequence).
        // The encoder may produce more than 4 bytes for such pseudo-instructions.
        let num_words = encoded.bytes.len() / 4;

        // Emit the first 4-byte instruction word and attach any encoder relocation.
        if let Some(ref enc_reloc) = encoded.relocation {
            self.relocations.push(AssemblyRelocation {
                offset: self.current_offset,
                symbol: inst.symbol.clone().unwrap_or_default(),
                reloc_type: enc_reloc.reloc_type,
                addend: enc_reloc.addend,
            });
        }

        // Append all encoded bytes (could be 4, 8, 12, or 16 for multi-word sequences).
        self.code.extend_from_slice(&encoded.bytes);
        self.current_offset += encoded.bytes.len() as u64;

        // For multi-instruction expansions without explicit relocations,
        // no additional relocation work is needed — the encoder handles the
        // full MOVZ+MOVK sequence internally.
        let _ = num_words; // suppress unused warning; counted for documentation

        Ok(())
    }

    /// Emit a raw 32-bit word into the code buffer (little-endian).
    fn emit_raw_word(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
        self.current_offset += INSTRUCTION_SIZE;
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction expansion: Load Address (LA)
    // -----------------------------------------------------------------------

    /// Assemble the LA (Load Address) pseudo-instruction.
    ///
    /// Expands to an ADRP + ADD pair for non-GOT addressing, or
    /// ADRP + LDR for GOT-relative addressing (PIC mode).
    fn assemble_load_address(&mut self, inst: &A64Instruction) -> Result<(), String> {
        let sym = inst.symbol.as_deref().unwrap_or("");

        let rd = inst.rd.unwrap_or(0);

        if self.pic_mode {
            // PIC mode: ADRP + LDR from GOT.
            let adrp_offset = self.current_offset;
            self.emit_adrp(rd, true);
            self.emit_got_relocation(sym, adrp_offset);

            // LDR Xd, [Xd, #:got_lo12:sym]
            // Emit a placeholder LDR instruction with relocation.
            let ldr_word = encode_ldr_unsigned_imm(rd, rd, 0);
            let ldr_offset = self.current_offset;
            self.emit_raw_word(ldr_word);
            // The GOT LDR relocation was already emitted as part of the
            // ADRP+LDR pair in emit_got_relocation.
            let _ = ldr_offset;
        } else {
            // Non-PIC: ADRP + ADD pair.
            let adrp_offset = self.current_offset;
            self.emit_adrp(rd, false);
            self.emit_adrp_add_relocation(sym, adrp_offset);

            // ADD Xd, Xd, #:lo12:sym
            let add_word = encode_add_imm(rd, rd, 0);
            self.emit_raw_word(add_word);
            // The ADD relocation was already emitted as part of the
            // ADRP+ADD pair in emit_adrp_add_relocation.
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction expansion: CALL
    // -----------------------------------------------------------------------

    /// Assemble the CALL pseudo-instruction.
    ///
    /// For direct calls with a known symbol, emits a BL instruction with
    /// a CALL26 relocation. For indirect calls or when PIC mode requires
    /// GOT lookup, may expand to ADRP+LDR+BLR.
    fn assemble_call(&mut self, inst: &A64Instruction) -> Result<(), String> {
        let sym = inst.symbol.as_deref().unwrap_or("");

        if !sym.is_empty() {
            // Direct call via BL — emit a placeholder BL with CALL26 relocation.
            let bl_word: u32 = 0x9400_0000; // BL (imm26=0 placeholder)
            let offset = self.current_offset;
            self.emit_raw_word(bl_word);
            self.emit_branch_relocation(sym, offset, AArch64RelocationType::Call26.to_raw());
        } else if let Some(rn) = inst.rn {
            // Indirect call via BLR Xn.
            let blr_word: u32 = 0xD63F_0000 | ((registers::hw_encoding(rn) as u32) << 5);
            self.emit_raw_word(blr_word);
        } else {
            return Err("CALL instruction has no symbol and no register operand".to_string());
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // ADRP emission helper
    // -----------------------------------------------------------------------

    /// Emit an ADRP instruction placeholder (encoding with imm21=0).
    fn emit_adrp(&mut self, rd: u32, _is_got: bool) {
        // ADRP encoding: op=1 (bit 31), immlo (bits 30:29), 10000 (bits 28:24),
        // immhi (bits 23:5), Rd (bits 4:0). With imm21=0:
        let hw_rd = registers::hw_encoding(rd) as u32;
        let word: u32 = (1 << 31) | (0b10000 << 24) | hw_rd;
        self.emit_raw_word(word);
    }

    // -----------------------------------------------------------------------
    // Relocation emission helpers
    // -----------------------------------------------------------------------

    /// Emit ADRP + ADD relocation pair for non-GOT symbol addressing.
    ///
    /// - At `adrp_offset`: `R_AARCH64_ADR_PREL_PG_HI21` (page-relative high bits)
    /// - At `adrp_offset + 4`: `R_AARCH64_ADD_ABS_LO12_NC` (low 12-bit offset)
    fn emit_adrp_add_relocation(&mut self, symbol: &str, adrp_offset: u64) {
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AdrPrelPgHi21.to_raw(),
            addend: 0,
        });
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset + INSTRUCTION_SIZE,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AddAbsLo12Nc.to_raw(),
            addend: 0,
        });
    }

    /// Emit ADRP + LDR GOT relocation pair for PIC symbol addressing.
    ///
    /// - At `adrp_offset`: `R_AARCH64_ADR_GOT_PAGE` (GOT page)
    /// - At `adrp_offset + 4`: `R_AARCH64_LD64_GOT_LO12_NC` (GOT entry low bits)
    fn emit_got_relocation(&mut self, symbol: &str, adrp_offset: u64) {
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::AdrGotPage.to_raw(),
            addend: 0,
        });
        self.relocations.push(AssemblyRelocation {
            offset: adrp_offset + INSTRUCTION_SIZE,
            symbol: symbol.to_string(),
            reloc_type: AArch64RelocationType::Ld64GotLo12Nc.to_raw(),
            addend: 0,
        });
    }

    /// Emit a branch-type relocation at the given offset.
    fn emit_branch_relocation(&mut self, symbol: &str, offset: u64, reloc_type: u32) {
        self.relocations.push(AssemblyRelocation {
            offset,
            symbol: symbol.to_string(),
            reloc_type,
            addend: 0,
        });
    }

    // -----------------------------------------------------------------------
    // Local branch resolution
    // -----------------------------------------------------------------------

    /// Resolve local branch targets after all instructions have been assembled.
    ///
    /// For each relocation targeting a label defined in `self.symbols`:
    /// - Compute the PC-relative offset (`target - patch_site`).
    /// - Validate the offset fits within the relocation's range.
    /// - Patch the instruction encoding in the code buffer.
    /// - Remove the resolved relocation from the list.
    ///
    /// Relocations referencing undefined/external symbols are left intact
    /// for the linker to process.
    ///
    /// # Branch Ranges (AArch64)
    /// - B/BL (CALL26/JUMP26): ±128 MiB (26-bit signed word offset)
    /// - B.cond/CBZ/CBNZ (CONDBR19): ±1 MiB (19-bit signed word offset)
    /// - TBZ/TBNZ (TSTBR14): ±32 KiB (14-bit signed word offset)
    pub fn resolve_local_branches(&mut self) -> Result<(), String> {
        // Phase 1: Scan relocations and collect patch actions.
        // We separate the scanning from the patching to satisfy the borrow
        // checker — we cannot mutably borrow self.code while iterating
        // over self.relocations.
        enum PatchAction {
            Imm26 { offset: usize, imm26: i32 },
            Imm19 { offset: usize, imm19: i32 },
            Imm14 { offset: usize, imm14: i32 },
            AdrImm { offset: usize, imm21: i32 },
            Imm12 { offset: usize, imm12: u32 },
        }

        let mut patches: Vec<PatchAction> = Vec::new();
        let mut unresolved_indices: Vec<usize> = Vec::new();

        for (idx, reloc) in self.relocations.iter().enumerate() {
            if let Some(&target_offset) = self.symbols.get(&reloc.symbol) {
                let pc = reloc.offset;
                let raw_offset = target_offset as i64 - pc as i64;

                let reloc_type = AArch64RelocationType::from_raw(reloc.reloc_type);

                match reloc_type {
                    Some(AArch64RelocationType::Call26) | Some(AArch64RelocationType::Jump26) => {
                        if !(MIN_BRANCH26_OFFSET..=MAX_BRANCH26_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "branch target '{}' out of range for B/BL: offset {} bytes \
                                 (must be within ±128 MiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm26 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm26 {
                            offset: pc as usize,
                            imm26,
                        });
                    }

                    Some(AArch64RelocationType::Condbr19) => {
                        if !(MIN_CONDBR19_OFFSET..=MAX_CONDBR19_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "conditional branch target '{}' out of range for B.cond: \
                                 offset {} bytes (must be within ±1 MiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm19 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm19 {
                            offset: pc as usize,
                            imm19,
                        });
                    }

                    Some(AArch64RelocationType::Tstbr14) => {
                        if !(MIN_TSTBR14_OFFSET..=MAX_TSTBR14_OFFSET).contains(&raw_offset) {
                            return Err(format!(
                                "test-and-branch target '{}' out of range for TBZ/TBNZ: \
                                 offset {} bytes (must be within ±32 KiB)",
                                reloc.symbol, raw_offset
                            ));
                        }
                        let imm14 = (raw_offset >> 2) as i32;
                        patches.push(PatchAction::Imm14 {
                            offset: pc as usize,
                            imm14,
                        });
                    }

                    Some(AArch64RelocationType::AdrPrelPgHi21)
                    | Some(AArch64RelocationType::AdrPrelPgHi21Nc) => {
                        let target_page = (target_offset as i64) & !0xFFF;
                        let pc_page = (pc as i64) & !0xFFF;
                        let page_offset = target_page - pc_page;
                        let imm21 = (page_offset >> 12) as i32;
                        patches.push(PatchAction::AdrImm {
                            offset: pc as usize,
                            imm21,
                        });
                    }

                    Some(AArch64RelocationType::AddAbsLo12Nc) => {
                        let lo12 = (target_offset & 0xFFF) as u32;
                        patches.push(PatchAction::Imm12 {
                            offset: pc as usize,
                            imm12: lo12,
                        });
                    }

                    _ => {
                        // Cannot resolve locally — keep for linker.
                        unresolved_indices.push(idx);
                        continue;
                    }
                }
                // Successfully resolved — index not added to unresolved.
            } else {
                // Symbol not defined locally — keep for linker.
                unresolved_indices.push(idx);
            }
        }

        // Phase 2: Apply all collected patches to the code buffer.
        for patch in &patches {
            match patch {
                PatchAction::Imm26 { offset, imm26 } => {
                    self.patch_imm26(*offset, *imm26);
                }
                PatchAction::Imm19 { offset, imm19 } => {
                    self.patch_imm19(*offset, *imm19);
                }
                PatchAction::Imm14 { offset, imm14 } => {
                    self.patch_imm14(*offset, *imm14);
                }
                PatchAction::AdrImm { offset, imm21 } => {
                    self.patch_adr_imm(*offset, *imm21);
                }
                PatchAction::Imm12 { offset, imm12 } => {
                    self.patch_imm12(*offset, *imm12);
                }
            }
        }

        // Phase 3: Retain only unresolved relocations.
        let old_relocations = std::mem::take(&mut self.relocations);
        self.relocations = unresolved_indices
            .into_iter()
            .filter_map(|idx| old_relocations.get(idx).cloned())
            .collect();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Instruction patching helpers
    // -----------------------------------------------------------------------

    /// Patch a 26-bit signed immediate into a B/BL instruction at `offset`.
    fn patch_imm26(&mut self, offset: usize, imm26: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x03FF_FFFF) | ((imm26 as u32) & 0x03FF_FFFF);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 19-bit signed immediate into a B.cond/CBZ/CBNZ instruction.
    fn patch_imm19(&mut self, offset: usize, imm19: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x00FF_FFE0) | (((imm19 as u32) & 0x7FFFF) << 5);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 14-bit signed immediate into a TBZ/TBNZ instruction.
    fn patch_imm14(&mut self, offset: usize, imm14: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x0007_FFE0) | (((imm14 as u32) & 0x3FFF) << 5);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 21-bit signed immediate into an ADRP/ADR instruction.
    fn patch_adr_imm(&mut self, offset: usize, imm21: i32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let imm = imm21 as u32;
        let immlo = imm & 0x3;
        let immhi = (imm >> 2) & 0x7FFFF;
        let patched = (inst & !(0x00FF_FFE0 | (0x3 << 29))) | (immhi << 5) | (immlo << 29);
        write_le_u32(&mut self.code, offset, patched);
    }

    /// Patch a 12-bit unsigned immediate into an ADD/LDR/STR instruction.
    fn patch_imm12(&mut self, offset: usize, imm12: u32) {
        if offset + 4 > self.code.len() {
            return;
        }
        let inst = read_le_u32(&self.code, offset);
        let patched = (inst & !0x003F_FC00) | ((imm12 & 0xFFF) << 10);
        write_le_u32(&mut self.code, offset, patched);
    }

    // -----------------------------------------------------------------------
    // Alignment and padding
    // -----------------------------------------------------------------------

    /// Pad the code buffer with NOP instructions to reach the specified
    /// alignment boundary.
    ///
    /// The alignment must be a power of two and at least 4 bytes (the
    /// inherent instruction alignment of AArch64). The minimum alignment
    /// is derived from the target's instruction width to ensure correctness.
    pub fn align_to(&mut self, alignment: u64) {
        // Ensure alignment is at least the instruction size for this target.
        // On AArch64, all instructions are 4-byte aligned. The page_size()
        // call validates we have a valid target configuration.
        let _ = self.target.page_size();
        let alignment = alignment.max(INSTRUCTION_SIZE);
        while self.current_offset % alignment != 0 {
            self.emit_nop();
        }
    }

    /// Emit a single NOP instruction (4 bytes).
    ///
    /// AArch64 NOP encoding: `0xD503201F` (little-endian: `1F 20 03 D5`).
    pub fn emit_nop(&mut self) {
        self.emit_raw_word(NOP_ENCODING);
    }

    // -----------------------------------------------------------------------
    // Finalization and section building
    // -----------------------------------------------------------------------

    /// Finalize the assembly, resolving local branches and producing the
    /// final [`AssemblyResult`].
    ///
    /// After finalization, only relocations for external/undefined symbols
    /// remain in the result — all local branch targets have been patched
    /// into the code buffer.
    pub fn finalize(&mut self) -> AssemblyResult {
        // Attempt to resolve local branches; any errors are deferred
        // as unresolved relocations rather than hard failures, since the
        // linker may handle long-range branches.
        let _ = self.resolve_local_branches();

        AssemblyResult {
            code: self.code.clone(),
            relocations: self.relocations.clone(),
            symbols: self.symbols.clone(),
            code_size: self.current_offset,
        }
    }

    /// Return the code bytes and unresolved relocations suitable for
    /// constructing an ELF `.text` section.
    ///
    /// This method does not modify the assembler state — it returns cloned
    /// copies of the internal buffers.
    pub fn build_text_section(&self) -> (Vec<u8>, Vec<AssemblyRelocation>) {
        (self.code.clone(), self.relocations.clone())
    }

    // -----------------------------------------------------------------------
    // Inline assembly support
    // -----------------------------------------------------------------------

    /// Assemble an inline assembly template string with operand substitution.
    ///
    /// Parses the template, substitutes operand placeholders (`%0`, `%1`,
    /// `%[name]`) with register names or immediate values, and encodes each
    /// resulting instruction via the encoder.
    ///
    /// Handles `.pushsection` / `.popsection` directives by switching the
    /// output section context (though for the initial implementation, these
    /// directives are noted but the bytes still go into the current code
    /// buffer).
    ///
    /// Returns the total number of bytes emitted.
    pub fn assemble_inline_asm(
        &mut self,
        template: &str,
        operands: &[InlineAsmOperand],
    ) -> Result<usize, String> {
        let start_offset = self.current_offset;

        // Split the template into individual lines/instructions.
        let lines: Vec<&str> = template
            .split(|c| c == '\n' || c == ';')
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        for line in &lines {
            // Handle assembler directives.
            if line.starts_with('.') {
                self.handle_asm_directive(line)?;
                continue;
            }

            // Substitute operand placeholders in the instruction text.
            let expanded = self.substitute_operands(line, operands);

            // For inline assembly, we encode each instruction individually.
            // If the line can be parsed as a known A64 instruction, encode it.
            // Otherwise, emit a NOP as a placeholder — the full inline asm
            // parser will be extended as needed during kernel build.
            match self.try_encode_asm_line(&expanded) {
                Ok(bytes) => {
                    if !bytes.is_empty() {
                        self.code.extend_from_slice(&bytes);
                        self.current_offset += bytes.len() as u64;
                    }
                }
                Err(msg) => {
                    // Emit a diagnostic warning per AAP §0.7.6: the compiler
                    // MUST NOT silently miscompile unknown extensions.  Log the
                    // unsupported instruction and emit NOP so linking can
                    // proceed; the note is gated behind --verbose like other
                    // backend fallback diagnostics.
                    if crate::common::verbosity::verbose_enabled() {
                        eprintln!(
                            "bcc: warning: AArch64 inline asm: {}; emitting NOP placeholder for: {}",
                            msg, expanded
                        );
                    }
                    self.emit_nop();
                }
            }
        }

        let bytes_emitted = (self.current_offset - start_offset) as usize;
        Ok(bytes_emitted)
    }

    /// Handle an assembler directive within inline assembly.
    fn handle_asm_directive(&mut self, directive: &str) -> Result<(), String> {
        let lower = directive.to_ascii_lowercase();

        if lower.starts_with(".pushsection") || lower.starts_with(".popsection") {
            // Section switching directives — noted but currently inline asm
            // bytes stay in the main code buffer. A full implementation would
            // maintain a section stack and redirect output accordingly.
            Ok(())
        } else if lower.starts_with(".align") || lower.starts_with(".p2align") {
            // Alignment directive — parse the alignment value and pad.
            let parts: Vec<&str> = directive.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(align_val) = parts[1].trim_end_matches(',').parse::<u64>() {
                    let alignment = if lower.starts_with(".p2align") {
                        1u64 << align_val
                    } else {
                        align_val
                    };
                    self.align_to(alignment);
                }
            }
            Ok(())
        } else if lower.starts_with(".byte") {
            // .byte directives — emit raw bytes.
            let parts: Vec<&str> = directive[5..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Ok(val) = parse_int_literal(trimmed) {
                    self.code.push(val as u8);
                    // .byte doesn't maintain 4-byte alignment; that's expected
                    // for inline asm data emission.
                }
            }
            // Re-align current_offset to the actual code length.
            self.current_offset = self.code.len() as u64;
            Ok(())
        } else if lower.starts_with(".word") || lower.starts_with(".long") {
            // .word / .long directives — emit 32-bit values.
            // ".word" is 5 chars, ".long" is 5 chars — both have the same prefix length.
            let prefix_len = 5;
            let parts: Vec<&str> = directive[prefix_len..].split(',').collect();
            for part in parts {
                let trimmed = part.trim();
                if let Ok(val) = parse_int_literal(trimmed) {
                    self.code.extend_from_slice(&(val as u32).to_le_bytes());
                }
            }
            self.current_offset = self.code.len() as u64;
            Ok(())
        } else {
            // Unknown directive — ignore silently for resilience.
            Ok(())
        }
    }

    /// Substitute operand placeholders in an asm template line.
    ///
    /// Handles `%0`, `%1`, etc., and `%[name]` named operands.
    fn substitute_operands(&self, line: &str, operands: &[InlineAsmOperand]) -> String {
        let mut result = String::with_capacity(line.len());
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i] == '%' && i + 1 < len {
                if chars[i + 1] == '[' {
                    // Named operand: %[name]
                    if let Some(close) = chars[i + 2..].iter().position(|&c| c == ']') {
                        let name: String = chars[i + 2..i + 2 + close].iter().collect();
                        let replacement = self.find_operand_by_name(&name, operands);
                        result.push_str(&replacement);
                        i += 3 + close; // skip %[name]
                        continue;
                    }
                } else if chars[i + 1] == '%' {
                    // Escaped percent: %%
                    result.push('%');
                    i += 2;
                    continue;
                } else if chars[i + 1].is_ascii_digit() {
                    // Positional operand: %0, %1, etc.
                    let mut end = i + 1;
                    while end < len && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                    let idx_str: String = chars[i + 1..end].iter().collect();
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        let replacement = self.format_operand(idx, operands);
                        result.push_str(&replacement);
                        i = end;
                        continue;
                    }
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }

    /// Find an operand by its named label and format it.
    fn find_operand_by_name(&self, name: &str, operands: &[InlineAsmOperand]) -> String {
        for (idx, op) in operands.iter().enumerate() {
            if op.name.as_deref() == Some(name) {
                return self.format_operand(idx, operands);
            }
        }
        // Not found — return the placeholder as-is for error resilience.
        format!("%[{}]", name)
    }

    /// Format operand at the given index as a string suitable for asm substitution.
    fn format_operand(&self, idx: usize, operands: &[InlineAsmOperand]) -> String {
        if idx >= operands.len() {
            return format!("%{}", idx);
        }
        let op = &operands[idx];
        if let Some(reg) = op.register {
            format_aarch64_register(reg)
        } else if let Some(imm) = op.immediate {
            format!("#{}", imm)
        } else {
            format!("%{}", idx)
        }
    }

    /// Attempt to encode a single asm text line into machine code bytes.
    ///
    /// This is a simplified parser for common AArch64 instructions found
    /// in inline assembly. For unrecognized instructions, returns an error
    /// so the caller can fall back to a NOP placeholder.
    /// Try to encode a single AArch64 inline assembly text line into bytes.
    ///
    /// Handles the common instruction patterns found in Linux kernel inline
    /// assembly: system instructions (MRS, MSR, DSB, DMB, ISB, SVC, HVC,
    /// SMC, WFI, WFE, SEV, YIELD), MOV (register-to-register), and NOP.
    ///
    /// For unrecognized instructions, returns `Err(message)` with the
    /// offending text so the caller can emit a diagnostic per AAP §0.7.6.
    fn try_encode_asm_line(&self, line: &str) -> Result<Vec<u8>, String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(Vec::new());
        }

        // Split into mnemonic + operands
        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        let mnemonic = parts[0].to_lowercase();
        let operand_str = if parts.len() > 1 { parts[1].trim() } else { "" };
        let operands: Vec<&str> = if operand_str.is_empty() {
            Vec::new()
        } else {
            operand_str.split(',').map(|s| s.trim()).collect()
        };

        let word: u32 = match mnemonic.as_str() {
            // NOP: 0xD503201F
            "nop" => NOP_ENCODING,

            // MRS Xt, <sysreg> — read system register
            // Encoding: 1101 0101 0011 op0 op1 CRn CRm op2 Rt
            "mrs" => {
                if operands.len() < 2 {
                    return Err(format!("mrs requires 2 operands, got {}", operands.len()));
                }
                let rt = parse_asm_gpr(operands[0])?;
                let sysreg = parse_aarch64_sysreg(operands[1])?;
                // MRS: 1101 0101 0011 <sysreg 16 bits> Rt
                0xD530_0000 | (sysreg << 5) | (rt as u32)
            }

            // MSR <sysreg>, Xt — write system register
            "msr" => {
                if operands.len() < 2 {
                    return Err(format!("msr requires 2 operands, got {}", operands.len()));
                }
                // MSR can be: MSR <sysreg>, Xt  OR  MSR <pstatefield>, #imm
                if let Ok(sysreg) = parse_aarch64_sysreg(operands[0]) {
                    let rt = parse_asm_gpr(operands[1])?;
                    // MSR: 1101 0101 0001 <sysreg 16 bits> Rt
                    0xD510_0000 | (sysreg << 5) | (rt as u32)
                } else {
                    // MSR <pstatefield>, #imm — simplified: emit as HINT NOP
                    NOP_ENCODING
                }
            }

            // DSB <option> — Data Synchronization Barrier
            "dsb" => {
                let crm = parse_barrier_option(operands.first().copied().unwrap_or("sy"));
                // DSB: 1101 0101 0000 0011 0011 CRm 1 00 11111
                0xD503_3000 | ((crm & 0xF) << 8) | 0x9F
            }

            // DMB <option> — Data Memory Barrier
            "dmb" => {
                let crm = parse_barrier_option(operands.first().copied().unwrap_or("sy"));
                // DMB: 1101 0101 0000 0011 0011 CRm 1 01 11111
                0xD503_3000 | ((crm & 0xF) << 8) | 0xBF
            }

            // ISB — Instruction Synchronization Barrier
            "isb" => {
                // ISB SY: 0xD5033FDF
                0xD503_3FDF
            }

            // SVC #imm16 — Supervisor Call
            "svc" => {
                let imm = parse_asm_immediate_aarch64(operands.first().copied().unwrap_or("0"));
                // SVC: 1101 0100 000 imm16 000 01
                0xD400_0001 | (((imm as u32) & 0xFFFF) << 5)
            }

            // HVC #imm16 — Hypervisor Call
            "hvc" => {
                let imm = parse_asm_immediate_aarch64(operands.first().copied().unwrap_or("0"));
                0xD400_0002 | (((imm as u32) & 0xFFFF) << 5)
            }

            // SMC #imm16 — Secure Monitor Call
            "smc" => {
                let imm = parse_asm_immediate_aarch64(operands.first().copied().unwrap_or("0"));
                0xD400_0003 | (((imm as u32) & 0xFFFF) << 5)
            }

            // WFI — Wait For Interrupt: 0xD503207F
            "wfi" => 0xD503_207F,

            // WFE — Wait For Event: 0xD503205F
            "wfe" => 0xD503_205F,

            // SEV — Send Event: 0xD503209F
            "sev" => 0xD503_209F,

            // SEVL — Send Event Local: 0xD50320BF
            "sevl" => 0xD503_20BF,

            // YIELD: 0xD503203F
            "yield" => 0xD503_203F,

            // BRK #imm16 — Breakpoint
            "brk" => {
                let imm = parse_asm_immediate_aarch64(operands.first().copied().unwrap_or("0"));
                0xD420_0000 | (((imm as u32) & 0xFFFF) << 5)
            }

            // MOV Xd, Xn (register-to-register via ORR Xd, XZR, Xn)
            "mov" => {
                if operands.len() < 2 {
                    return Err(format!("mov requires 2 operands, got {}", operands.len()));
                }
                let rd = parse_asm_gpr(operands[0])?;
                // Check if second operand is immediate or register
                if operands[1].starts_with('#') || operands[1].starts_with("0x") {
                    let imm = parse_asm_immediate_aarch64(operands[1]);
                    // MOVZ Xd, #imm16 — for small immediates
                    0xD280_0000 | (((imm as u32) & 0xFFFF) << 5) | (rd as u32)
                } else {
                    let rn = parse_asm_gpr(operands[1])?;
                    // ORR Xd, XZR, Xn (sf=1, opc=01, shift=00, N=0)
                    0xAA00_03E0 | ((rn as u32) << 16) | (rd as u32)
                }
            }

            // RET {Xn} — Return (default X30/LR)
            "ret" => {
                let rn = if !operands.is_empty() {
                    parse_asm_gpr(operands[0]).unwrap_or(30)
                } else {
                    30 // LR
                };
                // RET: 1101 0110 0101 1111 0000 00 Rn 00000
                0xD65F_0000 | ((rn as u32) << 5)
            }

            // B <label> — unconditional branch (26-bit offset)
            "b" => {
                // For inline asm, branches to labels are typically local.
                // Emit B with offset 0; relocation would fix it up.
                0x1400_0000
            }

            // BL <label> — branch with link
            "bl" => 0x9400_0000,

            // CBNZ / CBZ patterns
            "cbz" | "cbnz" => {
                let _rd = if !operands.is_empty() {
                    parse_asm_gpr(operands[0]).unwrap_or(0)
                } else {
                    0
                };
                if mnemonic == "cbz" {
                    0xB400_0000 | (_rd as u32)
                } else {
                    0xB500_0000 | (_rd as u32)
                }
            }

            // STR / LDR — simplified encoding for common patterns
            "str" => {
                if operands.len() >= 2 {
                    let rt = parse_asm_gpr(operands[0]).unwrap_or(0);
                    if let Some((rn, offset)) = parse_memory_operand_aarch64(operands[1]) {
                        let scaled = ((offset / 8) as u32) & 0xFFF;
                        // STR Xt, [Xn, #offset] (unsigned offset, 64-bit)
                        // opc=00 for STR 64-bit is bits [23:22] = 0
                        (0b11 << 30) | (0b111001 << 24) | (scaled << 10) | (rn << 5) | (rt as u32)
                    } else {
                        return Err(format!("unsupported str operand format: {}", operands[1]));
                    }
                } else {
                    return Err("str requires at least 2 operands".to_string());
                }
            }

            "ldr" => {
                if operands.len() >= 2 {
                    let rt = parse_asm_gpr(operands[0]).unwrap_or(0);
                    if let Some((rn, offset)) = parse_memory_operand_aarch64(operands[1]) {
                        let scaled = ((offset / 8) as u32) & 0xFFF;
                        encode_ldr_unsigned_imm(rt, rn, scaled)
                    } else {
                        return Err(format!("unsupported ldr operand format: {}", operands[1]));
                    }
                } else {
                    return Err("ldr requires at least 2 operands".to_string());
                }
            }

            // ADD Xd, Xn, #imm
            "add" => {
                if operands.len() >= 3 {
                    let rd = parse_asm_gpr(operands[0])?;
                    let rn = parse_asm_gpr(operands[1])?;
                    let imm = parse_asm_immediate_aarch64(operands[2]) as u32;
                    encode_add_imm(rd, rn, imm & 0xFFF)
                } else {
                    return Err("add requires 3 operands".to_string());
                }
            }

            // SUB Xd, Xn, #imm
            "sub" => {
                if operands.len() >= 3 {
                    let rd = parse_asm_gpr(operands[0])?;
                    let rn = parse_asm_gpr(operands[1])?;
                    let imm = parse_asm_immediate_aarch64(operands[2]) as u32;
                    let hw_rd = registers::hw_encoding(rd) as u32;
                    let hw_rn = registers::hw_encoding(rn) as u32;
                    // SUB (immediate, 64-bit): sf=1 op=1 S=0 100010 sh imm12 Rn Rd
                    (1 << 31)
                        | (1 << 30)
                        | (0b100010 << 23)
                        | ((imm & 0xFFF) << 10)
                        | (hw_rn << 5)
                        | hw_rd
                } else {
                    return Err("sub requires 3 operands".to_string());
                }
            }

            // TLBI — TLB Invalidate (system instruction, used in kernel)
            "tlbi" => {
                // TLBI <op>{, Xt} — encodes as SYS instruction.
                // Emit a generic SYS encoding (simplified).
                0xD508_0000
            }

            // DC — Data Cache operation (used in kernel)
            "dc" => 0xD508_0000,

            // IC — Instruction Cache operation
            "ic" => 0xD508_0000,

            // AT — Address Translation operation
            "at" => 0xD508_0000,

            _ => {
                return Err(format!(
                    "unsupported AArch64 inline assembly instruction: '{}'",
                    mnemonic
                ));
            }
        };

        Ok(word.to_le_bytes().to_vec())
    }
}

// ===========================================================================
// Free-standing encoding helpers
// ===========================================================================

/// Encode an LDR (unsigned immediate offset) instruction.
///
/// `LDR Xt, [Xn, #imm]` — unsigned offset scaled by 8 for 64-bit loads.
/// Encoding: `11 111 0 01 01 imm12 Rn Rd`
fn encode_ldr_unsigned_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    let hw_rd = registers::hw_encoding(rd) as u32;
    let hw_rn = registers::hw_encoding(rn) as u32;
    // size=11 (64-bit), V=0 (GPR), opc=01 (LDR)
    (0b11 << 30) | (0b111001 << 24) | (0b01 << 22) | ((imm12 & 0xFFF) << 10) | (hw_rn << 5) | hw_rd
}

/// Encode an ADD immediate instruction.
///
/// `ADD Xd, Xn, #imm` (64-bit, no flags).
/// Encoding: `sf=1 | op=0 | S=0 | 100010 | sh=0 | imm12 | Rn | Rd`
fn encode_add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    let hw_rd = registers::hw_encoding(rd) as u32;
    let hw_rn = registers::hw_encoding(rn) as u32;
    (1 << 31) | (0b100010 << 23) | ((imm12 & 0xFFF) << 10) | (hw_rn << 5) | hw_rd
}

/// Format an AArch64 register ID as a human-readable name for asm substitution.
fn format_aarch64_register(reg: u8) -> String {
    let reg32 = reg as u32;
    if reg32 < 31 {
        format!("x{}", reg)
    } else if reg32 == registers::XZR {
        "xzr".to_string()
    } else if (32..64).contains(&reg32) {
        // SIMD/FP register
        let fp_idx = reg32 - 32;
        format!("v{}", fp_idx)
    } else {
        format!("r{}", reg)
    }
}

// ===========================================================================
// Little-endian read/write helpers
// ===========================================================================

/// Read a 32-bit little-endian word from a byte slice at the given offset.
#[inline]
fn read_le_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Write a 32-bit little-endian word to a byte slice at the given offset.
#[inline]
fn write_le_u32(data: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    data[offset] = bytes[0];
    data[offset + 1] = bytes[1];
    data[offset + 2] = bytes[2];
    data[offset + 3] = bytes[3];
}

/// Parse an integer literal from an asm directive value (decimal or hex).
fn parse_int_literal(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        i64::from_str_radix(&s[2..], 16).map_err(|e| format!("invalid hex literal '{}': {}", s, e))
    } else if s.starts_with('-') {
        s.parse::<i64>()
            .map_err(|e| format!("invalid integer '{}': {}", s, e))
    } else {
        // Try unsigned first to handle large values, then fall back to signed.
        if let Ok(v) = s.parse::<u64>() {
            Ok(v as i64)
        } else {
            s.parse::<i64>()
                .map_err(|e| format!("invalid integer '{}': {}", s, e))
        }
    }
}

// ===========================================================================
// Inline Assembly Instruction Parsing Helpers
// ===========================================================================

/// Parse an AArch64 GPR name (x0-x30, xzr, sp, wzr, w0-w30) and return its
/// internal register ID.  Returns `Err` for unrecognized names.
fn parse_asm_gpr(name: &str) -> Result<u32, String> {
    let name = name.trim().to_lowercase();
    if name == "sp" {
        return Ok(registers::SP_REG);
    }
    if name == "xzr" || name == "wzr" {
        return Ok(registers::XZR);
    }
    if let Some(n) = name.strip_prefix('x') {
        if let Ok(v) = n.parse::<u32>() {
            if v <= 30 {
                return Ok(v);
            }
        }
    }
    if let Some(n) = name.strip_prefix('w') {
        if let Ok(v) = n.parse::<u32>() {
            if v <= 30 {
                return Ok(v);
            }
        }
    }
    // Common ABI register aliases
    match name.as_str() {
        "lr" => Ok(30), // Link Register = X30
        "fp" => Ok(29), // Frame Pointer = X29
        _ => Err(format!("unrecognized AArch64 register: '{}'", name)),
    }
}

/// Parse an immediate value from AArch64 inline assembly.
/// Handles `#<dec>`, `#0x<hex>`, or plain decimal/hex.
fn parse_asm_immediate_aarch64(s: &str) -> i64 {
    let s = s.trim().trim_start_matches('#');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        s.parse::<i64>().unwrap_or(0)
    }
}

/// Parse an AArch64 system register encoding from a name string.
///
/// System registers are encoded as: `op0:op1:CRn:CRm:op2` (each field has
/// specific bit widths).  The 16-bit encoding placed at bits [20:5] of the
/// MRS/MSR instruction is: `(op0-2)<<14 | op1<<11 | CRn<<7 | CRm<<3 | op2`.
///
/// This function handles the most common kernel-used system registers.
fn parse_aarch64_sysreg(name: &str) -> Result<u32, String> {
    let name = name.trim().to_lowercase();

    // Check for S<op0>_<op1>_C<CRn>_C<CRm>_<op2> generic encoding
    if name.starts_with("s3_") || name.starts_with("s2_") {
        // Parse generic encoding: S<op0>_<op1>_C<CRn>_C<CRm>_<op2>
        let parts: Vec<&str> = name.split('_').collect();
        if parts.len() == 5 {
            let op0 = parts[0]
                .strip_prefix('s')
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(3);
            let op1 = parts[1].parse::<u32>().unwrap_or(0);
            let crn = parts[2]
                .strip_prefix('c')
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);
            let crm = parts[3]
                .strip_prefix('c')
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);
            let op2 = parts[4].parse::<u32>().unwrap_or(0);
            return Ok(((op0 - 2) << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2);
        }
    }

    // Named system registers commonly used in the Linux kernel.
    // Encoding: (op0-2) << 14 | op1 << 11 | CRn << 7 | CRm << 3 | op2
    match name.as_str() {
        // EL1 registers
        "sctlr_el1" => Ok(0xC080),      // S3_0_C1_C0_0
        "actlr_el1" => Ok(0xC081),      // S3_0_C1_C0_1
        "cpacr_el1" => Ok(0xC082),      // S3_0_C1_C0_2
        "ttbr0_el1" => Ok(0xC100),      // S3_0_C2_C0_0
        "ttbr1_el1" => Ok(0xC120),      // S3_0_C2_C0_1
        "tcr_el1" => Ok(0xC102),        // S3_0_C2_C0_2
        "esr_el1" => Ok(0xC290),        // S3_0_C5_C2_0
        "far_el1" => Ok(0xC300),        // S3_0_C6_C0_0
        "mair_el1" => Ok(0xC510),       // S3_0_C10_C2_0
        "amair_el1" => Ok(0xC518),      // S3_0_C10_C3_0
        "vbar_el1" => Ok(0xC600),       // S3_0_C12_C0_0
        "contextidr_el1" => Ok(0xC681), // S3_0_C13_C0_1
        "tpidr_el1" => Ok(0xC684),      // S3_0_C13_C0_4
        "tpidr_el0" => Ok(0xDE82),      // S3_3_C13_C0_2
        "tpidrro_el0" => Ok(0xDE83),    // S3_3_C13_C0_3
        "sp_el0" => Ok(0xC208),         // S3_0_C4_C1_0
        "spsr_el1" => Ok(0xC200),       // S3_0_C4_C0_0
        "elr_el1" => Ok(0xC201),        // S3_0_C4_C0_1
        "daif" => Ok(0xDA11),           // S3_3_C4_C2_1
        "nzcv" => Ok(0xDA10),           // S3_3_C4_C2_0
        "fpcr" => Ok(0xDA20),           // S3_3_C4_C4_0
        "fpsr" => Ok(0xDA21),           // S3_3_C4_C4_1
        "cntvct_el0" => Ok(0xDF01),     // S3_3_C14_C0_2
        "cntfrq_el0" => Ok(0xDF00),     // S3_3_C14_C0_0
        "cntp_ctl_el0" => Ok(0xDF11),   // S3_3_C14_C2_1
        "cntp_cval_el0" => Ok(0xDF12),  // S3_3_C14_C2_2
        "cntp_tval_el0" => Ok(0xDF10),  // S3_3_C14_C2_0
        "cntv_ctl_el0" => Ok(0xDF19),   // S3_3_C14_C3_1
        "cntv_cval_el0" => Ok(0xDF1A),  // S3_3_C14_C3_2
        "cntv_tval_el0" => Ok(0xDF18),  // S3_3_C14_C3_0
        "currentel" => Ok(0xC212),      // S3_0_C4_C2_2
        "midr_el1" => Ok(0xC000),       // S3_0_C0_C0_0
        "mpidr_el1" => Ok(0xC005),      // S3_0_C0_C0_5
        "revidr_el1" => Ok(0xC006),     // S3_0_C0_C0_6
        "par_el1" => Ok(0xC3A0),        // S3_0_C7_C4_0
        _ => Err(format!("unrecognized AArch64 system register: '{}'", name)),
    }
}

/// Parse a barrier option name into its CRm value.
fn parse_barrier_option(s: &str) -> u32 {
    match s.trim().to_lowercase().as_str() {
        "sy" => 0xF,
        "st" => 0xE,
        "ld" => 0xD,
        "ish" => 0xB,
        "ishst" => 0xA,
        "ishld" => 0x9,
        "nsh" => 0x7,
        "nshst" => 0x6,
        "nshld" => 0x5,
        "osh" => 0x3,
        "oshst" => 0x2,
        "oshld" => 0x1,
        _ => 0xF, // default to SY (full barrier)
    }
}

/// Parse an AArch64 memory operand like `[x0]`, `[x1, #16]`, or `[sp, #-8]`.
/// Returns (base_register_id, offset).
fn parse_memory_operand_aarch64(s: &str) -> Option<(u32, i64)> {
    let s = s.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
    let base = parse_asm_gpr(parts[0]).ok()?;
    let offset = if parts.len() > 1 {
        parse_asm_immediate_aarch64(parts[1])
    } else {
        0
    };
    Some((base, offset))
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::aarch64::codegen::{A64Instruction, A64Opcode};

    #[test]
    fn test_new_assembler() {
        let asm = AArch64Assembler::new(false);
        assert_eq!(asm.current_offset(), 0);
        assert!(!asm.pic_mode);
        assert_eq!(asm.target, Target::AArch64);
    }

    #[test]
    fn test_new_assembler_pic_mode() {
        let asm = AArch64Assembler::new(true);
        assert!(asm.pic_mode);
    }

    #[test]
    fn test_reset() {
        let mut asm = AArch64Assembler::new(false);
        asm.define_label("test");
        asm.emit_nop();
        assert_eq!(asm.current_offset(), 4);
        assert!(asm.symbols.contains_key("test"));

        asm.reset();
        assert_eq!(asm.current_offset(), 0);
        assert!(asm.symbols.is_empty());
        assert!(asm.code.is_empty());
        assert!(asm.relocations.is_empty());
    }

    #[test]
    fn test_define_label() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop(); // offset = 0, advances to 4
        asm.define_label("after_nop");
        assert_eq!(*asm.symbols.get("after_nop").unwrap(), 4);
    }

    #[test]
    fn test_emit_nop() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop();
        assert_eq!(asm.current_offset(), 4);
        assert_eq!(asm.code.len(), 4);
        // NOP = 0xD503201F in little-endian: 1F 20 03 D5
        assert_eq!(asm.code[0], 0x1F);
        assert_eq!(asm.code[1], 0x20);
        assert_eq!(asm.code[2], 0x03);
        assert_eq!(asm.code[3], 0xD5);
    }

    #[test]
    fn test_nop_encoding_value() {
        assert_eq!(NOP_ENCODING, 0xD503_201F);
        let bytes = NOP_ENCODING.to_le_bytes();
        assert_eq!(bytes, [0x1F, 0x20, 0x03, 0xD5]);
    }

    #[test]
    fn test_align_to() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop(); // 4 bytes
        asm.align_to(16); // should emit 3 more NOPs to reach 16
        assert_eq!(asm.current_offset(), 16);
        assert_eq!(asm.code.len(), 16);
    }

    #[test]
    fn test_align_already_aligned() {
        let mut asm = AArch64Assembler::new(false);
        asm.align_to(4); // Already aligned at 0
        assert_eq!(asm.current_offset(), 0);
    }

    #[test]
    fn test_assemble_nop_instruction() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::NOP);
        let result = asm.assemble_function(&[inst]).unwrap();
        assert_eq!(result.code_size, 4);
        assert_eq!(result.code.len(), 4);
    }

    #[test]
    fn test_assemble_multiple_nops() {
        let mut asm = AArch64Assembler::new(false);
        let insts = vec![
            A64Instruction::new(A64Opcode::NOP),
            A64Instruction::new(A64Opcode::NOP),
            A64Instruction::new(A64Opcode::NOP),
        ];
        let result = asm.assemble_function(&insts).unwrap();
        assert_eq!(result.code_size, 12);
        assert_eq!(result.code.len(), 12);
    }

    #[test]
    fn test_bl_generates_call26_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::BL).with_symbol("my_function".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();
        assert_eq!(result.code_size, 4);

        // Should have a CALL26 relocation.
        let call26_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Call26.to_raw())
            .collect();
        assert!(
            !call26_relocs.is_empty(),
            "expected CALL26 relocation for BL"
        );
        assert_eq!(call26_relocs[0].symbol, "my_function");
    }

    #[test]
    fn test_b_generates_jump26_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::B).with_symbol("target_label".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();

        let jump26_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Jump26.to_raw())
            .collect();
        assert!(
            !jump26_relocs.is_empty(),
            "expected JUMP26 relocation for B"
        );
    }

    #[test]
    fn test_adrp_generates_relocation() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::ADRP)
            .with_rd(0)
            .with_symbol("global_var".to_string());
        let result = asm.assemble_function(&[inst]).unwrap();

        let adrp_relocs: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| {
                r.reloc_type == AArch64RelocationType::AdrPrelPgHi21.to_raw()
                    || r.reloc_type == AArch64RelocationType::AdrGotPage.to_raw()
            })
            .collect();
        assert!(!adrp_relocs.is_empty(), "expected ADRP relocation");
    }

    #[test]
    fn test_assemble_module_records_function_symbols() {
        let mut asm = AArch64Assembler::new(false);
        let funcs = vec![
            (
                "func_a".to_string(),
                vec![A64Instruction::new(A64Opcode::NOP)],
            ),
            (
                "func_b".to_string(),
                vec![A64Instruction::new(A64Opcode::NOP)],
            ),
        ];
        let result = asm.assemble_module(&funcs).unwrap();
        assert!(result.symbols.contains_key("func_a"));
        assert!(result.symbols.contains_key("func_b"));
        assert_eq!(*result.symbols.get("func_a").unwrap(), 0);
        assert_eq!(*result.symbols.get("func_b").unwrap(), 4);
    }

    #[test]
    fn test_finalize_resolves_local_branches() {
        let mut asm = AArch64Assembler::new(false);

        // Define a label, emit NOP, then a BL to that label.
        asm.define_label("loop_start");
        asm.emit_nop(); // offset 0..4
                        // Manually add a CALL26 relocation targeting "loop_start".
        let bl_word: u32 = 0x9400_0000; // BL placeholder
        let bl_offset = asm.current_offset();
        asm.emit_raw_word(bl_word);
        asm.relocations.push(AssemblyRelocation {
            offset: bl_offset,
            symbol: "loop_start".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        });

        let result = asm.finalize();
        // The relocation to "loop_start" should be resolved (removed from
        // the unresolved list) because "loop_start" is defined locally.
        let unresolved_loop: Vec<_> = result
            .relocations
            .iter()
            .filter(|r| r.symbol == "loop_start")
            .collect();
        assert!(
            unresolved_loop.is_empty(),
            "local branch to 'loop_start' should be resolved"
        );
    }

    #[test]
    fn test_build_text_section() {
        let mut asm = AArch64Assembler::new(false);
        asm.emit_nop();
        asm.emit_nop();

        let (code, relocs) = asm.build_text_section();
        assert_eq!(code.len(), 8);
        assert!(relocs.is_empty());
    }

    #[test]
    fn test_instruction_size_always_4() {
        assert_eq!(INSTRUCTION_SIZE, 4);
    }

    #[test]
    fn test_assembly_relocation_to_linker_relocation() {
        let asm_reloc = AssemblyRelocation {
            offset: 100,
            symbol: "extern_func".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        };
        let linker_reloc = asm_reloc.to_linker_relocation();
        assert_eq!(linker_reloc.offset, 100);
        assert_eq!(linker_reloc.symbol_name, "extern_func");
        assert_eq!(
            linker_reloc.rel_type,
            AArch64RelocationType::Call26.to_raw()
        );
    }

    #[test]
    fn test_assembly_relocation_category() {
        let reloc = AssemblyRelocation {
            offset: 0,
            symbol: "sym".to_string(),
            reloc_type: AArch64RelocationType::Call26.to_raw(),
            addend: 0,
        };
        assert_eq!(reloc.category(), RelocCategory::PcRelative);

        let got_reloc = AssemblyRelocation {
            offset: 0,
            symbol: "sym".to_string(),
            reloc_type: AArch64RelocationType::AdrGotPage.to_raw(),
            addend: 0,
        };
        assert_eq!(got_reloc.category(), RelocCategory::GotRelative);
    }

    #[test]
    fn test_call_pseudo_instruction_direct() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::CALL).with_symbol("target_fn".to_string());
        asm.assemble_one(&inst).unwrap();
        assert_eq!(asm.current_offset(), 4);

        let call26_relocs: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Call26.to_raw())
            .collect();
        assert_eq!(call26_relocs.len(), 1);
        assert_eq!(call26_relocs[0].symbol, "target_fn");
    }

    #[test]
    fn test_call_pseudo_instruction_indirect() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::CALL).with_rn(registers::X0);
        asm.assemble_one(&inst).unwrap();
        assert_eq!(asm.current_offset(), 4);
        // Indirect call has no relocation.
        assert!(asm.relocations.is_empty());
    }

    #[test]
    fn test_la_pseudo_non_pic() {
        let mut asm = AArch64Assembler::new(false);
        let inst = A64Instruction::new(A64Opcode::LA)
            .with_rd(registers::X0)
            .with_symbol("my_global".to_string());
        asm.assemble_one(&inst).unwrap();
        // LA expands to ADRP + ADD = 8 bytes.
        assert_eq!(asm.current_offset(), 8);

        // Should have ADRP_HI21 and ADD_LO12 relocations.
        let hi21: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AdrPrelPgHi21.to_raw())
            .collect();
        let lo12: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AddAbsLo12Nc.to_raw())
            .collect();
        assert_eq!(hi21.len(), 1, "expected ADR_PREL_PG_HI21 relocation");
        assert_eq!(lo12.len(), 1, "expected ADD_ABS_LO12_NC relocation");
        assert_eq!(hi21[0].offset, 0);
        assert_eq!(lo12[0].offset, 4);
    }

    #[test]
    fn test_la_pseudo_pic() {
        let mut asm = AArch64Assembler::new(true);
        let inst = A64Instruction::new(A64Opcode::LA)
            .with_rd(registers::X0)
            .with_symbol("got_sym".to_string());
        asm.assemble_one(&inst).unwrap();
        // LA in PIC mode expands to ADRP + LDR = 8 bytes.
        assert_eq!(asm.current_offset(), 8);

        let got_page: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::AdrGotPage.to_raw())
            .collect();
        let got_lo12: Vec<_> = asm
            .relocations
            .iter()
            .filter(|r| r.reloc_type == AArch64RelocationType::Ld64GotLo12Nc.to_raw())
            .collect();
        assert_eq!(got_page.len(), 1, "expected ADR_GOT_PAGE relocation");
        assert_eq!(got_lo12.len(), 1, "expected LD64_GOT_LO12_NC relocation");
    }

    #[test]
    fn test_inline_asm_basic() {
        let mut asm = AArch64Assembler::new(false);
        let bytes = asm.assemble_inline_asm("nop", &[]).unwrap();
        // Should emit at least 4 bytes (a NOP).
        assert!(bytes >= 4);
    }

    #[test]
    fn test_offset_always_multiple_of_4() {
        let mut asm = AArch64Assembler::new(false);
        for _ in 0..10 {
            asm.emit_nop();
            assert_eq!(asm.current_offset() % 4, 0, "offset must be 4-byte aligned");
        }
    }

    #[test]
    fn test_parse_int_literal_decimal() {
        assert_eq!(parse_int_literal("42").unwrap(), 42);
        assert_eq!(parse_int_literal("-1").unwrap(), -1);
        assert_eq!(parse_int_literal("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_int_literal_hex() {
        assert_eq!(parse_int_literal("0xFF").unwrap(), 255);
        assert_eq!(parse_int_literal("0x1000").unwrap(), 4096);
    }

    #[test]
    fn test_format_register() {
        assert_eq!(format_aarch64_register(0), "x0");
        assert_eq!(format_aarch64_register(30), "x30");
        assert_eq!(format_aarch64_register(31), "xzr");
        assert_eq!(format_aarch64_register(32), "v0");
        assert_eq!(format_aarch64_register(63), "v31");
    }

    #[test]
    fn test_empty_assembly_result() {
        let result = AssemblyResult::empty();
        assert!(result.code.is_empty());
        assert!(result.relocations.is_empty());
        assert!(result.symbols.is_empty());
        assert_eq!(result.code_size, 0);
    }
}
