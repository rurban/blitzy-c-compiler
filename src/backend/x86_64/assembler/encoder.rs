//! x86-64 instruction encoder — binary encoding engine for the built-in assembler.
//!
//! This module converts x86-64 machine instructions (represented as
//! [`MachineInstruction`] with [`X86Opcode`] opcodes) into their binary byte
//! representation. It handles:
//!
//! - **REX prefix generation** (0x40–0x4F): REX.W for 64-bit operand size,
//!   REX.R/X/B for extended register encoding (R8–R15, XMM8–XMM15)
//! - **ModR/M byte construction**: 2-bit mod + 3-bit reg + 3-bit r/m fields
//! - **SIB byte construction**: 2-bit scale + 3-bit index + 3-bit base fields
//! - **Displacement encoding**: 8-bit or 32-bit signed displacements
//! - **Immediate encoding**: 8/16/32/64-bit immediate values
//! - **Relocation emission**: Creates [`RelocationEntry`] records for symbol references
//! - **SSE/SSE2 instruction encoding**: Mandatory prefix handling (0xF2/0xF3/0x66)
//!
//! ## x86-64 Instruction Format (in order)
//!
//! 1. Legacy prefixes (0–4 bytes): 0x66, 0x67, 0xF2, 0xF3, segment overrides
//! 2. REX prefix (0–1 byte): 0100 WRXB
//! 3. Opcode (1–3 bytes)
//! 4. ModR/M (0–1 byte): mod(2) + reg(3) + r/m(3)
//! 5. SIB (0–1 byte): scale(2) + index(3) + base(3)
//! 6. Displacement (0/1/4 bytes)
//! 7. Immediate (0/1/2/4/8 bytes)
//!
//! ## Zero-Dependency Mandate
//!
//! Only `std` and `crate::` references. No external crates.

use super::relocations::{RelocationEntry, X86_64RelocationType};
use crate::backend::traits::{MachineInstruction, MachineOperand};
use crate::backend::x86_64::codegen::X86Opcode;
use crate::backend::x86_64::registers::{
    hw_encoding, is_gpr, is_sse, needs_rex, needs_rex_for_byte_reg, OPERAND_SIZE_PREFIX, R10, R11,
    R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, XMM0, XMM8,
};
use crate::common::fx_hash::FxHashMap;

// ===========================================================================
// EncodedInstruction — encoder output
// ===========================================================================

/// Result of encoding a single x86-64 instruction.
///
/// Contains the raw machine code bytes (1–15 bytes per x86-64 instruction)
/// and any relocation entries generated for symbol references.
#[derive(Debug, Clone)]
pub struct EncodedInstruction {
    /// The raw bytes of the encoded instruction (1–15 bytes for x86-64).
    pub bytes: Vec<u8>,
    /// Relocation entries generated during encoding (e.g., for external
    /// symbol references in call/jmp/mov instructions).
    pub relocations: Vec<RelocationEntry>,
}

impl EncodedInstruction {
    /// Create a new encoded instruction with the given bytes and no relocations.
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            relocations: Vec::new(),
        }
    }

    /// Create a new encoded instruction with bytes and relocations.
    fn with_relocations(bytes: Vec<u8>, relocations: Vec<RelocationEntry>) -> Self {
        Self { bytes, relocations }
    }
}

// ===========================================================================
// MemoryEncoding — memory operand encoding result
// ===========================================================================

/// Result of encoding a memory operand into ModR/M, optional SIB, and
/// displacement bytes.
#[derive(Debug, Clone)]
pub struct MemoryEncoding {
    /// The ModR/M byte encoding the memory addressing mode.
    pub modrm: u8,
    /// Optional SIB byte (present when base is RSP/R12 or scaled-index
    /// addressing is used).
    pub sib: Option<u8>,
    /// Displacement bytes (0, 1, or 4 bytes depending on the mod field).
    pub displacement: Vec<u8>,
}

// ===========================================================================
// EncodingForm — instruction encoding formats
// ===========================================================================

/// x86-64 instruction encoding forms.
///
/// Determines how operands are mapped into the ModR/M byte fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingForm {
    /// Register in reg field, register/memory in r/m field.
    RegRm,
    /// Register/memory in r/m field, register in reg field.
    RmReg,
    /// Register/memory in r/m field, immediate follows.
    RmImm,
    /// R/m only — opcode extension in reg field.
    RmOnly,
    /// Register encoded in lower 3 bits of opcode (e.g., PUSH r64).
    OpcodeReg,
    /// No operands (e.g., RET, NOP, CDQ).
    NoOperands,
    /// Relative offset (e.g., JMP rel32, CALL rel32).
    Relative,
    /// Special encoding handled case-by-case.
    Special,
}

// ===========================================================================
// OpcodeEntry — opcode table entry
// ===========================================================================

/// Encoding metadata for an x86-64 opcode.
///
/// Maps an [`X86Opcode`] variant to its binary encoding information,
/// enabling table-driven instruction encoding.
#[derive(Debug, Clone)]
pub struct OpcodeEntry {
    /// Primary opcode byte(s) (1–3 bytes).
    pub opcode: &'static [u8],
    /// Opcode extension in ModR/M reg field (`/0`–`/7`), if used.
    pub reg_opext: Option<u8>,
    /// Whether this instruction defaults to 64-bit operand size (REX.W).
    pub default_64bit: bool,
    /// The instruction encoding form.
    pub form: EncodingForm,
}

// ===========================================================================
// REX Prefix Construction
// ===========================================================================

/// Construct a REX prefix byte.
///
/// REX format: 0100 WRXB
/// - W: 1 = 64-bit operand size
/// - R: extension of ModR/M reg field (selects R8–R15)
/// - X: extension of SIB index field
/// - B: extension of ModR/M r/m field or SIB base field
#[inline]
fn rex_byte(w: bool, r: bool, x: bool, b: bool) -> u8 {
    let mut rex: u8 = 0x40;
    if w {
        rex |= 0x08;
    }
    if r {
        rex |= 0x04;
    }
    if x {
        rex |= 0x02;
    }
    if b {
        rex |= 0x01;
    }
    rex
}

/// Determine if a REX prefix is needed and construct it.
///
/// Returns `Some(rex_byte)` if any REX bit is set, `None` otherwise.
/// REX.W forces 64-bit operand size, REX.R extends the reg field,
/// REX.X extends the SIB index field, and REX.B extends the r/m or
/// SIB base field.
fn compute_rex(
    operand_size_64: bool,
    reg: Option<u16>,
    index: Option<u16>,
    rm_or_base: Option<u16>,
) -> Option<u8> {
    let w = operand_size_64;
    let r = reg.map_or(false, needs_rex);
    let x = index.map_or(false, needs_rex);
    let b = rm_or_base.map_or(false, needs_rex);
    if w || r || x || b {
        Some(rex_byte(w, r, x, b))
    } else {
        None
    }
}

/// Determine if a REX prefix is needed, forcing one for byte-sized operand
/// access on registers RSP/RBP/RSI/RDI (hw encodings 4–7).
///
/// On x86-64, byte-sized register operand encodings 4–7 without REX select
/// the legacy high-byte registers %ah/%ch/%dh/%bh. With any REX prefix they
/// select the low-byte forms %spl/%bpl/%sil/%dil. Callers pass the register
/// operands that participate in the byte operation via `byte_regs`.
fn compute_rex_byte_aware(
    operand_size_64: bool,
    reg: Option<u16>,
    index: Option<u16>,
    rm_or_base: Option<u16>,
    byte_regs: &[u16],
) -> Option<u8> {
    let force = byte_regs.iter().any(|r| needs_rex_for_byte_reg(*r));
    let w = operand_size_64;
    let r = reg.map_or(false, needs_rex);
    let x = index.map_or(false, needs_rex);
    let b = rm_or_base.map_or(false, needs_rex);
    if w || r || x || b || force {
        Some(rex_byte(w, r, x, b))
    } else {
        None
    }
}

// ===========================================================================
// ModR/M Byte Construction
// ===========================================================================

/// Construct a ModR/M byte.
///
/// ModR/M format: [mod(2)][reg(3)][r/m(3)]
/// - mod: 00=no disp, 01=disp8, 10=disp32, 11=reg-reg
/// - reg: register operand or opcode extension
/// - r/m: register or memory operand
#[inline]
fn modrm_byte(mod_bits: u8, reg: u8, rm: u8) -> u8 {
    ((mod_bits & 0x3) << 6) | ((reg & 0x7) << 3) | (rm & 0x7)
}

// ===========================================================================
// SIB Byte Construction
// ===========================================================================

/// Construct a SIB byte.
///
/// SIB format: [scale(2)][index(3)][base(3)]
/// - scale: 00=×1, 01=×2, 10=×4, 11=×8
/// - index: index register (100b = no index)
/// - base: base register (101b with mod=00 = disp32 only)
#[inline]
fn sib_byte(scale: u8, index: u8, base: u8) -> u8 {
    ((scale & 0x3) << 6) | ((index & 0x7) << 3) | (base & 0x7)
}

/// Convert scale factor (1, 2, 4, 8) to SIB scale encoding (0, 1, 2, 3).
#[inline]
fn scale_encoding(scale: u8) -> u8 {
    match scale {
        1 => 0b00,
        2 => 0b01,
        4 => 0b10,
        8 => 0b11,
        _ => 0b00, // default to ×1 for invalid scales
    }
}

// ===========================================================================
// Displacement and Immediate Encoding
// ===========================================================================

/// Encode a displacement value as 1 or 4 bytes.
///
/// Uses disp8 (sign-extended) if the value fits in [-128, 127],
/// otherwise disp32 (4 bytes, little-endian).
#[allow(dead_code)]
fn encode_displacement(disp: i64) -> Vec<u8> {
    if (-128..=127).contains(&disp) {
        vec![disp as i8 as u8]
    } else {
        (disp as i32).to_le_bytes().to_vec()
    }
}

/// Encode an immediate value in the specified number of bytes (little-endian).
fn encode_immediate(imm: i64, size: u8) -> Vec<u8> {
    match size {
        1 => vec![imm as u8],
        2 => (imm as i16).to_le_bytes().to_vec(),
        4 => (imm as i32).to_le_bytes().to_vec(),
        8 => imm.to_le_bytes().to_vec(),
        _ => vec![imm as u8],
    }
}

/// Compute mod bits and displacement bytes for a memory operand.
///
/// - If displacement is 0 and base encoding is not 5 (RBP/R13): mod=00, no disp
/// - If displacement fits in i8: mod=01, 1-byte disp
/// - Otherwise: mod=10, 4-byte disp
///
/// Note: base_enc==5 with mod=00 means RIP-relative addressing, so we
/// must use mod=01 with disp8=0 when the actual displacement is 0 but
/// the base is RBP or R13.
fn compute_mod_disp(displacement: i64, base_enc: u8) -> (u8, Vec<u8>) {
    if displacement == 0 && base_enc != 5 {
        (0b00, vec![])
    } else if (-128..=127).contains(&displacement) {
        (0b01, vec![displacement as i8 as u8])
    } else {
        (0b10, (displacement as i32).to_le_bytes().to_vec())
    }
}

// ===========================================================================
// Memory Operand Encoding
// ===========================================================================

/// Encode a memory operand into ModR/M + optional SIB + displacement.
///
/// Handles all x86-64 memory addressing modes:
/// - `[base]` — simple base register
/// - `[base + disp8/disp32]` — base + displacement
/// - `[base + index*scale + disp]` — SIB addressing
/// - `[disp32]` — absolute addressing (via SIB escape)
///
/// Special encoding rules:
/// - RSP/R12 (hw_encoding & 7 == 4) as base always require SIB byte
/// - RBP/R13 (hw_encoding & 7 == 5) as base with 0 displacement use mod=01
/// - RSP (hw_encoding 4) as SIB index means "no index register"
fn encode_memory_operand(
    reg_or_opext: u8,
    base: Option<u16>,
    index: Option<u16>,
    scale: u8,
    displacement: i64,
) -> MemoryEncoding {
    let base_enc_low = base.map(|b| hw_encoding(b) & 0x7);
    let has_real_index = index.is_some() && index.map(|i| hw_encoding(i) & 0x7).unwrap_or(4) != 4;

    let need_sib = has_real_index || base_enc_low == Some(SIB_REQUIRED_ENC) || base.is_none();

    if need_sib {
        let index_enc = match index {
            Some(idx) if hw_encoding(idx) & 0x7 != 4 => hw_encoding(idx) & 0x7,
            _ => 0b100, // 100 = no index
        };
        let scale_enc = if has_real_index {
            scale_encoding(scale)
        } else {
            0b00
        };

        match base {
            None => {
                // No base: mod=00, rm=100 (SIB), SIB base=101 = disp32 only
                MemoryEncoding {
                    modrm: modrm_byte(0b00, reg_or_opext, 0b100),
                    sib: Some(sib_byte(scale_enc, index_enc, 0b101)),
                    displacement: (displacement as i32).to_le_bytes().to_vec(),
                }
            }
            Some(b) => {
                let b_enc = hw_encoding(b) & 0x7;
                let (mod_bits, disp_bytes) = compute_mod_disp(displacement, b_enc);
                MemoryEncoding {
                    modrm: modrm_byte(mod_bits, reg_or_opext, 0b100),
                    sib: Some(sib_byte(scale_enc, index_enc, b_enc)),
                    displacement: disp_bytes,
                }
            }
        }
    } else {
        // No SIB byte needed
        let base_reg = base.expect("Base register required without SIB");
        let rm = hw_encoding(base_reg) & 0x7;
        let (mod_bits, disp_bytes) = compute_mod_disp(displacement, rm);
        MemoryEncoding {
            modrm: modrm_byte(mod_bits, reg_or_opext, rm),
            sib: None,
            displacement: disp_bytes,
        }
    }
}

// ===========================================================================
// Condition Code and ALU Helpers
// ===========================================================================

/// Map an X86Opcode conditional variant to its x86-64 condition code (cc).
///
/// All 16 x86 condition codes are covered (Intel SDM Vol. 2, Appendix B):
///
/// | cc   | Code | Mnemonic | Condition                                |
/// |------|------|----------|------------------------------------------|
/// | 0x00 | O    | JO/SETO  | Overflow (OF=1)                          |
/// | 0x01 | NO   | JNO      | Not overflow (OF=0)                      |
/// | 0x02 | B/C  | JB/SETB  | Below / Carry (CF=1)                     |
/// | 0x03 | AE/NC| JAE      | Above or equal / No carry (CF=0)         |
/// | 0x04 | E/Z  | JE/SETE  | Equal / Zero (ZF=1)                      |
/// | 0x05 | NE/NZ| JNE/SETNE| Not equal / Not zero (ZF=0)              |
/// | 0x06 | BE/NA| JBE/SETBE| Below or equal (CF=1 OR ZF=1)            |
/// | 0x07 | A/NBE| JA/SETA  | Above (CF=0 AND ZF=0)                    |
/// | 0x08 | S    | JS/SETS  | Sign (SF=1)                              |
/// | 0x09 | NS   | JNS/SETNS| Not sign (SF=0)                          |
/// | 0x0A | P/PE | JP/SETP  | Parity even (PF=1) — also used for NaN   |
/// | 0x0B | NP/PO| JNP/SETNP| Parity odd (PF=0) — used for ordered cmp |
/// | 0x0C | L/NGE| JL/SETL  | Less (SF≠OF)                             |
/// | 0x0D | GE/NL| JGE/SETGE| Greater or equal (SF=OF)                 |
/// | 0x0E | LE/NG| JLE/SETLE| Less or equal (ZF=1 OR SF≠OF)           |
/// | 0x0F | G/NLE| JG/SETG  | Greater (ZF=0 AND SF=OF)                 |
fn condition_code(op: &X86Opcode) -> u8 {
    match op {
        // 0x00 — Overflow
        X86Opcode::Jo | X86Opcode::Cmovo | X86Opcode::Seto => 0x00,
        // 0x01 — Not overflow
        X86Opcode::Jno | X86Opcode::Cmovno | X86Opcode::Setno => 0x01,
        // 0x02 — Below / Carry (unsigned <)
        X86Opcode::Jb | X86Opcode::Cmovb | X86Opcode::Setb => 0x02,
        // 0x03 — Above or equal / No carry (unsigned >=)
        X86Opcode::Jae | X86Opcode::Cmovae | X86Opcode::Setae => 0x03,
        // 0x04 — Equal / Zero
        X86Opcode::Je | X86Opcode::Cmove | X86Opcode::Sete => 0x04,
        // 0x05 — Not equal / Not zero
        X86Opcode::Jne | X86Opcode::Cmovne | X86Opcode::Setne => 0x05,
        // 0x06 — Below or equal (unsigned <=)
        X86Opcode::Jbe | X86Opcode::Cmovbe | X86Opcode::Setbe => 0x06,
        // 0x07 — Above (unsigned >)
        X86Opcode::Ja | X86Opcode::Cmova | X86Opcode::Seta => 0x07,
        // 0x08 — Sign
        X86Opcode::Js | X86Opcode::Cmovs | X86Opcode::Sets => 0x08,
        // 0x09 — Not sign
        X86Opcode::Jns | X86Opcode::Cmovns | X86Opcode::Setns => 0x09,
        // 0x0A — Parity even (PF=1)
        X86Opcode::Jp | X86Opcode::Cmovp | X86Opcode::Setp => 0x0A,
        // 0x0B — Parity odd / not parity (PF=0)
        X86Opcode::Jnp | X86Opcode::Cmovnp | X86Opcode::Setnp => 0x0B,
        // 0x0C — Less (signed)
        X86Opcode::Jl | X86Opcode::Cmovl | X86Opcode::Setl => 0x0C,
        // 0x0D — Greater or equal (signed)
        X86Opcode::Jge | X86Opcode::Cmovge | X86Opcode::Setge => 0x0D,
        // 0x0E — Less or equal (signed)
        X86Opcode::Jle | X86Opcode::Cmovle | X86Opcode::Setle => 0x0E,
        // 0x0F — Greater (signed)
        X86Opcode::Jg | X86Opcode::Cmovg | X86Opcode::Setg => 0x0F,
        // Fallback for any opcode passed incorrectly — Equal as safe default.
        _ => 0x04,
    }
}

/// Return (opcode_extension, rm_reg_opcode, reg_rm_opcode) for ALU ops.
fn alu_info(op: &X86Opcode) -> Option<(u8, u8, u8)> {
    match op {
        X86Opcode::Add => Some((0, 0x01, 0x03)),
        X86Opcode::Or => Some((1, 0x09, 0x0B)),
        X86Opcode::And => Some((4, 0x21, 0x23)),
        X86Opcode::Sub => Some((5, 0x29, 0x2B)),
        X86Opcode::Xor => Some((6, 0x31, 0x33)),
        X86Opcode::Cmp => Some((7, 0x39, 0x3B)),
        _ => None,
    }
}

/// Return the opcode extension for shift instructions.
fn shift_opext(op: &X86Opcode) -> u8 {
    match op {
        X86Opcode::Shl => 4,
        X86Opcode::Shr => 5,
        X86Opcode::Sar => 7,
        _ => 4,
    }
}

// ===========================================================================
// Opcode Table Construction
// ===========================================================================

/// Build the opcode table mapping X86Opcode discriminants to their
/// binary encoding metadata using FxHashMap for O(1) lookup.
fn build_opcode_table() -> FxHashMap<u32, OpcodeEntry> {
    let mut table: FxHashMap<u32, OpcodeEntry> = FxHashMap::default();

    table.insert(
        X86Opcode::Ret.as_u32(),
        OpcodeEntry {
            opcode: &[0xC3],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Nop.as_u32(),
        OpcodeEntry {
            opcode: &[0x90],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Cdq.as_u32(),
        OpcodeEntry {
            opcode: &[0x99],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Cqo.as_u32(),
        OpcodeEntry {
            opcode: &[0x99],
            reg_opext: None,
            default_64bit: true,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Endbr64.as_u32(),
        OpcodeEntry {
            opcode: &[0xF3, 0x0F, 0x1E, 0xFA],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Pause.as_u32(),
        OpcodeEntry {
            opcode: &[0xF3, 0x90],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Lfence.as_u32(),
        OpcodeEntry {
            opcode: &[0x0F, 0xAE, 0xE8],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Ud2.as_u32(),
        OpcodeEntry {
            opcode: &[0x0F, 0x0B],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Leave.as_u32(),
        OpcodeEntry {
            opcode: &[0xC9],
            reg_opext: None,
            default_64bit: false,
            form: EncodingForm::NoOperands,
        },
    );
    table.insert(
        X86Opcode::Neg.as_u32(),
        OpcodeEntry {
            opcode: &[0xF7],
            reg_opext: Some(3),
            default_64bit: true,
            form: EncodingForm::RmOnly,
        },
    );
    table.insert(
        X86Opcode::Not.as_u32(),
        OpcodeEntry {
            opcode: &[0xF7],
            reg_opext: Some(2),
            default_64bit: true,
            form: EncodingForm::RmOnly,
        },
    );
    table.insert(
        X86Opcode::Idiv.as_u32(),
        OpcodeEntry {
            opcode: &[0xF7],
            reg_opext: Some(7),
            default_64bit: true,
            form: EncodingForm::RmOnly,
        },
    );
    table.insert(
        X86Opcode::Div.as_u32(),
        OpcodeEntry {
            opcode: &[0xF7],
            reg_opext: Some(6),
            default_64bit: true,
            form: EncodingForm::RmOnly,
        },
    );

    table
}

// ===========================================================================
// Register Encoding Constants
// ===========================================================================

/// Registers whose hardware encoding & 7 == 4 require SIB byte.
const SIB_REQUIRED_ENC: u8 = 4;

/// Registers whose hardware encoding & 7 == 5 conflict with RIP-relative.
const RIP_CONFLICT_ENC: u8 = 5;

/// Compile-time verification of register encoding properties.
#[allow(dead_code)]
const _: () = {
    assert!(RSP as u8 % 8 == SIB_REQUIRED_ENC);
    assert!(R12 as u8 % 8 == SIB_REQUIRED_ENC);
    assert!(RBP as u8 % 8 == RIP_CONFLICT_ENC);
    assert!(R13 as u8 % 8 == RIP_CONFLICT_ENC);
};

// ===========================================================================
// X86_64Encoder — main encoder struct
// ===========================================================================

/// x86-64 instruction encoder.
///
/// Encodes x86-64 machine instructions into their binary representation,
/// handling REX prefixes, ModR/M bytes, SIB bytes, displacements, and
/// immediates. Tracks the current offset within the `.text` section for
/// relocation offset computation.
pub struct X86_64Encoder {
    /// Current byte offset in the `.text` section. Updated after each
    /// instruction is encoded so that relocation offsets are correct.
    pub current_offset: usize,
    /// Opcode table for O(1) lookup of encoding metadata.
    opcode_table: FxHashMap<u32, OpcodeEntry>,
    /// Whether position-independent code generation is active (`-fPIC`).
    /// When true, global data references use GOT-relative relocations
    /// (R_X86_64_REX_GOTPCRELX) instead of direct PC-relative
    /// (R_X86_64_PC32).
    pub pic_enabled: bool,
}

impl X86_64Encoder {
    /// Create a new encoder starting at the given text section offset.
    pub fn new(current_offset: usize) -> Self {
        // Validate register encoding properties at runtime (debug only).
        debug_assert!(!needs_rex(RAX) && !needs_rex(RBX));
        debug_assert!(!needs_rex(RCX) && !needs_rex(RDX));
        debug_assert!(!needs_rex(RSI) && !needs_rex(RDI));
        debug_assert!(needs_rex(R8) && needs_rex(R9));
        debug_assert!(needs_rex(R10) && needs_rex(R11));
        debug_assert!(needs_rex(R14) && needs_rex(R15));
        debug_assert!(is_gpr(RAX) && is_sse(XMM0) && is_sse(XMM8));
        debug_assert_eq!(OPERAND_SIZE_PREFIX, 0x66);

        Self {
            current_offset,
            opcode_table: build_opcode_table(),
            pic_enabled: false,
        }
    }

    // ===================================================================
    // Main Instruction Encoding Entry Point
    // ===================================================================

    /// Encode a single machine instruction into bytes.
    ///
    /// This is the main dispatch function called from `mod.rs` for each
    /// [`MachineInstruction`] in the function. It converts the opcode `u32`
    /// back to [`X86Opcode`], examines operand types, and delegates to
    /// the appropriate encoding helper.
    pub fn encode_instruction(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        // Normalise the operand layout.  The x86-64 codegen places the
        // destination in `inst.result` and the source(s) in
        // `inst.operands`, but the encoding helpers expect the
        // destination in `operands[0]` and the source in `operands[1]`.
        // When `inst.result` is present we prepend it to the operand
        // list so every downstream helper sees the expected convention.
        let normalised: MachineInstruction;
        let inst = if let Some(ref res) = inst.result {
            let mut ni = MachineInstruction::new(inst.opcode);
            ni.operands.push(res.clone());
            ni.operands.extend(inst.operands.iter().cloned());
            ni.is_terminator = inst.is_terminator;
            ni.is_branch = inst.is_branch;
            ni.is_call = inst.is_call;
            ni.operand_size = inst.operand_size;
            normalised = ni;
            &normalised
        } else {
            inst
        };

        let opcode = match X86Opcode::from_u32(inst.opcode) {
            Some(op) => op,
            None => {
                // Unknown opcode — emit a UD2 (0F 0B) fault instruction to
                // make encoding failures immediately visible rather than
                // silently producing NOPs that mask miscompilation.  This
                // matches the i686 encoder's policy of surfacing errors for
                // unrecognised opcodes.
                let result = {
                    if crate::common::verbosity::verbose_enabled() {
                        eprintln!(
                            "[UD2] opcode={} ops={:?} res={:?}",
                            inst.opcode, inst.operands, inst.result
                        );
                    }
                    EncodedInstruction::new(vec![0x0F, 0x0B])
                };
                self.current_offset += result.bytes.len();
                return result;
            }
        };

        // Handle no-operand instructions via opcode table first.
        if let Some(entry) = self.opcode_table.get(&inst.opcode) {
            if entry.form == EncodingForm::NoOperands {
                let mut bytes = Vec::with_capacity(8);
                if entry.default_64bit {
                    bytes.push(rex_byte(true, false, false, false));
                }
                bytes.extend_from_slice(entry.opcode);
                let result = EncodedInstruction::new(bytes);
                self.current_offset += result.bytes.len();
                return result;
            }
        }

        let result = match opcode {
            // ALU binary ops
            X86Opcode::Add
            | X86Opcode::Sub
            | X86Opcode::And
            | X86Opcode::Or
            | X86Opcode::Xor
            | X86Opcode::Cmp => {
                let (opext, rm_reg_opc, reg_rm_opc) = alu_info(&opcode).unwrap();
                self.encode_alu_op(inst, opext, rm_reg_opc, reg_rm_opc)
            }
            X86Opcode::Test => self.encode_test(inst),
            X86Opcode::Mov => self.encode_mov(inst),
            X86Opcode::Lea => self.encode_lea(inst),
            X86Opcode::Shl | X86Opcode::Shr | X86Opcode::Sar => self.encode_shift(inst, &opcode),
            X86Opcode::Neg | X86Opcode::Not | X86Opcode::Idiv | X86Opcode::Div => {
                self.encode_unary(inst, &opcode)
            }
            X86Opcode::Imul => self.encode_imul(inst),
            X86Opcode::Push => {
                match inst.operands.first() {
                    Some(MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    }) => {
                        // PUSH m64: FF /6 with ModR/M — push 8 bytes from
                        // a memory location (e.g. frame slot for struct
                        // eightbytes that must be passed on the stack).
                        // The /6 extension field is encoded by passing a
                        // pseudo-register whose hw encoding low 3 bits are
                        // 110 (= 6).  RSI has hw encoding 6.
                        // PUSH in 64-bit mode is always 64-bit (no REX.W
                        // needed), but passing size=8 is harmless.
                        self.encode_reg_mem_op(
                            &[0xFF],
                            RSI, // hw_encoding(RSI) & 7 == 6 == /6
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            0, // size 0 → no REX.W (PUSH is implicitly 64-bit)
                        )
                    }
                    Some(MachineOperand::Immediate(imm)) => {
                        // PUSH imm8: 6A ib / PUSH imm32: 68 id
                        let v = *imm;
                        if (-128..=127).contains(&v) {
                            EncodedInstruction::new(vec![0x6A, v as u8])
                        } else {
                            let mut bytes = vec![0x68];
                            bytes.extend_from_slice(&(v as i32).to_le_bytes());
                            EncodedInstruction::new(bytes)
                        }
                    }
                    _ => {
                        let reg = self.extract_register(&inst.operands, 0);
                        self.encode_push_pop(0x50, reg)
                    }
                }
            }
            X86Opcode::Pop => {
                let reg = self.extract_register(&inst.operands, 0);
                self.encode_push_pop(0x58, reg)
            }
            X86Opcode::MovZX => self.encode_movzx_movsx(inst, &[0x0F, 0xB6], &[0x0F, 0xB7]),
            X86Opcode::MovSX => self.encode_movsx(inst),
            X86Opcode::Je
            | X86Opcode::Jne
            | X86Opcode::Jl
            | X86Opcode::Jle
            | X86Opcode::Jg
            | X86Opcode::Jge
            | X86Opcode::Jb
            | X86Opcode::Jbe
            | X86Opcode::Ja
            | X86Opcode::Jae
            | X86Opcode::Jp
            | X86Opcode::Jnp
            | X86Opcode::Js
            | X86Opcode::Jns => {
                let cc = condition_code(&opcode);
                self.encode_jcc(inst, cc)
            }
            X86Opcode::Jmp => self.encode_jmp(inst),
            X86Opcode::Call => self.encode_call(inst),
            X86Opcode::Cmove
            | X86Opcode::Cmovne
            | X86Opcode::Cmovl
            | X86Opcode::Cmovle
            | X86Opcode::Cmovg
            | X86Opcode::Cmovge
            | X86Opcode::Cmovb
            | X86Opcode::Cmovbe
            | X86Opcode::Cmova
            | X86Opcode::Cmovae
            | X86Opcode::Cmovp
            | X86Opcode::Cmovnp => {
                let cc = condition_code(&opcode);
                self.encode_cmovcc(inst, cc)
            }
            X86Opcode::Seto
            | X86Opcode::Sete
            | X86Opcode::Setne
            | X86Opcode::Setl
            | X86Opcode::Setle
            | X86Opcode::Setg
            | X86Opcode::Setge
            | X86Opcode::Setb
            | X86Opcode::Setbe
            | X86Opcode::Seta
            | X86Opcode::Setae
            | X86Opcode::Setp
            | X86Opcode::Setnp => {
                let cc = condition_code(&opcode);
                // SETcc destination comes from inst.result (not operands)
                // because the instruction selector places the destination
                // in the result field, leaving operands empty.
                let dst = if let Some(MachineOperand::Register(r)) = &inst.result {
                    *r
                } else {
                    self.extract_register(&inst.operands, 0)
                };
                self.encode_setcc(inst, cc, dst)
            }
            X86Opcode::Movsd => self.encode_movsd(inst),
            X86Opcode::Movss => self.encode_movss(inst),
            X86Opcode::Addsd => self.encode_sse_op(inst, 0xF2, 0x58),
            X86Opcode::Subsd => self.encode_sse_op(inst, 0xF2, 0x5C),
            X86Opcode::Mulsd => self.encode_sse_op(inst, 0xF2, 0x59),
            X86Opcode::Divsd => self.encode_sse_op(inst, 0xF2, 0x5E),
            X86Opcode::Addss => self.encode_sse_op(inst, 0xF3, 0x58),
            X86Opcode::Subss => self.encode_sse_op(inst, 0xF3, 0x5C),
            X86Opcode::Mulss => self.encode_sse_op(inst, 0xF3, 0x59),
            X86Opcode::Divss => self.encode_sse_op(inst, 0xF3, 0x5E),
            X86Opcode::Ucomisd => self.encode_ucomisd(inst),
            X86Opcode::Ucomiss => self.encode_ucomiss(inst),
            X86Opcode::Cvtsi2sd => self.encode_cvtsi2sd(inst),
            X86Opcode::Cvtsi2ss => self.encode_cvtsi2ss(inst),
            X86Opcode::Cvtsd2si => self.encode_cvtsd2si(inst),
            X86Opcode::Cvtss2si => self.encode_cvtss2si(inst),
            X86Opcode::Cvtsd2ss => self.encode_cvtsd2ss(inst),
            X86Opcode::Cvtss2sd => self.encode_cvtss2sd(inst),
            // REP MOVSQ — repeat move qwords [RSI] → [RDI], RCX times
            X86Opcode::RepMovsq => EncodedInstruction::new(vec![0xF3, 0x48, 0xA5]),
            // Fixed-encoding instructions (fallback if not caught above)
            X86Opcode::Ret
            | X86Opcode::Nop
            | X86Opcode::Cdq
            | X86Opcode::Cqo
            | X86Opcode::Endbr64
            | X86Opcode::Pause
            | X86Opcode::Lfence
            | X86Opcode::Leave
            | X86Opcode::Ud2 => {
                if let Some(entry) = self.opcode_table.get(&inst.opcode) {
                    let mut bytes = Vec::new();
                    if entry.default_64bit {
                        bytes.push(rex_byte(true, false, false, false));
                    }
                    bytes.extend_from_slice(entry.opcode);
                    EncodedInstruction::new(bytes)
                } else {
                    {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // BSR dst, src — Bit Scan Reverse (0F BD /r)
            // NOTE: After normalization, result is in operands[0], src in operands[1].
            // encode_reg_reg_op puts param1→R/M, param2→REG.
            // BSR/BSF want dst in REG, src in R/M, so swap: (src, dst).
            X86Opcode::Bsr => {
                let dst_reg = match inst.operands.first() {
                    Some(MachineOperand::Register(r)) => *r,
                    _ => return EncodedInstruction::new(vec![0x0F, 0x0B]),
                };
                match inst.operands.get(1) {
                    Some(MachineOperand::Register(src_r)) => {
                        self.encode_reg_reg_op(&[0x0F, 0xBD], *src_r, dst_reg, 8)
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]),
                }
            }
            // BSF dst, src — Bit Scan Forward (0F BC /r)
            X86Opcode::Bsf => {
                let dst_reg = match inst.operands.first() {
                    Some(MachineOperand::Register(r)) => *r,
                    _ => return EncodedInstruction::new(vec![0x0F, 0x0B]),
                };
                match inst.operands.get(1) {
                    Some(MachineOperand::Register(src_r)) => {
                        self.encode_reg_reg_op(&[0x0F, 0xBC], *src_r, dst_reg, 8)
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]),
                }
            }
            // POPCNT dst, src — (F3 0F B8 /r)
            X86Opcode::Popcnt => {
                let dst_reg = match inst.operands.first() {
                    Some(MachineOperand::Register(r)) => *r,
                    _ => return EncodedInstruction::new(vec![0x0F, 0x0B]),
                };
                match inst.operands.get(1) {
                    Some(MachineOperand::Register(src_r)) => {
                        // POPCNT has mandatory F3 prefix before REX.
                        let mut bytes = Vec::with_capacity(8);
                        bytes.push(0xF3);
                        if let Some(rex) = compute_rex(true, Some(dst_reg), None, Some(*src_r)) {
                            bytes.push(rex);
                        }
                        bytes.push(0x0F);
                        bytes.push(0xB8);
                        bytes.push(modrm_byte(
                            0b11,
                            hw_encoding(dst_reg) & 0x7,
                            hw_encoding(*src_r) & 0x7,
                        ));
                        EncodedInstruction::new(bytes)
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]),
                }
            }
            // BSWAP r — (0F C8+rd) for 32-bit, (REX.W 0F C8+rd) for 64-bit
            X86Opcode::Bswap => {
                let dst_reg = match inst.operands.first() {
                    Some(MachineOperand::Register(r)) => *r,
                    _ => return EncodedInstruction::new(vec![0x0F, 0x0B]),
                };
                let mut bytes = Vec::with_capacity(4);
                let is_64 = inst.operand_size != 4;
                if is_64 {
                    // 64-bit: REX.W prefix required.
                    if let Some(rex) = compute_rex(true, None, None, Some(dst_reg)) {
                        bytes.push(rex);
                    } else {
                        bytes.push(0x48);
                    }
                } else {
                    // 32-bit: REX prefix only if register is R8-R15.
                    if hw_encoding(dst_reg) >= 8 {
                        bytes.push(rex_byte(false, false, false, true));
                    }
                }
                bytes.push(0x0F);
                bytes.push(0xC8 + (hw_encoding(dst_reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            // LoadInd dst, ptr — MOV dst, [ptr]
            X86Opcode::LoadInd => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(ptr))) => {
                        // MOV r64, [r64]:  encode as reg_mem_op with base=ptr
                        self.encode_reg_mem_op(&[0x8B], *dst, Some(*ptr), None, 1, 0, 8)
                    }
                    // Spill case: pointer was spilled to stack memory.
                    // Emit two-instruction sequence:
                    //   MOV dst, [base+disp]   — load the pointer value from stack
                    //   MOV dst, [dst]          — dereference the pointer
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let mut bytes = Vec::new();
                        let load_ptr = self.encode_reg_mem_op(
                            &[0x8B],
                            *dst,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&load_ptr.bytes);
                        let deref =
                            self.encode_reg_mem_op(&[0x8B], *dst, Some(*dst), None, 1, 0, 8);
                        bytes.extend_from_slice(&deref.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    // GlobalSymbol: LoadInd dst, GlobalSymbol →
                    // MOV dst, [rip+sym] (RIP-relative 64-bit load).
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::GlobalSymbol(sym)),
                    ) => {
                        let dst = *dst;
                        let sym_clone = sym.clone();
                        let opsz: u8 = if inst.operand_size > 0 {
                            inst.operand_size
                        } else {
                            8
                        };
                        let mut bytes = Vec::with_capacity(12);
                        if opsz == 2 {
                            bytes.push(OPERAND_SIZE_PREFIX);
                        }
                        let need_w = opsz == 8;
                        if opsz == 1 {
                            // MOVZX r32, byte [rip+disp]
                            if let Some(rex) = compute_rex(false, Some(dst), None, None) {
                                bytes.push(rex);
                            }
                            bytes.push(0x0F);
                            bytes.push(0xB6);
                        } else {
                            if let Some(rex) = compute_rex(need_w, Some(dst), None, None) {
                                bytes.push(rex);
                            } else if need_w {
                                bytes.push(rex_byte(true, false, false, false));
                            }
                            bytes.push(0x8B);
                        }
                        let reg_enc = hw_encoding(dst) & 0x7;
                        bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                        let reloc_offset = self.current_offset + bytes.len();
                        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                        let reloc_type = if self.pic_enabled {
                            X86_64RelocationType::RexGotPcRelX
                        } else {
                            X86_64RelocationType::Pc32
                        };
                        let reloc = RelocationEntry::new(
                            reloc_offset as u64,
                            sym_clone,
                            reloc_type,
                            -4,
                            ".text".to_string(),
                        );
                        EncodedInstruction::with_relocations(bytes, vec![reloc])
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // StoreInd ptr, src — MOV [ptr], src (64-bit)
            X86Opcode::StoreInd => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Register(src))) => {
                        // MOV [r64], r64:  encode as mem_reg_op with base=ptr
                        self.encode_mem_reg_op(&[0x89], Some(*ptr), None, 1, 0, *src, 8)
                    }
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Immediate(imm))) => {
                        // MOV [r64], imm32
                        let mut bytes = Vec::with_capacity(16);
                        let mem_enc = encode_memory_operand(0, Some(*ptr), None, 1, 0);
                        if let Some(rex) = compute_rex(true, None, None, Some(*ptr)) {
                            bytes.push(rex);
                        } else {
                            bytes.push(rex_byte(true, false, false, false));
                        }
                        bytes.push(0xC7);
                        bytes.push(mem_enc.modrm);
                        if let Some(s) = mem_enc.sib {
                            bytes.push(s);
                        }
                        bytes.extend_from_slice(&mem_enc.displacement);
                        bytes.extend_from_slice(&encode_immediate(*imm, 4));
                        EncodedInstruction::new(bytes)
                    }
                    // Spill case 1: pointer address is in stack memory, value in register.
                    // Use PUSH/MOV/POP sequence: push value, load ptr, pop into [ptr].
                    (
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                        Some(MachineOperand::Register(src)),
                    ) => {
                        let mut bytes = Vec::new();
                        // PUSH src — save the value to store onto stack
                        let push_inst = self.encode_push_pop(0x50, *src);
                        bytes.extend_from_slice(&push_inst.bytes);
                        // MOV src, [base+disp] — load the pointer address into src
                        let load_ptr = self.encode_reg_mem_op(
                            &[0x8B],
                            *src,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&load_ptr.bytes);
                        // POP [src] — pop saved value directly into memory at [pointer]
                        // POP m64: opcode 8F /0 with ModR/M, 64-bit default in long mode
                        let pop_indirect =
                            self.encode_reg_mem_op(&[0x8F], 0, Some(*src), None, 1, 0, 8);
                        bytes.extend_from_slice(&pop_indirect.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    // Spill case 2: pointer in register, value spilled to stack.
                    // Use PUSH mem / POP [ptr] to avoid clobbering any register.
                    (
                        Some(MachineOperand::Register(ptr)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let mut bytes = Vec::new();
                        // PUSH [base+disp] — push spilled value onto stack (FF /6)
                        let push_mem = self.encode_reg_mem_op(
                            &[0xFF],
                            6, // /6 = PUSH
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&push_mem.bytes);
                        // POP [ptr] — pop value into [ptr] (8F /0)
                        let pop_ind = self.encode_reg_mem_op(&[0x8F], 0, Some(*ptr), None, 1, 0, 8);
                        bytes.extend_from_slice(&pop_ind.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    // GlobalSymbol: StoreInd GlobalSymbol, Register →
                    // MOV [rip+sym], src (RIP-relative 64-bit store).
                    (
                        Some(MachineOperand::GlobalSymbol(sym)),
                        Some(MachineOperand::Register(src)),
                    ) => {
                        let src = *src;
                        let sym_clone = sym.clone();
                        let opsz: u8 = if inst.operand_size > 0 {
                            inst.operand_size
                        } else {
                            8
                        };
                        let mut bytes = Vec::with_capacity(12);
                        if opsz == 2 {
                            bytes.push(OPERAND_SIZE_PREFIX);
                        }
                        let need_w = opsz == 8;
                        if opsz == 1 {
                            // MOV byte [rip+disp], r8
                            // Must use byte-aware REX to select %sil/%dil/etc.
                            // instead of %dh/%bh when src is RSP/RBP/RSI/RDI.
                            if let Some(rex) =
                                compute_rex_byte_aware(false, Some(src), None, None, &[src])
                            {
                                bytes.push(rex);
                            }
                            bytes.push(0x88);
                        } else {
                            if let Some(rex) = compute_rex(need_w, Some(src), None, None) {
                                bytes.push(rex);
                            } else if need_w {
                                bytes.push(rex_byte(true, false, false, false));
                            }
                            bytes.push(0x89);
                        }
                        let reg_enc = hw_encoding(src) & 0x7;
                        bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                        let reloc_offset = self.current_offset + bytes.len();
                        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                        let reloc_type = if self.pic_enabled {
                            X86_64RelocationType::RexGotPcRelX
                        } else {
                            X86_64RelocationType::Pc32
                        };
                        let reloc = RelocationEntry::new(
                            reloc_offset as u64,
                            sym_clone,
                            reloc_type,
                            -4,
                            ".text".to_string(),
                        );
                        EncodedInstruction::with_relocations(bytes, vec![reloc])
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }

            // -- Sized indirect loads: LoadInd32/16/8 --
            // LoadInd32: MOV r32, [ptr] (zero-extends to r64 on x86-64)
            X86Opcode::LoadInd32 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(ptr))) => {
                        self.encode_reg_mem_op(&[0x8B], *dst, Some(*ptr), None, 1, 0, 4)
                    }
                    // Spill case: pointer in memory. Load pointer first, then deref.
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let mut bytes = Vec::new();
                        let load_ptr = self.encode_reg_mem_op(
                            &[0x8B],
                            *dst,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&load_ptr.bytes);
                        let deref =
                            self.encode_reg_mem_op(&[0x8B], *dst, Some(*dst), None, 1, 0, 4);
                        bytes.extend_from_slice(&deref.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    // GlobalSymbol: LoadInd32 dst, GlobalSymbol
                    // MOV r32, [rip+sym] (32-bit load, zero extends)
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::GlobalSymbol(sym)),
                    ) => {
                        let dst = *dst;
                        let sym_clone = sym.clone();
                        let mut bytes = Vec::with_capacity(12);
                        // 32-bit load — no REX.W
                        if let Some(rex) = compute_rex(false, Some(dst), None, None) {
                            bytes.push(rex);
                        }
                        bytes.push(0x8B);
                        let reg_enc = hw_encoding(dst) & 0x7;
                        bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                        let reloc_offset = self.current_offset + bytes.len();
                        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                        let reloc_type = if self.pic_enabled {
                            X86_64RelocationType::RexGotPcRelX
                        } else {
                            X86_64RelocationType::Pc32
                        };
                        let reloc = RelocationEntry::new(
                            reloc_offset as u64,
                            sym_clone,
                            reloc_type,
                            -4,
                            ".text".to_string(),
                        );
                        EncodedInstruction::with_relocations(bytes, vec![reloc])
                    }
                    // Immediate address: materialise into scratch, then load.
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::Immediate(addr)),
                    ) => {
                        let scratch: u16 = if *dst == 11 { 10 } else { 11 };
                        let mut bytes = Vec::new();
                        let mov_addr = self.encode_mov_imm64(scratch, *addr);
                        bytes.extend_from_slice(&mov_addr.bytes);
                        let load =
                            self.encode_reg_mem_op(&[0x8B], *dst, Some(scratch), None, 1, 0, 4);
                        bytes.extend_from_slice(&load.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // LoadInd16: MOVZX r32, word [ptr]
            X86Opcode::LoadInd16 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(ptr))) => {
                        // MOVZX r32, m16: 0F B7 /r
                        self.encode_reg_mem_op(&[0x0F, 0xB7], *dst, Some(*ptr), None, 1, 0, 4)
                    }
                    // Spill case: pointer in memory.
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let mut bytes = Vec::new();
                        let load_ptr = self.encode_reg_mem_op(
                            &[0x8B],
                            *dst,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&load_ptr.bytes);
                        let deref =
                            self.encode_reg_mem_op(&[0x0F, 0xB7], *dst, Some(*dst), None, 1, 0, 4);
                        bytes.extend_from_slice(&deref.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // LoadInd8: MOVZX r32, byte [ptr]
            X86Opcode::LoadInd8 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(ptr))) => {
                        // MOVZX r32, m8: 0F B6 /r
                        self.encode_reg_mem_op(&[0x0F, 0xB6], *dst, Some(*ptr), None, 1, 0, 4)
                    }
                    // Spill case: pointer in memory.
                    (
                        Some(MachineOperand::Register(dst)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let mut bytes = Vec::new();
                        let load_ptr = self.encode_reg_mem_op(
                            &[0x8B],
                            *dst,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            8,
                        );
                        bytes.extend_from_slice(&load_ptr.bytes);
                        let deref =
                            self.encode_reg_mem_op(&[0x0F, 0xB6], *dst, Some(*dst), None, 1, 0, 4);
                        bytes.extend_from_slice(&deref.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }

            // -- Sized indirect stores: StoreInd32/16/8 --
            // StoreInd32: MOV dword [ptr], r32
            X86Opcode::StoreInd32 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Register(src))) => {
                        self.encode_mem_reg_op(&[0x89], Some(*ptr), None, 1, 0, *src, 4)
                    }
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Immediate(imm))) => {
                        // MOV dword [ptr], imm32
                        let mut bytes = Vec::with_capacity(12);
                        let mem_enc = encode_memory_operand(0, Some(*ptr), None, 1, 0);
                        // For 32-bit store, no REX.W needed.
                        // But we still need REX if ptr uses r8-r15.
                        if let Some(rex) = compute_rex(false, None, None, Some(*ptr)) {
                            bytes.push(rex);
                        }
                        bytes.push(0xC7);
                        bytes.push(mem_enc.modrm);
                        if let Some(s) = mem_enc.sib {
                            bytes.push(s);
                        }
                        bytes.extend_from_slice(&mem_enc.displacement);
                        bytes.extend_from_slice(&encode_immediate(*imm, 4));
                        EncodedInstruction::new(bytes)
                    }
                    // Spill case: ptr in register, value spilled to stack.
                    // Use a scratch register that does NOT collide with ptr.
                    (
                        Some(MachineOperand::Register(ptr)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        // Choose scratch != ptr. Default to R11; if ptr IS R11, use R10.
                        let scratch: u16 = if *ptr == 11 { 10 } else { 11 };
                        let mut bytes = Vec::new();
                        // PUSH scratch — save scratch register
                        let save = self.encode_push_pop(0x50, scratch);
                        bytes.extend_from_slice(&save.bytes);
                        // MOV scratch_32, [base+disp] — load 32-bit spilled value
                        let load = self.encode_reg_mem_op(
                            &[0x8B],
                            scratch,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            4,
                        );
                        bytes.extend_from_slice(&load.bytes);
                        // MOV dword [ptr], scratch_32
                        let store =
                            self.encode_mem_reg_op(&[0x89], Some(*ptr), None, 1, 0, scratch, 4);
                        bytes.extend_from_slice(&store.bytes);
                        // POP scratch — restore scratch register
                        let restore = self.encode_push_pop(0x58, scratch);
                        bytes.extend_from_slice(&restore.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    // Immediate address (e.g. NULL or unresolved value):
                    // materialise the address into R11 scratch then store.
                    (
                        Some(MachineOperand::Immediate(addr)),
                        Some(MachineOperand::Register(src)),
                    ) => {
                        let scratch: u16 = if *src == 11 { 10 } else { 11 };
                        let mut bytes = Vec::new();
                        let mov_addr = self.encode_mov_imm64(scratch, *addr);
                        bytes.extend_from_slice(&mov_addr.bytes);
                        let store =
                            self.encode_mem_reg_op(&[0x89], Some(scratch), None, 1, 0, *src, 4);
                        bytes.extend_from_slice(&store.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    (
                        Some(MachineOperand::Immediate(addr)),
                        Some(MachineOperand::Immediate(val)),
                    ) => {
                        let scratch: u16 = 11;
                        let mut bytes = Vec::new();
                        let mov_addr = self.encode_mov_imm64(scratch, *addr);
                        bytes.extend_from_slice(&mov_addr.bytes);
                        let mem_enc = encode_memory_operand(0, Some(scratch), None, 1, 0);
                        if let Some(rex) = compute_rex(false, None, None, Some(scratch)) {
                            bytes.push(rex);
                        }
                        bytes.push(0xC7);
                        bytes.push(mem_enc.modrm);
                        if let Some(s) = mem_enc.sib {
                            bytes.push(s);
                        }
                        bytes.extend_from_slice(&mem_enc.displacement);
                        bytes.extend_from_slice(&encode_immediate(*val, 4));
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // StoreInd16: MOV word [ptr], r16
            X86Opcode::StoreInd16 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Register(src))) => {
                        // 66 prefix + MOV [m], r16
                        self.encode_mem_reg_op(&[0x89], Some(*ptr), None, 1, 0, *src, 2)
                    }
                    // Spill case: ptr in register, value spilled to stack.
                    (
                        Some(MachineOperand::Register(ptr)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let scratch: u16 = if *ptr == 11 { 10 } else { 11 };
                        let mut bytes = Vec::new();
                        let save = self.encode_push_pop(0x50, scratch);
                        bytes.extend_from_slice(&save.bytes);
                        let load = self.encode_reg_mem_op(
                            &[0x8B],
                            scratch,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            4,
                        );
                        bytes.extend_from_slice(&load.bytes);
                        let store =
                            self.encode_mem_reg_op(&[0x89], Some(*ptr), None, 1, 0, scratch, 2);
                        bytes.extend_from_slice(&store.bytes);
                        let restore = self.encode_push_pop(0x58, scratch);
                        bytes.extend_from_slice(&restore.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }
            // StoreInd8: MOV byte [ptr], r8
            X86Opcode::StoreInd8 => {
                let ops = &inst.operands;
                match (ops.first(), ops.get(1)) {
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Register(src))) => {
                        // MOV [m], r8: opcode 0x88
                        self.encode_mem_reg_op(&[0x88], Some(*ptr), None, 1, 0, *src, 1)
                    }
                    (Some(MachineOperand::Register(ptr)), Some(MachineOperand::Immediate(imm))) => {
                        // MOV byte [ptr], imm8: opcode 0xC6 /0
                        let mut bytes = Vec::with_capacity(8);
                        let mem_enc = encode_memory_operand(0, Some(*ptr), None, 1, 0);
                        if let Some(rex) = compute_rex(false, None, None, Some(*ptr)) {
                            bytes.push(rex);
                        }
                        bytes.push(0xC6);
                        bytes.push(mem_enc.modrm);
                        if let Some(s) = mem_enc.sib {
                            bytes.push(s);
                        }
                        bytes.extend_from_slice(&mem_enc.displacement);
                        bytes.push(*imm as u8);
                        EncodedInstruction::new(bytes)
                    }
                    // Spill case: ptr in register, value spilled to stack.
                    (
                        Some(MachineOperand::Register(ptr)),
                        Some(MachineOperand::Memory {
                            base,
                            index,
                            scale,
                            displacement,
                        }),
                    ) => {
                        let scratch: u16 = if *ptr == 11 { 10 } else { 11 };
                        let mut bytes = Vec::new();
                        let save = self.encode_push_pop(0x50, scratch);
                        bytes.extend_from_slice(&save.bytes);
                        let load = self.encode_reg_mem_op(
                            &[0x8B],
                            scratch,
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            4,
                        );
                        bytes.extend_from_slice(&load.bytes);
                        let store =
                            self.encode_mem_reg_op(&[0x88], Some(*ptr), None, 1, 0, scratch, 1);
                        bytes.extend_from_slice(&store.bytes);
                        let restore = self.encode_push_pop(0x58, scratch);
                        bytes.extend_from_slice(&restore.bytes);
                        EncodedInstruction::new(bytes)
                    }
                    _ => {
                        if crate::common::verbosity::verbose_enabled() {
                            eprintln!(
                                "[UD2] opcode={} ops={:?} res={:?}",
                                inst.opcode, inst.operands, inst.result
                            );
                        }
                        EncodedInstruction::new(vec![0x0F, 0x0B])
                    }
                }
            }

            // ---- x87 FPU instructions for long double ABI ----

            // FLD QWORD [mem] — load 64-bit double from memory into ST(0)
            // Opcode: 0xDD /0  (ModRM reg field = 0)
            X86Opcode::FldMem64 => {
                match inst.operands.first() {
                    Some(MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    }) => {
                        self.encode_reg_mem_op(
                            &[0xDD],
                            0, // /0 = FLD
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            0, // no REX.W needed for x87
                        )
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]), // UD2 fallback
                }
            }
            // FLD TBYTE [mem] — load 80-bit extended precision into ST(0)
            // Opcode: 0xDB /5  (ModRM reg field = 5)
            X86Opcode::FldMem80 => {
                match inst.operands.first() {
                    Some(MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    }) => {
                        self.encode_reg_mem_op(
                            &[0xDB],
                            5, // /5 = FLD m80
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            0, // no REX.W needed for x87
                        )
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]), // UD2 fallback
                }
            }
            // FSTP TWORD [mem] — store 80-bit extended from ST(0), pop stack
            // Opcode: 0xDB /7  (ModRM reg field = 7)
            X86Opcode::FstpMem80 => {
                match inst.operands.first() {
                    Some(MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    }) => {
                        self.encode_reg_mem_op(
                            &[0xDB],
                            7, // /7 = FSTP m80
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            0, // no REX.W needed for x87
                        )
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]),
                }
            }
            // FSTP QWORD [mem] — store 64-bit double from ST(0), pop stack
            // Opcode: 0xDD /3  (ModRM reg field = 3)
            X86Opcode::FstpMem64 => {
                match inst.operands.first() {
                    Some(MachineOperand::Memory {
                        base,
                        index,
                        scale,
                        displacement,
                    }) => {
                        self.encode_reg_mem_op(
                            &[0xDD],
                            3, // /3 = FSTP m64
                            *base,
                            *index,
                            *scale,
                            *displacement,
                            0,
                        )
                    }
                    _ => EncodedInstruction::new(vec![0x0F, 0x0B]),
                }
            }

            // Catch-all
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[ENCODER-DEBUG] ud2 from catch-all: opcode={}, operands={:?}, result={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                {
                    if crate::common::verbosity::verbose_enabled() {
                        eprintln!(
                            "[UD2] opcode={} ops={:?} res={:?}",
                            inst.opcode, inst.operands, inst.result
                        );
                    }
                    EncodedInstruction::new(vec![0x0F, 0x0B])
                }
            }
        };

        self.current_offset += result.bytes.len();
        result
    }

    // ===================================================================
    // Public Encoding Helpers
    // ===================================================================

    /// Encode a two-register instruction (e.g., `add rax, rcx`).
    ///
    /// The source register goes in the ModR/M `reg` field and the
    /// destination register goes in the `r/m` field (mod=11).
    pub fn encode_reg_reg_op(
        &mut self,
        opcode: &[u8],
        dst: u16,
        src: u16,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let need_64 = operand_size == 8;
        // For byte-sized operands, ensure REX prefix for registers RSP/RBP/RSI/RDI
        // whose hw encodings 4-7 would otherwise select %ah/%ch/%dh/%bh.
        if operand_size == 1 {
            if let Some(rex) =
                compute_rex_byte_aware(need_64, Some(src), None, Some(dst), &[src, dst])
            {
                bytes.push(rex);
            }
        } else if let Some(rex) = compute_rex(need_64, Some(src), None, Some(dst)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(
            0b11,
            hw_encoding(src) & 0x7,
            hw_encoding(dst) & 0x7,
        ));
        EncodedInstruction::new(bytes)
    }

    /// Encode a register-immediate instruction (e.g., `add rax, 42`).
    pub fn encode_reg_imm_op(
        &mut self,
        opcode: &[u8],
        opext: u8,
        dst: u16,
        imm: i64,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let need_64 = operand_size == 8;
        // For byte-sized operands, ensure REX prefix for registers with hw encoding 4-7
        if operand_size == 1 {
            if let Some(rex) = compute_rex_byte_aware(need_64, None, None, Some(dst), &[dst]) {
                bytes.push(rex);
            }
        } else if let Some(rex) = compute_rex(need_64, None, None, Some(dst)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(0b11, opext, hw_encoding(dst) & 0x7));
        let imm_size = if opcode.last() == Some(&0x83) {
            1
        } else if operand_size <= 4 {
            operand_size.max(1)
        } else {
            4
        };
        bytes.extend_from_slice(&encode_immediate(imm, imm_size));
        EncodedInstruction::new(bytes)
    }

    /// Encode a register-memory instruction (e.g., `mov rax, [rbp-8]`).
    pub fn encode_reg_mem_op(
        &mut self,
        opcode: &[u8],
        reg: u16,
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        displacement: i64,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let reg_enc = hw_encoding(reg);
        let mem_enc = encode_memory_operand(reg_enc & 0x7, base, index, scale, displacement);
        let need_64 = operand_size == 8;
        // For byte-sized operands, ensure REX prefix for registers with hw encoding 4-7
        if operand_size == 1 {
            if let Some(rex) = compute_rex_byte_aware(need_64, Some(reg), index, base, &[reg]) {
                bytes.push(rex);
            }
        } else if let Some(rex) = compute_rex(need_64, Some(reg), index, base) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(mem_enc.modrm);
        if let Some(s) = mem_enc.sib {
            bytes.push(s);
        }
        bytes.extend_from_slice(&mem_enc.displacement);
        EncodedInstruction::new(bytes)
    }

    /// Encode a memory-register instruction (e.g., `mov [rbp-8], rax`).
    pub fn encode_mem_reg_op(
        &mut self,
        opcode: &[u8],
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        displacement: i64,
        reg: u16,
        operand_size: u8,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if operand_size == 2 {
            bytes.push(OPERAND_SIZE_PREFIX);
        }
        let reg_enc = hw_encoding(reg);
        let mem_enc = encode_memory_operand(reg_enc & 0x7, base, index, scale, displacement);
        let need_64 = operand_size == 8;
        // For byte-sized operands, ensure REX prefix for registers with hw encoding 4-7
        // (RSP/RBP/RSI/RDI) so the CPU selects %spl/%bpl/%sil/%dil instead of %ah/%ch/%dh/%bh.
        if operand_size == 1 {
            if let Some(rex) = compute_rex_byte_aware(need_64, Some(reg), index, base, &[reg]) {
                bytes.push(rex);
            }
        } else if let Some(rex) = compute_rex(need_64, Some(reg), index, base) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(mem_enc.modrm);
        if let Some(s) = mem_enc.sib {
            bytes.push(s);
        }
        bytes.extend_from_slice(&mem_enc.displacement);
        EncodedInstruction::new(bytes)
    }

    /// Encode a relative branch/call instruction (e.g., `jmp label`).
    pub fn encode_relative_op(&mut self, opcode: &[u8], target_offset: i64) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(opcode);
        let inst_size = (opcode.len() + 4) as i64;
        let rel32 = target_offset - (self.current_offset as i64 + inst_size);
        bytes.extend_from_slice(&(rel32 as i32).to_le_bytes());
        EncodedInstruction::new(bytes)
    }

    /// Encode a RIP-relative memory access (for PIC code).
    pub fn encode_rip_relative(
        &mut self,
        opcode: &[u8],
        reg: u16,
        symbol: &str,
        addend: i64,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(12);
        if let Some(rex) = compute_rex(true, Some(reg), None, None) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        let reg_enc = hw_encoding(reg) & 0x7;
        bytes.push(modrm_byte(0b00, reg_enc, 0b101));
        let reloc_offset = self.current_offset + bytes.len();
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        // In PIC mode, global data references use GOT-relative relocations
        // (R_X86_64_REX_GOTPCRELX) so the linker creates GOT entries for
        // external symbols.  In non-PIC mode, use direct PC-relative
        // (R_X86_64_PC32) addressing.
        let reloc_type = if self.pic_enabled {
            X86_64RelocationType::RexGotPcRelX
        } else {
            X86_64RelocationType::Pc32
        };
        let reloc = RelocationEntry::new(
            reloc_offset as u64,
            symbol.to_string(),
            reloc_type,
            addend - 4,
            ".text".to_string(),
        );
        EncodedInstruction::with_relocations(bytes, vec![reloc])
    }

    /// Encode a PUSH or POP instruction using the opcode+rd form.
    pub fn encode_push_pop(&mut self, base_opcode: u8, reg: u16) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(2);
        if needs_rex(reg) {
            bytes.push(rex_byte(false, false, false, true));
        }
        bytes.push(base_opcode + (hw_encoding(reg) & 0x7));
        EncodedInstruction::new(bytes)
    }

    /// Encode an SSE register-register instruction with mandatory prefix.
    pub fn encode_sse_reg_reg(
        &mut self,
        prefix: u8,
        opcode: &[u8],
        dst: u16,
        src: u16,
    ) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(8);
        bytes.push(prefix);
        if let Some(rex) = compute_rex(false, Some(dst), None, Some(src)) {
            bytes.push(rex);
        }
        bytes.extend_from_slice(opcode);
        bytes.push(modrm_byte(
            0b11,
            hw_encoding(dst) & 0x7,
            hw_encoding(src) & 0x7,
        ));
        EncodedInstruction::new(bytes)
    }

    // ===================================================================
    // Private Encoding Helpers
    // ===================================================================

    /// Extract a register from the operand list at the given index.
    fn extract_register(&self, operands: &[MachineOperand], idx: usize) -> u16 {
        match operands.get(idx) {
            Some(MachineOperand::Register(r)) => *r,
            Some(MachineOperand::FrameSlot(_)) => RBP,
            _ => RAX,
        }
    }

    /// Encode an ALU binary operation (ADD, SUB, AND, OR, XOR, CMP).
    /// Infer the operand size (in bytes) from a MachineInstruction.
    ///
    /// Uses `operand_size` if explicitly set (non-zero), otherwise defaults
    /// to 8 (64-bit) for x86-64.  This allows the instruction selector to
    /// emit 32-bit, 16-bit, or 8-bit operations by setting the field on
    /// the MachineInstruction.
    #[inline]
    fn infer_operand_size(inst: &MachineInstruction) -> u8 {
        if inst.operand_size > 0 {
            inst.operand_size
        } else {
            8 // x86-64 default: 64-bit operations
        }
    }

    fn encode_alu_op(
        &mut self,
        inst: &MachineInstruction,
        opext: u8,
        rm_reg_opc: u8,
        reg_rm_opc: u8,
    ) -> EncodedInstruction {
        // Handle 3-operand normalised form: [dst, dst_dup, src] → [dst, src]
        // The codegen emits mk_inst(Op, Some(dst), &[dst, src]) and the
        // normaliser prepends `result` to get [dst, dst, src].  The x86 ALU
        // encoding needs [dst, src] where dst is both source and destination.
        let ops: &[MachineOperand] = if inst.operands.len() >= 3 {
            // Skip the duplicate dst at index 1.
            &[inst.operands[0].clone(), inst.operands[2].clone()]
        } else {
            &inst.operands
        };
        // Re-borrow as a slice for the match below.
        let ops_ref: Vec<MachineOperand> = ops.to_vec();
        let ops = &ops_ref;
        let opsz = Self::infer_operand_size(inst);
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[rm_reg_opc], *dst, *src, opsz)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                if (-128..=127).contains(&imm) {
                    self.encode_reg_imm_op(&[0x83], opext, *dst, imm, opsz)
                } else {
                    self.encode_reg_imm_op(&[0x81], opext, *dst, imm, opsz)
                }
            }
            (
                Some(MachineOperand::Register(reg)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => self.encode_reg_mem_op(
                &[reg_rm_opc],
                *reg,
                *base,
                *index,
                *scale,
                *displacement,
                opsz,
            ),
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => self
                .encode_reg_mem_op(
                    &[reg_rm_opc],
                    *reg,
                    Some(RBP),
                    None,
                    1,
                    *offset as i64,
                    opsz,
                ),
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Register(reg)),
            ) => self.encode_mem_reg_op(
                &[rm_reg_opc],
                *base,
                *index,
                *scale,
                *displacement,
                *reg,
                opsz,
            ),
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Immediate(imm)),
            ) => {
                let imm = *imm;
                let (base, index, scale, displacement) = (*base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(16);
                let opc = if (-128..=127).contains(&imm) {
                    0x83u8
                } else {
                    0x81u8
                };
                let mem_enc = encode_memory_operand(opext, base, index, scale, displacement);
                if let Some(rex) = compute_rex(true, None, index, base) {
                    bytes.push(rex);
                }
                bytes.push(opc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                let imm_size = if opc == 0x83 { 1 } else { 4 };
                bytes.extend_from_slice(&encode_immediate(imm, imm_size));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode TEST instruction.
    fn encode_test(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[0x85], *dst, *src, 8)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                self.encode_reg_imm_op(&[0xF7], 0, *dst, *imm, 8)
            }
            // TEST reg, [mem] — register first, memory second.
            // TEST is commutative, so we encode as TEST [mem], reg (0x85 /r).
            (
                Some(MachineOperand::Register(reg)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => self.encode_mem_reg_op(&[0x85], *base, *index, *scale, *displacement, *reg, 8),
            // TEST reg, [frame slot] — register first, frame slot second.
            // Encode as TEST [RBP+offset], reg (commutative swap).
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                self.encode_mem_reg_op(&[0x85], Some(RBP), None, 1, *offset as i64, *reg, 8)
            }
            // TEST [mem], reg — memory operand first, register second.
            // Encoding: REX.W 85 ModR/M SIB (optional) disp (optional)
            // Used by the stack probe loop: test [rsp], rsp
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Register(src)),
            ) => self.encode_mem_reg_op(&[0x85], *base, *index, *scale, *displacement, *src, 8),
            // TEST [mem], imm — memory operand first, immediate second.
            // Uses F7 /0 encoding (reg field = 0 for TEST)
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Immediate(imm)),
            ) => {
                // Encode as TEST r/m64, imm32: REX.W F7 /0 modrm [sib] [disp] imm32
                let mem_enc = encode_memory_operand(0, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(16);
                if let Some(rex) = compute_rex(true, None, *index, *base) {
                    bytes.push(rex);
                } else {
                    bytes.push(rex_byte(true, false, false, false));
                }
                bytes.push(0xF7);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                // Append 32-bit immediate
                let imm_val = *imm as i32;
                bytes.extend_from_slice(&imm_val.to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            // TEST [frame slot], reg
            (Some(MachineOperand::FrameSlot(offset)), Some(MachineOperand::Register(reg))) => {
                self.encode_mem_reg_op(&[0x85], Some(RBP), None, 1, *offset as i64, *reg, 8)
            }
            // TEST [frame slot], imm
            (Some(MachineOperand::FrameSlot(offset)), Some(MachineOperand::Immediate(imm))) => {
                let mem_enc = encode_memory_operand(0, Some(RBP), None, 1, *offset as i64);
                let mut bytes = Vec::with_capacity(16);
                bytes.push(rex_byte(true, false, false, false));
                bytes.push(0xF7);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                let imm_val = *imm as i32;
                bytes.extend_from_slice(&imm_val.to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode MOV instruction — handles multiple forms.
    fn encode_mov(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let opsz = Self::infer_operand_size(inst);
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                self.encode_reg_reg_op(&[0x89], *dst, *src, opsz)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let imm = *imm;
                let dst = *dst;
                // For 32-bit MOV, use the 32-bit encoding path.
                if opsz <= 4 || (i32::MIN as i64..=i32::MAX as i64).contains(&imm) {
                    self.encode_reg_imm_op(&[0xC7], 0, dst, imm, opsz)
                } else {
                    // MOV r64, imm64: REX.W + 0xB8+rd + imm64
                    let mut bytes = Vec::with_capacity(10);
                    let rex = compute_rex(true, None, None, Some(dst))
                        .unwrap_or(rex_byte(true, false, false, false));
                    bytes.push(rex);
                    bytes.push(0xB8 + (hw_encoding(dst) & 0x7));
                    bytes.extend_from_slice(&imm.to_le_bytes());
                    EncodedInstruction::new(bytes)
                }
            }
            (
                Some(MachineOperand::Register(reg)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => {
                // Use byte opcode 0x8A for 1-byte loads, 0x8B for wider.
                let opc: &[u8] = if opsz == 1 { &[0x8A] } else { &[0x8B] };
                self.encode_reg_mem_op(opc, *reg, *base, *index, *scale, *displacement, opsz)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                let opc: &[u8] = if opsz == 1 { &[0x8A] } else { &[0x8B] };
                self.encode_reg_mem_op(opc, *reg, Some(RBP), None, 1, *offset as i64, opsz)
            }
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Register(reg)),
            ) => {
                // Use byte opcode 0x88 for 1-byte stores, 0x89 for wider.
                let opc: &[u8] = if opsz == 1 { &[0x88] } else { &[0x89] };
                self.encode_mem_reg_op(opc, *base, *index, *scale, *displacement, *reg, opsz)
            }
            (Some(MachineOperand::FrameSlot(offset)), Some(MachineOperand::Register(reg))) => {
                let opc: &[u8] = if opsz == 1 { &[0x88] } else { &[0x89] };
                self.encode_mem_reg_op(opc, Some(RBP), None, 1, *offset as i64, *reg, opsz)
            }
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Immediate(imm)),
            ) => {
                let (base, index, scale, displacement) = (*base, *index, *scale, *displacement);
                let need_64 = opsz == 8;
                let mut bytes = Vec::with_capacity(16);
                let mem_enc = encode_memory_operand(0, base, index, scale, displacement);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, None, index, base) {
                    bytes.push(rex);
                }
                bytes.push(0xC7);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                bytes.extend_from_slice(&encode_immediate(*imm, 4));
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::GlobalSymbol(sym))) => {
                // RIP-relative load: MOV reg, [rip + sym]
                // Use operand size to determine encoding width.
                // For opsz <= 4 (I32/I16/I8), do NOT set REX.W to avoid
                // reading 8 bytes from a 4-byte global.
                let reg = *reg;
                let sym_clone = sym.clone();
                let mut bytes = Vec::with_capacity(12);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                let need_w = opsz == 8;
                if opsz == 1 {
                    // 8-bit load: MOVZX r32, byte [rip+disp32]
                    // Use 0x0F 0xB6 (MOVZX r32, r/m8) for zero-extension.
                    if let Some(rex) = compute_rex(false, Some(reg), None, None) {
                        bytes.push(rex);
                    }
                    bytes.push(0x0F);
                    bytes.push(0xB6);
                } else {
                    if let Some(rex) = compute_rex(need_w, Some(reg), None, None) {
                        bytes.push(rex);
                    } else if need_w {
                        bytes.push(rex_byte(true, false, false, false));
                    }
                    bytes.push(0x8B);
                }
                let reg_enc = hw_encoding(reg) & 0x7;
                bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc_type = if self.pic_enabled {
                    X86_64RelocationType::RexGotPcRelX
                } else {
                    X86_64RelocationType::Pc32
                };
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym_clone,
                    reloc_type,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            (Some(MachineOperand::GlobalSymbol(sym)), Some(MachineOperand::Register(reg))) => {
                let sym_clone = sym.clone();
                let reg = *reg;
                let mut bytes = Vec::with_capacity(12);
                // Use operand size to determine the encoding width.
                // opsz 1 → byte store, 2 → word, 4 → dword, 8 → qword.
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                let need_w = opsz == 8;
                if opsz == 1 {
                    // 8-bit store: MOV [rip+disp32], r8  (opcode 0x88)
                    // Byte-aware REX for %sil/%dil/%spl/%bpl selection.
                    if let Some(rex) = compute_rex_byte_aware(false, Some(reg), None, None, &[reg])
                    {
                        bytes.push(rex);
                    }
                    bytes.push(0x88);
                } else {
                    // 16/32/64-bit store: MOV [rip+disp32], r16/r32/r64 (0x89)
                    // REX.W=1 only for 64-bit.
                    if let Some(rex) = compute_rex(need_w, Some(reg), None, None) {
                        bytes.push(rex);
                    } else if need_w {
                        bytes.push(rex_byte(true, false, false, false));
                    }
                    bytes.push(0x89);
                }
                let reg_enc = hw_encoding(reg) & 0x7;
                bytes.push(modrm_byte(0b00, reg_enc, 0b101));
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym_clone,
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode LEA instruction.
    fn encode_lea(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (
                Some(MachineOperand::Register(reg)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => self.encode_reg_mem_op(&[0x8D], *reg, *base, *index, *scale, *displacement, 8),
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::FrameSlot(offset))) => {
                self.encode_reg_mem_op(&[0x8D], *reg, Some(RBP), None, 1, *offset as i64, 8)
            }
            (Some(MachineOperand::Register(reg)), Some(MachineOperand::GlobalSymbol(sym))) => {
                if self.pic_enabled {
                    self.encode_rip_relative(&[0x8B], *reg, sym, 0)
                } else {
                    self.encode_rip_relative(&[0x8D], *reg, sym, 0)
                }
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode shift instruction (SHL, SHR, SAR).
    fn encode_shift(
        &mut self,
        inst: &MachineInstruction,
        opcode: &X86Opcode,
    ) -> EncodedInstruction {
        let opext = shift_opext(opcode);
        let opsz = Self::infer_operand_size(inst);
        let need_64 = opsz == 8;
        // Handle normalization: 3-operand form [dst, dst, count] → strip dup
        let ops = &inst.operands;
        let (dst_op, count_op) = if ops.len() == 3 {
            // Normalized: [dst, dst_dup, count] — skip the duplicate
            (&ops[0], &ops[2])
        } else {
            // Original 2-operand: [dst, count]
            (
                ops.first().unwrap_or(&MachineOperand::Immediate(0)),
                ops.get(1).unwrap_or(&MachineOperand::Immediate(0)),
            )
        };
        match (dst_op, count_op) {
            (MachineOperand::Register(dst), MachineOperand::Register(cl)) if *cl == RCX => {
                let mut bytes = Vec::with_capacity(4);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, None, None, Some(*dst)) {
                    bytes.push(rex);
                }
                bytes.push(0xD3);
                bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                EncodedInstruction::new(bytes)
            }
            (MachineOperand::Register(dst), MachineOperand::Immediate(imm)) => {
                let mut bytes = Vec::with_capacity(5);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, None, None, Some(*dst)) {
                    bytes.push(rex);
                }
                if *imm == 1 {
                    bytes.push(0xD1);
                    bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                } else {
                    bytes.push(0xC1);
                    bytes.push(modrm_byte(0b11, opext, hw_encoding(*dst) & 0x7));
                    bytes.push(*imm as u8);
                }
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode unary instruction (NEG, NOT, IDIV, DIV).
    fn encode_unary(
        &mut self,
        inst: &MachineInstruction,
        opcode: &X86Opcode,
    ) -> EncodedInstruction {
        let entry = self.opcode_table.get(&opcode.as_u32());
        let (opc_byte, opext) = match entry {
            Some(e) => (e.opcode[0], e.reg_opext.unwrap_or(0)),
            None => (0xF7, 0),
        };
        let opsz = Self::infer_operand_size(inst);
        let need_64 = opsz == 8;
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Register(reg)) => {
                let mut bytes = Vec::with_capacity(4);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, None, None, Some(*reg)) {
                    bytes.push(rex);
                }
                bytes.push(opc_byte);
                bytes.push(modrm_byte(0b11, opext, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Memory {
                base,
                index,
                scale,
                displacement,
            }) => {
                let mut bytes = Vec::with_capacity(12);
                let mem_enc = encode_memory_operand(opext, *base, *index, *scale, *displacement);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, None, *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(opc_byte);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode IMUL instruction.
    fn encode_imul(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let opsz = Self::infer_operand_size(inst);
        let need_64 = opsz == 8;
        // Handle 3-operand normalised form: [dst, dst_dup, src] → [dst, src]
        // (same pattern as encode_alu_op)
        let ops_vec: Vec<MachineOperand>;
        let ops: &[MachineOperand] = if inst.operands.len() >= 3 {
            if let (
                Some(MachineOperand::Register(_)),
                Some(MachineOperand::Register(_)),
                Some(MachineOperand::Register(_)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                ops_vec = vec![inst.operands[0].clone(), inst.operands[2].clone()];
                &ops_vec
            } else {
                &inst.operands
            }
        } else {
            &inst.operands
        };
        match (ops.first(), ops.get(1), ops.get(2)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)), None) => {
                let mut bytes = Vec::with_capacity(5);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0xAF);
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*dst) & 0x7,
                    hw_encoding(*src) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::Register(src)),
                Some(MachineOperand::Immediate(imm)),
            ) => {
                let imm = *imm;
                let mut bytes = Vec::with_capacity(8);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                if (-128..=127).contains(&imm) {
                    bytes.push(0x6B);
                    bytes.push(modrm_byte(
                        0b11,
                        hw_encoding(*dst) & 0x7,
                        hw_encoding(*src) & 0x7,
                    ));
                    bytes.push(imm as u8);
                } else {
                    bytes.push(0x69);
                    bytes.push(modrm_byte(
                        0b11,
                        hw_encoding(*dst) & 0x7,
                        hw_encoding(*src) & 0x7,
                    ));
                    bytes.extend_from_slice(&(imm as i32).to_le_bytes());
                }
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                None,
            ) => {
                let mut bytes = Vec::with_capacity(10);
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0xAF);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Two-operand IMUL with spilled source (FrameSlot → [RBP+disp]).
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::FrameSlot(offset)),
                None,
            ) => {
                let mut bytes = Vec::with_capacity(10);
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    Some(RBP),
                    None,
                    1,
                    *offset as i64,
                );
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(*dst), None, Some(RBP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0xAF);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Three-operand IMUL with spilled source: IMUL dst, [RBP+disp], imm
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::FrameSlot(offset)),
                Some(MachineOperand::Immediate(imm)),
            ) => {
                let imm = *imm;
                let mut bytes = Vec::with_capacity(12);
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    Some(RBP),
                    None,
                    1,
                    *offset as i64,
                );
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(*dst), None, Some(RBP)) {
                    bytes.push(rex);
                }
                if (-128..=127).contains(&imm) {
                    bytes.push(0x6B);
                } else {
                    bytes.push(0x69);
                }
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                if (-128..=127).contains(&imm) {
                    bytes.push(imm as u8);
                } else {
                    bytes.extend_from_slice(&(imm as i32).to_le_bytes());
                }
                EncodedInstruction::new(bytes)
            }
            // Two-operand IMUL with immediate: IMUL reg, imm → IMUL reg, reg, imm.
            // x86 has no direct 2-operand reg,imm form; we encode as the
            // 3-operand variant where dst and src are the same register.
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm)), None) => {
                let imm = *imm;
                let dst = *dst;
                // Optimisation: multiply by 0 → zero the register.
                if imm == 0 {
                    let mut xor_inst = inst.clone();
                    xor_inst.opcode = X86Opcode::Xor.as_u32();
                    xor_inst.operands =
                        vec![MachineOperand::Register(dst), MachineOperand::Register(dst)];
                    // encode_alu_op(inst, opext, rm_reg_opc, reg_rm_opc)
                    // XOR: opext=6, rm_reg=0x31, reg_rm=0x33
                    return self.encode_alu_op(&xor_inst, 6, 0x31, 0x33);
                }
                // General case: IMUL dst, dst, imm.
                let mut bytes = Vec::with_capacity(8);
                if opsz == 2 {
                    bytes.push(OPERAND_SIZE_PREFIX);
                }
                if let Some(rex) = compute_rex(need_64, Some(dst), None, Some(dst)) {
                    bytes.push(rex);
                }
                if (-128..=127).contains(&imm) {
                    bytes.push(0x6B);
                    bytes.push(modrm_byte(
                        0b11,
                        hw_encoding(dst) & 0x7,
                        hw_encoding(dst) & 0x7,
                    ));
                    bytes.push(imm as u8);
                } else {
                    bytes.push(0x69);
                    bytes.push(modrm_byte(
                        0b11,
                        hw_encoding(dst) & 0x7,
                        hw_encoding(dst) & 0x7,
                    ));
                    bytes.extend_from_slice(&(imm as i32).to_le_bytes());
                }
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode MOVZX.
    fn encode_movzx_movsx(
        &mut self,
        inst: &MachineInstruction,
        byte_opc: &[u8],
        word_opc: &[u8],
    ) -> EncodedInstruction {
        // Choose byte vs word source opcode based on operand_size:
        // operand_size == 2 → word source (MOVZWL / MOVSWL)
        // otherwise → byte source (MOVZBL / MOVSBL)
        let use_word = inst.operand_size == 2;
        let opc = if use_word { word_opc } else { byte_opc };
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            // MOVZX reg, immediate → emit MOV reg, zero-extended-immediate.
            // x86 has no MOVZX from immediate; the extension is a no-op on
            // an already-known constant — just materialize it.
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let val = if use_word { *imm & 0xFFFF } else { *imm & 0xFF };
                // Reuse the Mov encoder with the zero-extended value.
                let mut mov_inst = inst.clone();
                mov_inst.opcode = X86Opcode::Mov.as_u32();
                mov_inst.operands = vec![
                    MachineOperand::Register(*dst),
                    MachineOperand::Immediate(val),
                ];
                mov_inst.operand_size = 4; // 32-bit MOV zero-extends to 64 bits on x86-64
                self.encode_mov(&mov_inst)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                // MOVZBL/MOVSBL read a byte register in the r/m field.
                // Without REX, hw encodings 4-7 address AH/CH/DH/BH.
                // With REX, they address SPL/BPL/SIL/DIL.  We must emit
                // a plain REX (0x40) when the source byte register is in
                // this range so the correct low-byte register is read,
                // matching what SETcc already does for byte destinations.
                let rex_from_ext = compute_rex(false, Some(*dst), None, Some(*src));
                let src_hw = hw_encoding(*src);
                if let Some(rex) = rex_from_ext {
                    bytes.push(rex);
                } else if !use_word && (4..=7).contains(&src_hw) {
                    bytes.push(0x40); // plain REX to access SPL/BPL/SIL/DIL
                }
                bytes.extend_from_slice(opc);
                bytes.push(modrm_byte(0b11, hw_encoding(*dst) & 0x7, src_hw & 0x7));
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(false, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.extend_from_slice(opc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode MOVSX / MOVSXD.
    fn encode_movsx(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        // Dispatch based on operand_size which encodes the SOURCE width:
        //   1 → MOVSX r64, r/m8  (0x0F 0xBE with REX.W)
        //   2 → MOVSX r64, r/m16 (0x0F 0xBF with REX.W)
        //   4 → MOVSXD r64, r/m32 (0x63 with REX.W)
        let src_bytes = inst.operand_size;
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            // MOVSX reg, immediate → sign-extend at compile time, emit MOV.
            // x86 has no MOVSX from an immediate; the extension is trivial
            // on a known constant — just materialize the sign-extended value.
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Immediate(imm))) => {
                let val = match src_bytes {
                    1 => *imm as i8 as i64,
                    2 => *imm as i16 as i64,
                    4 => *imm as i32 as i64,
                    _ => *imm,
                };
                let mut mov_inst = inst.clone();
                mov_inst.opcode = X86Opcode::Mov.as_u32();
                mov_inst.operands = vec![
                    MachineOperand::Register(*dst),
                    MachineOperand::Immediate(val),
                ];
                // Use 8-byte operand size for potentially negative values;
                // 4-byte is fine when val fits in i32 (encode_mov handles this).
                mov_inst.operand_size = 8;
                self.encode_mov(&mov_inst)
            }
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                match src_bytes {
                    1 => {
                        // MOVSX r64, r/m8: REX.W 0x0F 0xBE /r
                        // Need REX.W for 64-bit destination, plus REX.B if
                        // src uses r8-r15, REX.R if dst uses r8-r15.
                        let mut rex = 0x48u8; // REX.W
                        if hw_encoding(*dst) >= 8 {
                            rex |= 0x04; // REX.R
                        }
                        if hw_encoding(*src) >= 8 {
                            rex |= 0x01; // REX.B
                        }
                        bytes.push(rex);
                        bytes.push(0x0F);
                        bytes.push(0xBE);
                        bytes.push(modrm_byte(
                            0b11,
                            hw_encoding(*dst) & 0x7,
                            hw_encoding(*src) & 0x7,
                        ));
                    }
                    2 => {
                        // MOVSX r64, r/m16: REX.W 0x0F 0xBF /r
                        let mut rex = 0x48u8; // REX.W
                        if hw_encoding(*dst) >= 8 {
                            rex |= 0x04; // REX.R
                        }
                        if hw_encoding(*src) >= 8 {
                            rex |= 0x01; // REX.B
                        }
                        bytes.push(rex);
                        bytes.push(0x0F);
                        bytes.push(0xBF);
                        bytes.push(modrm_byte(
                            0b11,
                            hw_encoding(*dst) & 0x7,
                            hw_encoding(*src) & 0x7,
                        ));
                    }
                    _ => {
                        // MOVSXD r64, r/m32: REX.W 0x63 /r (default for 4 bytes or unset)
                        if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                            bytes.push(rex);
                        }
                        bytes.push(0x63);
                        bytes.push(modrm_byte(
                            0b11,
                            hw_encoding(*dst) & 0x7,
                            hw_encoding(*src) & 0x7,
                        ));
                    }
                }
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(10);
                match src_bytes {
                    1 => {
                        // MOVSX r64, m8: REX.W 0x0F 0xBE /r
                        let mut rex = 0x48u8; // REX.W
                        if hw_encoding(*dst) >= 8 {
                            rex |= 0x04; // REX.R
                        }
                        if let Some(b) = base {
                            if hw_encoding(*b) >= 8 {
                                rex |= 0x01; // REX.B
                            }
                        }
                        if let Some(x) = index {
                            if hw_encoding(*x) >= 8 {
                                rex |= 0x02; // REX.X
                            }
                        }
                        bytes.push(rex);
                        bytes.push(0x0F);
                        bytes.push(0xBE);
                    }
                    2 => {
                        // MOVSX r64, m16: REX.W 0x0F 0xBF /r
                        let mut rex = 0x48u8; // REX.W
                        if hw_encoding(*dst) >= 8 {
                            rex |= 0x04; // REX.R
                        }
                        if let Some(b) = base {
                            if hw_encoding(*b) >= 8 {
                                rex |= 0x01; // REX.B
                            }
                        }
                        if let Some(x) = index {
                            if hw_encoding(*x) >= 8 {
                                rex |= 0x02; // REX.X
                            }
                        }
                        bytes.push(rex);
                        bytes.push(0x0F);
                        bytes.push(0xBF);
                    }
                    _ => {
                        // MOVSXD r64, m32: REX.W 0x63 /r
                        if let Some(rex) = compute_rex(true, Some(*dst), *index, *base) {
                            bytes.push(rex);
                        }
                        bytes.push(0x63);
                    }
                }
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode conditional jump (Jcc rel32).
    fn encode_jcc(&mut self, inst: &MachineInstruction, cc: u8) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Immediate(target)) => {
                // Jcc rel32: 0x0F 0x80+cc rel32
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                let rel = *target - (self.current_offset as i64 + 6); // 6 = instruction length
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::BlockLabel(label)) => {
                // Jcc rel32 with placeholder — will need patching
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
                let reloc = RelocationEntry::new(
                    (self.current_offset + 2) as u64,
                    format!(".L{}", label),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::GlobalSymbol(sym)) => {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0x0F);
                bytes.push(0x80 + cc);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 2) as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode JMP instruction.
    fn encode_jmp(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::Immediate(target)) => {
                // JMP rel32: 0xE9 rel32
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                let rel = *target - (self.current_offset as i64 + 5);
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::BlockLabel(label)) => {
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 1) as u64,
                    format!(".L{}", label),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::GlobalSymbol(sym)) => {
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE9);
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    (self.current_offset + 1) as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::Register(reg)) => {
                // JMP r/m64: 0xFF /4
                let mut bytes = Vec::with_capacity(3);
                if needs_rex(*reg) {
                    bytes.push(rex_byte(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm_byte(0b11, 4, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CALL instruction.
    fn encode_call(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match ops.first() {
            Some(MachineOperand::GlobalSymbol(sym)) => {
                // CALL rel32: 0xE8 rel32
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE8);
                let reloc_offset = self.current_offset + 1;
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::pc32_call(
                    reloc_offset as u64,
                    sym.clone(),
                    true, // use PLT for PIC safety
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            Some(MachineOperand::Register(reg)) => {
                // CALL r/m64: 0xFF /2
                let mut bytes = Vec::with_capacity(3);
                if needs_rex(*reg) {
                    bytes.push(rex_byte(false, false, false, true));
                }
                bytes.push(0xFF);
                bytes.push(modrm_byte(0b11, 2, hw_encoding(*reg) & 0x7));
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Immediate(target)) => {
                // CALL rel32 with absolute target
                let mut bytes = Vec::with_capacity(5);
                bytes.push(0xE8);
                let rel = *target - (self.current_offset as i64 + 5);
                bytes.extend_from_slice(&(rel as i32).to_le_bytes());
                EncodedInstruction::new(bytes)
            }
            Some(MachineOperand::Memory {
                base,
                index,
                scale,
                displacement,
            }) => {
                // CALL [mem]: 0xFF /2
                let mem_enc = encode_memory_operand(2, *base, *index, *scale, *displacement);
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(false, None, *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0xFF);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CMOVcc instruction.
    fn encode_cmovcc(&mut self, inst: &MachineInstruction, cc: u8) -> EncodedInstruction {
        // Handle 3-operand normalised form: [dst, dst_dup, src] → [dst, src].
        // The codegen emits mk_inst(CMOVcc, Some(dst), &[dst, src]) and the
        // normaliser prepends `result` to get [dst, dst, src].
        let ops_vec: Vec<MachineOperand>;
        let ops: &[MachineOperand] = if inst.operands.len() >= 3 {
            ops_vec = vec![inst.operands[0].clone(), inst.operands[2].clone()];
            &ops_vec
        } else {
            ops_vec = Vec::new();
            let _ = &ops_vec; // suppress unused
            &inst.operands
        };
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src))) => {
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(true, Some(*dst), None, Some(*src)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x40 + cc);
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*dst) & 0x7,
                    hw_encoding(*src) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(dst)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*dst) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(true, Some(*dst), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x40 + cc);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode SETcc instruction.
    fn encode_setcc(&mut self, _inst: &MachineInstruction, cc: u8, dst: u16) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(5);
        // SETcc needs REX if the destination is one of the high byte regs or R8-R15
        if needs_rex(dst) {
            bytes.push(rex_byte(false, false, false, true));
        } else {
            // Need REX.=0 for SPL/BPL/SIL/DIL access in byte operations
            let hw = hw_encoding(dst);
            if (4..=7).contains(&hw) {
                bytes.push(0x40); // plain REX to access low byte of RSP/RBP/RSI/RDI
            }
        }
        bytes.push(0x0F);
        bytes.push(0x90 + cc);
        bytes.push(modrm_byte(0b11, 0, hw_encoding(dst) & 0x7));
        EncodedInstruction::new(bytes)
    }

    /// Encode MOVSD (scalar double).
    fn encode_movsd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)))
                if is_sse(*dst) && is_sse(*src) =>
            {
                self.encode_sse_reg_reg(0xF2, &[0x0F, 0x10], *dst, *src)
            }
            // Store: movsd [gpr], xmm — GPR pointer, XMM source.
            // Encodes as MOVSD [reg], xmm → F2 0F 11 /r (mod=00 r/m=gpr)
            (Some(MachineOperand::Register(gpr)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) && !is_sse(*gpr) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(*gpr), None, 1, 0);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Load: movsd xmm, [gpr] — XMM destination, GPR pointer.
            // Encodes as MOVSD xmm, [reg] → F2 0F 10 /r (mod=00 r/m=gpr)
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Register(gpr)))
                if is_sse(*xmm) && !is_sse(*gpr) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(*gpr), None, 1, 0);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(xmm)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) if is_sse(*xmm) => {
                // MOVSD xmm, [mem]: F2 0F 10 /r
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Register(xmm)),
            ) if is_sse(*xmm) => {
                // MOVSD [mem], xmm: F2 0F 11 /r
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::GlobalSymbol(sym)))
                if is_sse(*xmm) =>
            {
                // MOVSD xmm, [rip + sym]: F2 0F 10 /r with RIP-relative
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, None) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(modrm_byte(0b00, xmm_enc, 0b101)); // RIP-relative
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            (Some(MachineOperand::GlobalSymbol(sym)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) =>
            {
                // MOVSD [rip + sym], xmm: F2 0F 11 /r with RIP-relative
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, None) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(modrm_byte(0b00, xmm_enc, 0b101)); // RIP-relative
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::FrameSlot(off)))
                if is_sse(*xmm) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(RBP), None, 1, *off as i64);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RBP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::FrameSlot(off)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(RBP), None, 1, *off as i64);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RBP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Movsd xmm, Immediate(0) — zero a floating-point register.
            // Emitted by the register allocator's spill resolution when a
            // float value maps to Immediate(0) (the get_value fallback for
            // unmapped Values).  Encode as `xorps xmm, xmm` (0F 57 /r)
            // which zeroes the entire 128-bit register.
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Immediate(0)))
                if is_sse(*xmm) =>
            {
                let enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(5);
                // REX prefix if xmm >= XMM8 (need both REX.R and REX.B
                // since the same register appears in both reg and r/m).
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*xmm)) {
                    bytes.push(rex);
                }
                // xorps xmm, xmm: 0F 57 /r (mod=11, reg=xmm, r/m=xmm)
                bytes.push(0x0F);
                bytes.push(0x57);
                bytes.push(modrm_byte(0b11, enc, enc));
                EncodedInstruction::new(bytes)
            }
            // Movsd xmm, Immediate(non-zero) — materialize a float constant
            // from its raw bit pattern.  This is unusual (float constants
            // should normally come from the .rodata pool), but can arise
            // when the constant cache misses.  We store the 64-bit
            // immediate to a temporary stack slot and load it back via
            // movsd.  We use [RSP-8] as scratch space (below the current
            // RSP, inside the red zone on System V AMD64).
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Immediate(val)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(32);
                let imm_bytes = (*val).to_le_bytes();
                // mov qword [rsp-8], imm32 (if fits) or mov rax, imm64 + mov [rsp-8], rax
                // Use: sub rsp, 8; mov [rsp], low32; mov [rsp+4], high32; movsd xmm, [rsp]; add rsp, 8
                // Simplest: push the 64-bit value via two 32-bit halves.
                // Actually use MOV r64, imm64 via RAX then MOV [RSP-8], RAX then MOVSD xmm, [RSP-8].
                let lo = i64::from_le_bytes(imm_bytes);
                // REX.W MOV RAX, imm64: 48 B8 <8 bytes>
                bytes.push(0x48);
                bytes.push(0xB8);
                bytes.extend_from_slice(&lo.to_le_bytes());
                // MOV [RSP - 8], RAX: 48 89 44 24 F8
                bytes.push(0x48);
                bytes.push(0x89);
                bytes.push(0x44); // ModRM: mod=01, reg=RAX(0), r/m=100 (SIB)
                bytes.push(0x24); // SIB: scale=0, index=RSP(4), base=RSP(4)
                bytes.push(0xF8u8); // disp8 = -8
                                    // MOVSD xmm, [RSP - 8]: F2 [REX] 0F 10 /r
                bytes.push(0xF2);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RSP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                bytes.push(0x44 | (xmm_enc << 3)); // ModRM: mod=01, reg=xmm, r/m=100 (SIB)
                bytes.push(0x24); // SIB: RSP-based
                bytes.push(0xF8u8); // disp8 = -8
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode MOVSS (scalar single).
    fn encode_movss(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)))
                if is_sse(*dst) && is_sse(*src) =>
            {
                self.encode_sse_reg_reg(0xF3, &[0x0F, 0x10], *dst, *src)
            }
            // Store: movss [gpr], xmm — GPR pointer, XMM source.
            (Some(MachineOperand::Register(gpr)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) && !is_sse(*gpr) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(*gpr), None, 1, 0);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Load: movss xmm, [gpr] — XMM destination, GPR pointer.
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Register(gpr)))
                if is_sse(*xmm) && !is_sse(*gpr) =>
            {
                let mem_enc =
                    encode_memory_operand(hw_encoding(*xmm) & 0x7, Some(*gpr), None, 1, 0);
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (
                Some(MachineOperand::Register(xmm)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::GlobalSymbol(sym)))
                if is_sse(*xmm) =>
            {
                // MOVSS xmm, [rip + sym]: F3 0F 10 /r with RIP-relative
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, None) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                bytes.push(modrm_byte(0b00, xmm_enc, 0b101));
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            (Some(MachineOperand::GlobalSymbol(sym)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) =>
            {
                // MOVSS [rip + sym], xmm: F3 0F 11 /r with RIP-relative
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, None) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(modrm_byte(0b00, xmm_enc, 0b101));
                let reloc_offset = self.current_offset + bytes.len();
                bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                let reloc = RelocationEntry::new(
                    reloc_offset as u64,
                    sym.clone(),
                    X86_64RelocationType::Pc32,
                    -4,
                    ".text".to_string(),
                );
                EncodedInstruction::with_relocations(bytes, vec![reloc])
            }
            (
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
                Some(MachineOperand::Register(xmm)),
            ) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x11);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            // Movss xmm, Immediate(0) — zero the register via xorps.
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Immediate(0)))
                if is_sse(*xmm) =>
            {
                let enc = hw_encoding(*xmm) & 0x7;
                let mut bytes = Vec::with_capacity(5);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(*xmm)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x57);
                bytes.push(modrm_byte(0b11, enc, enc));
                EncodedInstruction::new(bytes)
            }
            // Movss xmm, Immediate(non-zero) — materialize via stack scratch.
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Immediate(val)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(24);
                // MOV dword [RSP-4], imm32: C7 44 24 FC <imm32>
                bytes.push(0xC7);
                bytes.push(0x44);
                bytes.push(0x24);
                bytes.push(0xFC_u8); // disp8 = -4
                bytes.extend_from_slice(&(*val as i32).to_le_bytes());
                // MOVSS xmm, [RSP-4]: F3 [REX] 0F 10 /r
                bytes.push(0xF3);
                if let Some(rex) = compute_rex(false, Some(*xmm), None, Some(RSP)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x10);
                let xmm_enc = hw_encoding(*xmm) & 0x7;
                bytes.push(0x44 | (xmm_enc << 3));
                bytes.push(0x24);
                bytes.push(0xFC_u8);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode a generic SSE arithmetic operation (addsd, subsd, mulsd, divsd, etc.).
    fn encode_sse_op(
        &mut self,
        inst: &MachineInstruction,
        prefix: u8,
        opc_suffix: u8,
    ) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(dst)), Some(MachineOperand::Register(src)))
                if is_sse(*dst) && is_sse(*src) =>
            {
                self.encode_sse_reg_reg(prefix, &[0x0F, opc_suffix], *dst, *src)
            }
            (
                Some(MachineOperand::Register(xmm)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(prefix);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(opc_suffix);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode UCOMISD.
    /// Encode `MOVQ xmm, gpr` (66 REX.W 0F 6E /r) — moves 64-bit GPR value
    /// into the lower 64 bits of an XMM register.  Used to patch up float
    /// values that the register allocator placed in GPRs before a UCOMIS*
    /// comparison.
    fn encode_movq_gpr_to_xmm(&self, gpr: u16, xmm: u16) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(5);
        bytes.push(0x66); // mandatory prefix
                          // REX.W (64-bit operand) + register extension bits
        let rex = rex_byte(true, hw_encoding(xmm) >= 8, false, hw_encoding(gpr) >= 8);
        bytes.push(rex);
        bytes.push(0x0F);
        bytes.push(0x6E); // MOVQ xmm, r/m64
        bytes.push(modrm_byte(
            0b11,
            hw_encoding(xmm) & 0x7,
            hw_encoding(gpr) & 0x7,
        ));
        bytes
    }

    fn encode_ucomisd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        // Helper: resolve operands; if a GPR is encountered, emit MOVQ GPR→XMM15
        // and substitute XMM15 as the comparison operand.
        let lhs = ops.first().and_then(|o| {
            if let MachineOperand::Register(r) = o {
                Some(*r)
            } else {
                None
            }
        });
        let rhs = ops.get(1).and_then(|o| {
            if let MachineOperand::Register(r) = o {
                Some(*r)
            } else {
                None
            }
        });

        if let (Some(mut lhs_reg), Some(mut rhs_reg)) = (lhs, rhs) {
            let mut prefix_bytes = Vec::new();
            // If lhs is GPR, move to XMM15 via MOVQ (66 REX.W 0F 6E /r)
            const XMM15: u16 = 31;
            const XMM14: u16 = 30;
            if !is_sse(lhs_reg) {
                prefix_bytes.extend_from_slice(&self.encode_movq_gpr_to_xmm(lhs_reg, XMM15));
                lhs_reg = XMM15;
            }
            if !is_sse(rhs_reg) {
                let target = if lhs_reg == XMM15 { XMM14 } else { XMM15 };
                prefix_bytes.extend_from_slice(&self.encode_movq_gpr_to_xmm(rhs_reg, target));
                rhs_reg = target;
            }
            let mut bytes = prefix_bytes;
            bytes.push(0x66);
            if let Some(rex) = compute_rex(false, Some(lhs_reg), None, Some(rhs_reg)) {
                bytes.push(rex);
            }
            bytes.push(0x0F);
            bytes.push(0x2E);
            bytes.push(modrm_byte(
                0b11,
                hw_encoding(lhs_reg) & 0x7,
                hw_encoding(rhs_reg) & 0x7,
            ));
            return EncodedInstruction::new(bytes);
        }

        match (ops.first(), ops.get(1)) {
            (
                Some(MachineOperand::Register(xmm)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(12);
                bytes.push(0x66);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2E);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode UCOMISS — unordered compare scalar single-precision.
    /// Encoding: `0F 2E /r` (no mandatory prefix, unlike UCOMISD which uses 0x66).
    fn encode_ucomiss(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        // Handle GPR→XMM conversion similarly to encode_ucomisd.
        let lhs = ops.first().and_then(|o| {
            if let MachineOperand::Register(r) = o {
                Some(*r)
            } else {
                None
            }
        });
        let rhs = ops.get(1).and_then(|o| {
            if let MachineOperand::Register(r) = o {
                Some(*r)
            } else {
                None
            }
        });
        if let (Some(mut lhs_reg), Some(mut rhs_reg)) = (lhs, rhs) {
            let mut prefix_bytes = Vec::new();
            const XMM15: u16 = 31;
            const XMM14: u16 = 30;
            if !is_sse(lhs_reg) {
                prefix_bytes.extend_from_slice(&self.encode_movq_gpr_to_xmm(lhs_reg, XMM15));
                lhs_reg = XMM15;
            }
            if !is_sse(rhs_reg) {
                let target = if lhs_reg == XMM15 { XMM14 } else { XMM15 };
                prefix_bytes.extend_from_slice(&self.encode_movq_gpr_to_xmm(rhs_reg, target));
                rhs_reg = target;
            }
            let mut bytes = prefix_bytes;
            // No mandatory prefix for UCOMISS.
            if let Some(rex) = compute_rex(false, Some(lhs_reg), None, Some(rhs_reg)) {
                bytes.push(rex);
            }
            bytes.push(0x0F);
            bytes.push(0x2E);
            bytes.push(modrm_byte(
                0b11,
                hw_encoding(lhs_reg) & 0x7,
                hw_encoding(rhs_reg) & 0x7,
            ));
            return EncodedInstruction::new(bytes);
        }

        match (ops.first(), ops.get(1)) {
            (
                Some(MachineOperand::Register(xmm)),
                Some(MachineOperand::Memory {
                    base,
                    index,
                    scale,
                    displacement,
                }),
            ) if is_sse(*xmm) => {
                let mem_enc = encode_memory_operand(
                    hw_encoding(*xmm) & 0x7,
                    *base,
                    *index,
                    *scale,
                    *displacement,
                );
                let mut bytes = Vec::with_capacity(10);
                if let Some(rex) = compute_rex(false, Some(*xmm), *index, *base) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2E);
                bytes.push(mem_enc.modrm);
                if let Some(s) = mem_enc.sib {
                    bytes.push(s);
                }
                bytes.extend_from_slice(&mem_enc.displacement);
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CVTSI2SD (integer to double).
    ///
    /// `operand_size` controls REX.W: 8 → 64-bit source, else 32-bit.
    /// A 32-bit signed int (e.g. -1 in EAX) must be converted WITHOUT
    /// REX.W so that `cvtsi2sd` reads the 32-bit register and
    /// sign-extends internally, yielding -1.0 — not 4294967295.0.
    fn encode_cvtsi2sd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Register(gpr)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF2);
                let want_rexw = inst.operand_size == 8;
                if let Some(rex) = compute_rex(want_rexw, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2A);
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*xmm) & 0x7,
                    hw_encoding(*gpr) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CVTTSD2SI — truncating double-to-signed-integer conversion.
    ///
    /// Uses opcode `0x2C` (truncate toward zero) rather than `0x2D`
    /// (round per MXCSR) to match C cast semantics `(int)d`.
    /// Format: `F2 [REX.W] 0F 2C /r` — GPR ← truncate(XMM).
    /// `operand_size` controls REX.W: 8 → 64-bit dest, else 32-bit.
    fn encode_cvtsd2si(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(gpr)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF2);
                let want_rexw = inst.operand_size == 0 || inst.operand_size == 8;
                if let Some(rex) = compute_rex(want_rexw, Some(*gpr), None, Some(*xmm)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2C); // CVTTSD2SI — truncation
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*gpr) & 0x7,
                    hw_encoding(*xmm) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CVTSI2SS — signed-integer-to-float conversion.
    ///
    /// Format: `F3 [REX.W] 0F 2A /r` — XMM ← convert(GPR).
    /// `operand_size` controls REX.W: 8 → 64-bit source, else 32-bit.
    fn encode_cvtsi2ss(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(xmm)), Some(MachineOperand::Register(gpr)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF3);
                let want_rexw = inst.operand_size == 8;
                if let Some(rex) = compute_rex(want_rexw, Some(*xmm), None, Some(*gpr)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2A);
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*xmm) & 0x7,
                    hw_encoding(*gpr) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CVTTSS2SI — truncating float-to-signed-integer conversion.
    ///
    /// Format: `F3 [REX.W] 0F 2C /r` — GPR ← truncate(XMM).
    fn encode_cvtss2si(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        let ops = &inst.operands;
        match (ops.first(), ops.get(1)) {
            (Some(MachineOperand::Register(gpr)), Some(MachineOperand::Register(xmm)))
                if is_sse(*xmm) =>
            {
                let mut bytes = Vec::with_capacity(6);
                bytes.push(0xF3);
                let want_rexw = inst.operand_size == 0 || inst.operand_size == 8;
                if let Some(rex) = compute_rex(want_rexw, Some(*gpr), None, Some(*xmm)) {
                    bytes.push(rex);
                }
                bytes.push(0x0F);
                bytes.push(0x2C); // CVTTSS2SI — truncation
                bytes.push(modrm_byte(
                    0b11,
                    hw_encoding(*gpr) & 0x7,
                    hw_encoding(*xmm) & 0x7,
                ));
                EncodedInstruction::new(bytes)
            }
            _ => {
                if crate::common::verbosity::verbose_enabled() {
                    eprintln!(
                        "[UD2] opcode={} ops={:?} res={:?}",
                        inst.opcode, inst.operands, inst.result
                    );
                }
                EncodedInstruction::new(vec![0x0F, 0x0B])
            }
        }
    }

    /// Encode CVTSD2SS — double-to-float conversion.
    ///
    /// Format: `F2 0F 5A /r` — XMM(float) ← convert(XMM(double)).
    fn encode_cvtsd2ss(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        self.encode_sse_op(inst, 0xF2, 0x5A)
    }

    /// Encode CVTSS2SD — float-to-double conversion.
    ///
    /// Format: `F3 0F 5A /r` — XMM(double) ← convert(XMM(float)).
    fn encode_cvtss2sd(&mut self, inst: &MachineInstruction) -> EncodedInstruction {
        self.encode_sse_op(inst, 0xF3, 0x5A)
    }
    /// Encode `MOV r64, imm64` — loads a 64-bit immediate into a register.
    /// Used to materialise absolute addresses for indirect load/store when
    /// the pointer operand is an unresolved immediate value.
    fn encode_mov_imm64(&self, dst: u16, imm: i64) -> EncodedInstruction {
        let mut bytes = Vec::with_capacity(10);
        let rex =
            compute_rex(true, None, None, Some(dst)).unwrap_or(rex_byte(true, false, false, false));
        bytes.push(rex);
        bytes.push(0xB8 + (hw_encoding(dst) & 0x7));
        bytes.extend_from_slice(&imm.to_le_bytes());
        EncodedInstruction::new(bytes)
    }
} // end impl X86_64Encoder

// =====================================================================
// Unit Tests
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rex_byte() {
        assert_eq!(rex_byte(false, false, false, false), 0x40);
        assert_eq!(rex_byte(true, false, false, false), 0x48);
        assert_eq!(rex_byte(false, true, false, false), 0x44);
        assert_eq!(rex_byte(false, false, true, false), 0x42);
        assert_eq!(rex_byte(false, false, false, true), 0x41);
        assert_eq!(rex_byte(true, true, true, true), 0x4F);
    }

    #[test]
    fn test_modrm_byte() {
        // mod=11 reg=000 r/m=001 => 0b11_000_001 = 0xC1
        assert_eq!(modrm_byte(0b11, 0, 1), 0xC1);
        // mod=00 reg=101 r/m=100 => 0b00_101_100 = 0x2C
        assert_eq!(modrm_byte(0b00, 5, 4), 0x2C);
        // mod=01 reg=011 r/m=101 => 0b01_011_101 = 0x5D
        assert_eq!(modrm_byte(0b01, 3, 5), 0x5D);
    }

    #[test]
    fn test_sib_byte() {
        // scale=00 index=100 base=101 => 0b00_100_101 = 0x25
        assert_eq!(sib_byte(0b00, 4, 5), 0x25);
        // scale=10 index=001 base=000 => 0b10_001_000 = 0x88
        assert_eq!(sib_byte(0b10, 1, 0), 0x88);
    }

    #[test]
    fn test_scale_encoding() {
        assert_eq!(scale_encoding(1), 0b00);
        assert_eq!(scale_encoding(2), 0b01);
        assert_eq!(scale_encoding(4), 0b10);
        assert_eq!(scale_encoding(8), 0b11);
    }

    #[test]
    fn test_encode_displacement_i8() {
        let d = encode_displacement(10);
        assert_eq!(d, vec![10u8]);
    }

    #[test]
    fn test_encode_displacement_i32() {
        let d = encode_displacement(256);
        assert_eq!(d, (256i32).to_le_bytes().to_vec());
    }

    #[test]
    fn test_encode_immediate_sizes() {
        assert_eq!(encode_immediate(0x12, 1), vec![0x12]);
        assert_eq!(encode_immediate(0x1234, 2), vec![0x34, 0x12]);
        assert_eq!(
            encode_immediate(0x12345678, 4),
            vec![0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn test_compute_rex_none_needed() {
        assert!(compute_rex(false, Some(RAX), None, Some(RBX)).is_none());
    }

    #[test]
    fn test_compute_rex_w_only() {
        let rex = compute_rex(true, Some(RAX), None, Some(RBX));
        assert_eq!(rex, Some(0x48));
    }

    #[test]
    fn test_compute_rex_b_set() {
        // R8 needs REX.B
        let rex = compute_rex(true, Some(RAX), None, Some(R8));
        assert_eq!(rex, Some(0x49)); // W + B
    }

    #[test]
    fn test_compute_rex_r_set() {
        // R10 in reg field needs REX.R
        let rex = compute_rex(true, Some(R10), None, Some(RAX));
        assert_eq!(rex, Some(0x4C)); // W + R
    }

    #[test]
    fn test_encoder_nop() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Nop.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x90]);
    }

    #[test]
    fn test_encoder_ret() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Ret.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xC3]);
    }

    #[test]
    fn test_encoder_push_rax() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Push.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x50]);
    }

    #[test]
    fn test_encoder_push_r12() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Push.as_u32());
        inst.operands.push(MachineOperand::Register(R12));
        let result = enc.encode_instruction(&inst);
        // R12 hw_encoding = 12, lower 3 bits = 4, REX.B needed
        assert_eq!(result.bytes, vec![0x41, 0x50 + 4]);
    }

    #[test]
    fn test_encoder_pop_rbp() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Pop.as_u32());
        inst.operands.push(MachineOperand::Register(RBP));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x58 + 5]); // 0x5D
    }

    #[test]
    fn test_encoder_cdq() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Cdq.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x99]);
    }

    #[test]
    fn test_encoder_cqo() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Cqo.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x48, 0x99]);
    }

    #[test]
    fn test_encoder_endbr64() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Endbr64.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xF3, 0x0F, 0x1E, 0xFA]);
    }

    #[test]
    fn test_encoder_pause() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Pause.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0xF3, 0x90]);
    }

    #[test]
    fn test_encoder_lfence() {
        let mut enc = X86_64Encoder::new(0);
        let inst = MachineInstruction::new(X86Opcode::Lfence.as_u32());
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes, vec![0x0F, 0xAE, 0xE8]);
    }

    #[test]
    fn test_encoder_add_reg_reg() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Add.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.operands.push(MachineOperand::Register(RCX));
        let result = enc.encode_instruction(&inst);
        // ADD RAX, RCX: REX.W 0x01 ModRM(11,001,000)
        assert_eq!(result.bytes[0], 0x48); // REX.W
        assert_eq!(result.bytes[1], 0x01); // ADD r/m,r
        assert_eq!(result.bytes[2], modrm_byte(0b11, 1, 0)); // 0xC8
    }

    #[test]
    fn test_encoder_add_reg_imm8() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Add.as_u32());
        inst.operands.push(MachineOperand::Register(RAX));
        inst.operands.push(MachineOperand::Immediate(42));
        let result = enc.encode_instruction(&inst);
        // ADD RAX, 42: REX.W 0x83 ModRM(11, /0, RAX) imm8
        assert_eq!(result.bytes[0], 0x48); // REX.W
        assert_eq!(result.bytes[1], 0x83);
        assert_eq!(result.bytes[2], modrm_byte(0b11, 0, 0)); // 0xC0
        assert_eq!(result.bytes[3], 42);
    }

    #[test]
    fn test_encoder_mov_reg_reg() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Mov.as_u32());
        inst.operands.push(MachineOperand::Register(RDI));
        inst.operands.push(MachineOperand::Register(RSI));
        let result = enc.encode_instruction(&inst);
        // MOV RDI, RSI: REX.W 0x89 ModRM(11, RSI=6, RDI=7)
        assert_eq!(result.bytes[0], 0x48);
        assert_eq!(result.bytes[1], 0x89);
        assert_eq!(result.bytes[2], modrm_byte(0b11, 6, 7));
    }

    #[test]
    fn test_encoder_call_symbol() {
        let mut enc = X86_64Encoder::new(0);
        let mut inst = MachineInstruction::new(X86Opcode::Call.as_u32());
        inst.operands
            .push(MachineOperand::GlobalSymbol("printf".to_string()));
        let result = enc.encode_instruction(&inst);
        assert_eq!(result.bytes[0], 0xE8);
        assert_eq!(result.bytes.len(), 5);
        assert_eq!(result.relocations.len(), 1);
        assert_eq!(result.relocations[0].symbol, "printf");
    }
}
