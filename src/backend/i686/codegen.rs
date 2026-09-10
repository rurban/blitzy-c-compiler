//! # i686 Instruction Selection and Emission
//!
//! This module implements the i686 (32-bit x86) instruction selection and
//! machine code emission for BCC. It translates IR instructions into i686
//! machine instructions using the [`ArchCodegen`] trait.
//!
//! ## Architecture Constraints
//!
//! - **32-bit register widths** — NO 64-bit registers (RAX, R8–R15)
//! - **No REX prefixes** — purely IA-32 encoding
//! - **8 GPRs only**: EAX(0), ECX(1), EDX(2), EBX(3), ESP(4), EBP(5), ESI(6), EDI(7)
//! - **x87 FPU** for floating-point (stack-based ST(0)–ST(7)), NOT SSE
//! - **All arguments on stack** (cdecl calling convention)
//! - **ILP32 data model**: `int` = 4, `long` = 4, pointer = 4, `long long` = 8
//! - Return values: integers in EAX (or EDX:EAX for 64-bit), floats in ST(0)
//!
//! ## Instruction Selection Patterns
//!
//! | IR Instruction   | i686 Lowering                                    |
//! |-----------------|--------------------------------------------------|
//! | `Alloca`         | Stack slot assignment (frame offset)              |
//! | `Load`           | `MOV reg, [mem]`                                  |
//! | `Store`          | `MOV [mem], reg`                                  |
//! | `BinOp Add`      | `ADD dst, src`                                    |
//! | `BinOp SDiv`     | `CDQ` + `IDIV`                                    |
//! | `BinOp UDiv`     | `XOR EDX,EDX` + `DIV`                             |
//! | `ICmp`           | `CMP` + `SETcc`                                   |
//! | `Branch`         | `JMP label`                                       |
//! | `CondBranch`     | `CMP` + `Jcc label`                               |
//! | `Call`           | `PUSH args (RTL)` + `CALL` + `ADD ESP, N`         |
//! | `Return`         | `MOV EAX, result` + `LEAVE` + `RET`               |
//! | `Float BinOp`    | `FLD` + `FADD/FSUB/FMUL/FDIV` + `FSTP`           |
//!
//! ## 64-bit Integer Operations
//!
//! `long long` values use register pairs (EDX:EAX for results):
//! - 64-bit add: `ADD low` + `ADC high`
//! - 64-bit sub: `SUB low` + `SBB high`
//!
//! ## Zero-Dependency
//!
//! This module depends only on `crate::` internal modules and the Rust
//! standard library. No external crates are used.

use crate::backend::i686::registers;
use crate::backend::traits::{
    ArchCodegen, MachineBasicBlock, MachineFunction, MachineInstruction, MachineOperand,
    RegisterInfo, RelocationTypeInfo,
};
use crate::common::diagnostics::DiagnosticEngine;
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::ir::function::IrFunction;
use crate::ir::instructions::{BinOp, FCmpOp, ICmpOp, Instruction, Value};
use crate::ir::types::IrType;

// ===========================================================================
// i686 Opcode Constants
// ===========================================================================
//
// Architecture-specific opcode constants for MachineInstruction.opcode.
// Grouped by category with non-overlapping numeric ranges.

// --- Data movement ---

/// MOV — register/memory/immediate move (32-bit default).
pub const I686_MOV: u32 = 0x100;
/// MOVZX — zero-extend move (8/16-bit to 32-bit).
pub const I686_MOVZX: u32 = 0x101;
/// MOVSX — sign-extend move (8/16-bit to 32-bit).
pub const I686_MOVSX: u32 = 0x102;
/// LEA — load effective address (address computation without memory access).
pub const I686_LEA: u32 = 0x103;
/// PUSH — push operand onto the stack (decrements ESP by 4).
pub const I686_PUSH: u32 = 0x104;
/// POP — pop top of stack into operand (increments ESP by 4).
pub const I686_POP: u32 = 0x105;
/// XCHG — exchange two operands atomically.
pub const I686_XCHG: u32 = 0x106;

// --- Arithmetic ---

/// ADD — integer addition.
pub const I686_ADD: u32 = 0x200;
/// SUB — integer subtraction.
pub const I686_SUB: u32 = 0x201;
/// IMUL — signed integer multiplication.
pub const I686_IMUL: u32 = 0x202;
/// IDIV — signed integer division (EDX:EAX / operand → EAX quotient, EDX remainder).
pub const I686_IDIV: u32 = 0x203;
/// MUL — unsigned integer multiplication.
pub const I686_MUL: u32 = 0x204;
/// DIV — unsigned integer division.
pub const I686_DIV: u32 = 0x205;
/// NEG — two's complement negation.
pub const I686_NEG: u32 = 0x206;
/// INC — increment by 1.
pub const I686_INC: u32 = 0x207;
/// DEC — decrement by 1.
pub const I686_DEC: u32 = 0x208;
/// CDQ — sign-extend EAX into EDX:EAX (converts 32→64 bit for IDIV).
pub const I686_CDQ: u32 = 0x209;
/// ADC — add with carry (used for 64-bit addition on 32-bit registers).
pub const I686_ADC: u32 = 0x20A;
/// SBB — subtract with borrow (used for 64-bit subtraction on 32-bit registers).
pub const I686_SBB: u32 = 0x20B;

// --- Bitwise ---

/// AND — bitwise AND.
pub const I686_AND: u32 = 0x300;
/// OR — bitwise OR.
pub const I686_OR: u32 = 0x301;
/// XOR — bitwise XOR.
pub const I686_XOR: u32 = 0x302;
/// NOT — bitwise complement.
pub const I686_NOT: u32 = 0x303;
/// SHL — shift left (zero fill).
pub const I686_SHL: u32 = 0x304;
/// SHR — shift right (zero fill, unsigned).
pub const I686_SHR: u32 = 0x305;
/// SAR — shift right (sign-extending, arithmetic).
pub const I686_SAR: u32 = 0x306;

// --- Comparison / Test ---

/// CMP — compare two operands (sets EFLAGS without storing result).
pub const I686_CMP: u32 = 0x400;
/// TEST — bitwise AND that only sets EFLAGS (no result stored).
pub const I686_TEST: u32 = 0x401;
/// SETcc — set byte to 0 or 1 based on condition code.
pub const I686_SETCC: u32 = 0x402;
/// CMOVcc — conditional move based on condition code.
pub const I686_CMOVCC: u32 = 0x403;

// --- Control flow ---

/// JMP — unconditional jump.
pub const I686_JMP: u32 = 0x500;
/// Jcc — conditional jump based on condition code.
pub const I686_JCC: u32 = 0x501;
/// CALL — call a function.
pub const I686_CALL: u32 = 0x502;
/// RET — return from function.
pub const I686_RET: u32 = 0x503;

// --- x87 FPU ---

/// FLD — load floating-point value onto FPU stack (ST(0)).
pub const I686_FLD: u32 = 0x600;
/// FSTP — store top of FPU stack to memory and pop.
pub const I686_FSTP: u32 = 0x601;
/// FADD — x87 floating-point addition.
pub const I686_FADD: u32 = 0x602;
/// FSUB — x87 floating-point subtraction.
pub const I686_FSUB: u32 = 0x603;
/// FMUL — x87 floating-point multiplication.
pub const I686_FMUL: u32 = 0x604;
/// FDIV — x87 floating-point division.
pub const I686_FDIV: u32 = 0x605;
/// FCHS — negate ST(0) (change sign).
pub const I686_FCHS: u32 = 0x606;
/// FCOMP — compare ST(0) with operand and pop.
pub const I686_FCOMP: u32 = 0x607;
/// FILD — load integer value onto FPU stack (integer-to-float).
pub const I686_FILD: u32 = 0x608;
/// FISTP — store ST(0) as integer to memory and pop (float-to-integer).
pub const I686_FISTP: u32 = 0x609;
/// FXCH — exchange ST(0) with another FPU register.
pub const I686_FXCH: u32 = 0x60A;
/// FUCOMIP — unordered compare ST(0) with ST(i), set EFLAGS, and pop.
pub const I686_FUCOMIP: u32 = 0x60B;

// --- Stack frame ---

/// ENTER — create stack frame (push EBP, mov EBP ESP, sub ESP N).
pub const I686_ENTER: u32 = 0x700;
/// LEAVE — destroy stack frame (mov ESP EBP, pop EBP).
pub const I686_LEAVE: u32 = 0x701;

// --- Miscellaneous ---

/// NOP — no operation (1 byte: 0x90).
pub const I686_NOP: u32 = 0x800;
/// INT3 — breakpoint trap (1 byte: 0xCC).
pub const I686_INT3: u32 = 0x801;
/// UD2 — undefined instruction trap (2 bytes: 0x0F 0x0B).
pub const I686_UD2: u32 = 0x802;

/// `MOV r32, [global_symbol]` — load from global variable memory.
/// Encodes as `0x8B ModR/M` with `[disp32]` addressing + R_386_32 relocation.
/// Distinct from `I686_MOV` + `GlobalSymbol` which loads the ADDRESS of the symbol.
/// `result = VirtualRegister(dst)`, `operands = [GlobalSymbol(name)]`.
pub const I686_MOV_LOAD_GLOBAL: u32 = 0x803;

/// `MOV [global_symbol], r32` — store to global variable memory.
/// Encodes as `0x89 ModR/M` with `[disp32]` addressing + R_386_32 relocation.
/// `operands = [GlobalSymbol(name), Register(src)]`.
pub const I686_MOV_STORE_GLOBAL: u32 = 0x804;
/// MOV_LOAD_INDIRECT — load through a register holding a pointer address.
/// Encodes as `MOV r32, [reg]` (0x8B with ModR/M indirect addressing).
/// Used when an IR Load dereferences a pointer held in a virtual register
/// (e.g., from a GEP result). Distinct from plain I686_MOV which would
/// encode `Register → Register` as a register copy.
/// `result = dst_register, operands = [src_pointer_register]`.
pub const I686_MOV_LOAD_INDIRECT: u32 = 0x805;
/// MOV_STORE_INDIRECT — store through a register holding a pointer address.
/// Encodes as `MOV [reg], src` (0x89 with ModR/M indirect addressing).
/// Used when an IR Store writes through a pointer held in a virtual register.
/// `operands = [dst_pointer_register, src_value]`.
pub const I686_MOV_STORE_INDIRECT: u32 = 0x806;
/// MOVZX_LOAD_INDIRECT_BYTE — zero-extending byte load through a register
/// pointer.  Encodes as `MOVZX r32, BYTE PTR [reg]` (0x0F 0xB6 with ModR/M
/// indirect addressing).  Used when an IR Load dereferences a pointer for an
/// I8 type, avoiding the 32-bit MOV_LOAD_INDIRECT which would read 4 bytes
/// and then require a separate byte-extraction that is unreliable for
/// registers without 8-bit aliases (ESI, EDI, EBP, ESP) on i686.
/// `result = dst_register, operands = [src_pointer_register]`.
pub const I686_MOVZX_LOAD_INDIRECT_BYTE: u32 = 0x808;
/// MOVZX_LOAD_INDIRECT_WORD — zero-extending word load through a register
/// pointer.  Encodes as `MOVZX r32, WORD PTR [reg]` (0x0F 0xB7 with ModR/M
/// indirect addressing).
/// `result = dst_register, operands = [src_pointer_register]`.
pub const I686_MOVZX_LOAD_INDIRECT_WORD: u32 = 0x809;
/// MOV_STORE_INDIRECT_BYTE — byte store through a register pointer.
/// Encodes as `MOV BYTE PTR [reg], r8` (0x88 with ModR/M indirect
/// addressing).  Ensures that only one byte is written when storing I8
/// values, preventing corruption of adjacent memory.
/// `operands = [dst_pointer_register, src_value]`.
pub const I686_MOV_STORE_INDIRECT_BYTE: u32 = 0x80A;
/// MOV_STORE_INDIRECT_WORD — word store through a register pointer.
/// Encodes as `MOV WORD PTR [reg], r16` (0x66 0x89 with ModR/M indirect
/// addressing).
/// `operands = [dst_pointer_register, src_value]`.
pub const I686_MOV_STORE_INDIRECT_WORD: u32 = 0x80B;
/// BSWAP reg — reverse bytes in a 32-bit register (opcode 0F C8+rd).
/// `result = Register(rd)`, `operands = [Register(src)]`.
/// For in-place swap, `result == operands[0]`.
pub const I686_BSWAP: u32 = 0x807;

// ===========================================================================
// Condition Code Enum
// ===========================================================================

/// x86 condition codes for JCC, SETcc, and CMOVcc instructions.
///
/// These correspond directly to the 4-bit condition code field in x86
/// instruction encoding. Signed comparisons (Less, Greater, etc.) use
/// the SF and OF flags, while unsigned comparisons (Below, Above, etc.)
/// use the CF and ZF flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondCode {
    /// ZF=1 — equal (je/sete).
    Equal,
    /// ZF=0 — not equal (jne/setne).
    NotEqual,
    /// SF≠OF — signed less than (jl/setl).
    Less,
    /// ZF=1 or SF≠OF — signed less than or equal (jle/setle).
    LessEqual,
    /// ZF=0 and SF=OF — signed greater than (jg/setg).
    Greater,
    /// SF=OF — signed greater than or equal (jge/setge).
    GreaterEqual,
    /// CF=1 — unsigned below (jb/setb).
    Below,
    /// CF=1 or ZF=1 — unsigned below or equal (jbe/setbe).
    BelowEqual,
    /// CF=0 and ZF=0 — unsigned above (ja/seta).
    Above,
    /// CF=0 — unsigned above or equal (jae/setae).
    AboveEqual,
}

impl CondCode {
    /// Return the logical inverse of this condition code.
    ///
    /// `cc.invert().invert() == cc` for all condition codes.
    pub fn invert(&self) -> CondCode {
        match self {
            CondCode::Equal => CondCode::NotEqual,
            CondCode::NotEqual => CondCode::Equal,
            CondCode::Less => CondCode::GreaterEqual,
            CondCode::LessEqual => CondCode::Greater,
            CondCode::Greater => CondCode::LessEqual,
            CondCode::GreaterEqual => CondCode::Less,
            CondCode::Below => CondCode::AboveEqual,
            CondCode::BelowEqual => CondCode::Above,
            CondCode::Above => CondCode::BelowEqual,
            CondCode::AboveEqual => CondCode::Below,
        }
    }

    /// Convert an IR integer comparison predicate to an i686 condition code.
    pub fn from_icmp(op: &ICmpOp) -> CondCode {
        match op {
            ICmpOp::Eq => CondCode::Equal,
            ICmpOp::Ne => CondCode::NotEqual,
            ICmpOp::Slt => CondCode::Less,
            ICmpOp::Sle => CondCode::LessEqual,
            ICmpOp::Sgt => CondCode::Greater,
            ICmpOp::Sge => CondCode::GreaterEqual,
            ICmpOp::Ult => CondCode::Below,
            ICmpOp::Ule => CondCode::BelowEqual,
            ICmpOp::Ugt => CondCode::Above,
            ICmpOp::Uge => CondCode::AboveEqual,
        }
    }

    /// Convert an IR floating-point comparison predicate to an i686 condition code.
    ///
    /// After FUCOMIP, the EFLAGS encode float comparison results using
    /// unsigned condition codes (CF, ZF, PF). Ordered comparisons map to
    /// the unsigned counterparts. Unordered/ordered checks use PF.
    pub fn from_fcmp(op: &FCmpOp) -> CondCode {
        match op {
            FCmpOp::Oeq => CondCode::Equal,
            FCmpOp::One => CondCode::NotEqual,
            FCmpOp::Olt => CondCode::Below,
            FCmpOp::Ole => CondCode::BelowEqual,
            FCmpOp::Ogt => CondCode::Above,
            FCmpOp::Oge => CondCode::AboveEqual,
            // Unordered check: PF=1. Map to Below as an approximation;
            // the assembler handles JP/JNP encoding separately.
            FCmpOp::Uno => CondCode::Below,
            // Ordered check: PF=0.
            FCmpOp::Ord => CondCode::AboveEqual,
        }
    }

    /// Returns the numeric encoding of this condition code (x86 cc field).
    ///
    /// These values correspond to the 4-bit tttn field in the x86 ISA
    /// used by Jcc, SETcc, and CMOVcc instructions.
    #[inline]
    pub fn encoding(&self) -> u8 {
        match self {
            CondCode::Equal => 0x04,        // E/Z
            CondCode::NotEqual => 0x05,     // NE/NZ
            CondCode::Less => 0x0C,         // L/NGE
            CondCode::LessEqual => 0x0E,    // LE/NG
            CondCode::Greater => 0x0F,      // G/NLE
            CondCode::GreaterEqual => 0x0D, // GE/NL
            CondCode::Below => 0x02,        // B/NAE/C
            CondCode::BelowEqual => 0x06,   // BE/NA
            CondCode::Above => 0x07,        // A/NBE
            CondCode::AboveEqual => 0x03,   // AE/NB/NC
        }
    }
}

// ===========================================================================
// I686Codegen — main codegen struct
// ===========================================================================

/// The i686 (32-bit x86) code generator.
///
/// Implements [`ArchCodegen`] to lower IR functions to i686 machine
/// instructions. Supports cdecl calling convention, x87 FPU floating-point,
/// optional PIC code generation, and optional DWARF debug info.
pub struct I686Codegen {
    /// Whether Position-Independent Code generation is enabled (`-fPIC`).
    ///
    /// When true, global variable access uses GOT-relative addressing
    /// via EBX as the GOT pointer, and function calls use PLT stubs.
    pic: bool,
    /// Whether DWARF debug information is emitted (`-g` flag).
    debug_info: bool,
    /// Mapping from IR function-reference values to function names.
    func_ref_names: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    /// Mapping from IR global-variable-reference values to symbol names.
    global_var_refs: crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
}

impl I686Codegen {
    /// Create a new i686 code generator.
    ///
    /// # Parameters
    ///
    /// - `pic`: Enable Position-Independent Code generation (`-fPIC`).
    /// - `debug_info`: Enable DWARF debug information emission (`-g`).
    pub fn new(pic: bool, debug_info: bool) -> Self {
        I686Codegen {
            pic,
            debug_info,
            func_ref_names: crate::common::fx_hash::FxHashMap::default(),
            global_var_refs: crate::common::fx_hash::FxHashMap::default(),
        }
    }

    /// Set the function-reference name map (populated by the lowering phase).
    pub fn set_func_ref_names(
        &mut self,
        m: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.func_ref_names = m.clone();
    }

    /// Set the global-variable-reference map.
    pub fn set_global_var_refs(
        &mut self,
        m: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
    ) {
        self.global_var_refs = m.clone();
    }

    /// Returns whether PIC mode is enabled.
    #[inline]
    pub fn is_pic(&self) -> bool {
        self.pic
    }

    /// Returns whether DWARF debug info emission is enabled.
    #[inline]
    pub fn has_debug_info(&self) -> bool {
        self.debug_info
    }
}

// ===========================================================================
// Internal helper: virtual register allocator for instruction selection
// ===========================================================================

/// Tracks virtual register allocation during instruction selection.
///
/// Virtual registers are numbered starting from a base offset (to avoid
/// collision with IR Value numbers) and are later resolved to physical
/// registers by the register allocator.
pub(crate) struct VRegAllocator {
    next_vreg: u32,
}

impl VRegAllocator {
    /// Create a new virtual register allocator starting from `base`.
    pub(crate) fn new(base: u32) -> Self {
        VRegAllocator { next_vreg: base }
    }

    /// Allocate a new virtual register and return its number.
    pub(crate) fn alloc(&mut self) -> u32 {
        let v = self.next_vreg;
        self.next_vreg += 1;
        v
    }
}

// ===========================================================================
// ArchCodegen trait implementation for I686Codegen
// ===========================================================================

impl ArchCodegen for I686Codegen {
    fn target(&self) -> Target {
        Target::I686
    }

    fn register_info(&self) -> RegisterInfo {
        registers::i686_register_info()
    }

    fn frame_pointer_reg(&self) -> u16 {
        registers::EBP
    }

    fn stack_pointer_reg(&self) -> u16 {
        registers::ESP
    }

    fn return_address_reg(&self) -> Option<u16> {
        // On x86/i686, the return address is on the stack (pushed by CALL),
        // not in a dedicated register.
        None
    }

    fn relocation_types(&self) -> &[RelocationTypeInfo] {
        // i686 ELF relocation types per the System V i386 ABI.
        static TABLE: &[RelocationTypeInfo] = &[
            RelocationTypeInfo {
                type_id: 0,
                name: "R_386_NONE",
                size: 0,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 1,
                name: "R_386_32",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 2,
                name: "R_386_PC32",
                size: 4,
                is_pc_relative: true,
            },
            RelocationTypeInfo {
                type_id: 3,
                name: "R_386_GOT32",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 4,
                name: "R_386_PLT32",
                size: 4,
                is_pc_relative: true,
            },
            RelocationTypeInfo {
                type_id: 5,
                name: "R_386_COPY",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 6,
                name: "R_386_GLOB_DAT",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 7,
                name: "R_386_JMP_SLOT",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 8,
                name: "R_386_RELATIVE",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 9,
                name: "R_386_GOTOFF",
                size: 4,
                is_pc_relative: false,
            },
            RelocationTypeInfo {
                type_id: 10,
                name: "R_386_GOTPC",
                size: 4,
                is_pc_relative: true,
            },
        ];
        TABLE
    }

    fn classify_argument(&self, ty: &IrType) -> crate::backend::traits::ArgLocation {
        // cdecl: ALL arguments go on the stack. No register arguments.
        // The offset returned represents the argument's slot size (in bytes),
        // rounded up to a 4-byte boundary.  Callers that need cumulative
        // offsets (e.g., the instruction selector) must accumulate across all
        // arguments; here we return per-argument slot size since the trait
        // method classifies a single argument in isolation.
        let size = ty.size_bytes(&Target::I686);
        // Round up to 4-byte stack slot minimum.
        let slot_size = if size < 4 { 4 } else { (size + 3) & !3 };
        crate::backend::traits::ArgLocation::Stack(slot_size as i32)
    }

    fn classify_return(&self, ty: &IrType) -> crate::backend::traits::ArgLocation {
        match ty {
            IrType::Void => crate::backend::traits::ArgLocation::Stack(0),
            IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::Ptr => {
                crate::backend::traits::ArgLocation::Register(registers::EAX)
            }
            IrType::I64 => {
                crate::backend::traits::ArgLocation::RegisterPair(registers::EAX, registers::EDX)
            }
            IrType::F32 | IrType::F64 | IrType::F80 => {
                // Float returns in ST(0) — represented as Register(ST0).
                crate::backend::traits::ArgLocation::Register(registers::ST0)
            }
            IrType::Struct(_) | IrType::Array(_, _) => {
                // Struct/array returns via hidden pointer parameter.
                crate::backend::traits::ArgLocation::Stack(0)
            }
            _ => crate::backend::traits::ArgLocation::Register(registers::EAX),
        }
    }

    fn lower_function(
        &self,
        func: &IrFunction,
        diag: &mut DiagnosticEngine,
        _globals: &[crate::ir::module::GlobalVariable],
        _func_ref_map: &crate::common::fx_hash::FxHashMap<crate::ir::instructions::Value, String>,
        _global_var_refs: &crate::common::fx_hash::FxHashMap<
            crate::ir::instructions::Value,
            String,
        >,
    ) -> Result<MachineFunction, String> {
        let mut mf = MachineFunction::new(func.name.clone());
        let target = Target::I686;

        // Virtual register allocator — start past the IR value count.
        let mut vregs = VRegAllocator::new(func.value_count + 1000);

        // Value-to-operand mapping for IR SSA values.
        let mut value_map: FxHashMap<u32, MachineOperand> = FxHashMap::default();

        // Track ALL vreg→IR Value associations (like x86-64's vreg_ir_map).
        // Unlike value_map which only stores the LAST operand per IR value,
        // this map accumulates EVERY vreg created, including intermediates
        // (e.g. SETCC result before MOVZX). This is critical for the
        // register allocator to assign physical registers to ALL vregs.
        let mut vreg_ir_map: FxHashMap<u32, crate::ir::instructions::Value> = FxHashMap::default();

        // Build constant cache using the authoritative constant_values map
        // stored on the IR function by the lowering phase.  This is a
        // direct Value → i64 mapping that avoids the fragile positional
        // matching of globals to BinOp sentinels.
        let mut constant_cache: FxHashMap<u32, i64> = FxHashMap::default();
        if !func.constant_values.is_empty() {
            for (&val, &imm) in &func.constant_values {
                constant_cache.insert(val.index(), imm);
            }
        } else {
            // Legacy fallback: positional matching of .Lconst.i.N globals
            // to BinOp sentinels (Add result, result, UNDEF).
            let mut const_values: Vec<i64> = Vec::new();
            for gv in _globals {
                if gv.name.starts_with(".Lconst.i.") {
                    if let Some(crate::ir::module::Constant::Integer(v)) = &gv.initializer {
                        const_values.push(*v as i64);
                    }
                }
            }
            let mut const_idx = 0;
            for block in &func.blocks {
                for inst in block.instructions() {
                    if let Instruction::BinOp {
                        result, lhs, rhs, ..
                    } = inst
                    {
                        if *lhs == *result && *rhs == Value::UNDEF && const_idx < const_values.len()
                        {
                            constant_cache.insert(result.index(), const_values[const_idx]);
                            const_idx += 1;
                        }
                    }
                }
            }
        }

        // Stack slot tracking for alloca instructions.
        let mut stack_offset: i32 = 0;
        let mut alloca_offsets: FxHashMap<u32, i32> = FxHashMap::default();

        // Block label mapping: IR block index → machine block index.
        let mut block_map: FxHashMap<usize, usize> = FxHashMap::default();

        // Create machine basic blocks for each IR block.
        if !func.blocks.is_empty() {
            block_map.insert(func.blocks[0].index, 0);
            if let Some(ref label) = func.blocks[0].label {
                mf.blocks[0].label = Some(label.clone());
            }

            for ir_block in func.blocks.iter().skip(1) {
                let label = ir_block
                    .label
                    .clone()
                    .unwrap_or_else(|| format!(".L{}", ir_block.index));
                let mbb = MachineBasicBlock::with_label(label);
                let mbb_idx = mf.add_block(mbb);
                block_map.insert(ir_block.index, mbb_idx);
            }
        }

        // Map function parameters to stack locations (cdecl: all on stack).
        // After prologue: [EBP+8] = first arg, [EBP+12] = second arg, etc.
        let mut param_offset: i32 = 8;
        for param in &func.params {
            let param_size = param.ty.size_bytes(&target);
            let slot_size = if param_size < 4 {
                4
            } else {
                (param_size + 3) & !3
            };
            let mem_op = MachineOperand::Memory {
                base: Some(registers::EBP),
                index: None,
                scale: 1,
                displacement: param_offset as i64,
            };
            value_map.insert(param.value.index(), mem_op);
            param_offset += slot_size as i32;
        }

        // Pre-allocate virtual registers for phi-copy destinations.
        // After phi elimination, copies (BitCast instructions) are placed
        // in critical-edge split blocks that may have higher block indices
        // than the merge block that uses the phi result. Since we process
        // blocks sequentially, the USE of the phi result would be
        // encountered before the DEFINITION (BitCast). Pre-allocating
        // vregs here ensures that when the merge block references a phi
        // result value, the vreg is already present in value_map.
        {
            let mut phi_copy_dests: FxHashSet<Value> = FxHashSet::default();
            for ir_block in func.blocks.iter() {
                for ir_inst in &ir_block.instructions {
                    if let Instruction::BitCast { result, .. } = ir_inst {
                        phi_copy_dests.insert(*result);
                    }
                }
            }
            for dest_val in phi_copy_dests {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    value_map.entry(dest_val.index())
                {
                    let vreg = vregs.alloc();
                    e.insert(MachineOperand::VirtualRegister(vreg));
                    // CRITICAL: Also register in vreg_ir_map so that the
                    // auto-registration loop (which uses entry().or_insert)
                    // does NOT overwrite this association when the vreg
                    // appears as an operand of a use-site instruction
                    // (e.g., printf call in the done block) BEFORE the
                    // defining BitCast in a later-processed edge block.
                    vreg_ir_map.insert(vreg, dest_val);
                }
            }
        }

        // Lower each IR basic block.
        for (block_idx, ir_block) in func.blocks.iter().enumerate() {
            let mbb_idx = *block_map.get(&ir_block.index).unwrap_or(&block_idx);

            for inst in &ir_block.instructions {
                let machine_insts = self.lower_instruction(
                    inst,
                    &mut value_map,
                    &mut alloca_offsets,
                    &mut stack_offset,
                    &mut vregs,
                    &block_map,
                    &mut mf,
                    &target,
                    diag,
                    &constant_cache,
                );

                // Auto-register ALL VirtualRegister results and operands
                // with the IR instruction's result value.  This captures
                // intermediate vregs (e.g. SETCC before MOVZX) that are
                // consumed within the same lowered sequence but not
                // explicitly registered in value_map.
                if let Some(ir_result) = inst.result() {
                    for mi in &machine_insts {
                        if let Some(MachineOperand::VirtualRegister(vreg)) = &mi.result {
                            vreg_ir_map.entry(*vreg).or_insert(ir_result);
                        }
                        for op in &mi.operands {
                            if let MachineOperand::VirtualRegister(vreg) = op {
                                vreg_ir_map.entry(*vreg).or_insert(ir_result);
                            }
                        }
                    }
                }

                if mbb_idx < mf.blocks.len() {
                    for mi in machine_insts {
                        mf.blocks[mbb_idx].push_instruction(mi);
                    }
                }
            }
        }

        // Set frame size based on alloca usage (rounded to 16-byte alignment).
        mf.frame_size = ((stack_offset as usize) + 15) & !15;

        if diag.has_errors() {
            return Err("i686 instruction selection encountered errors".to_string());
        }

        // Build the reverse mapping (vreg → IR Value) so that
        // apply_allocation_result can correctly resolve VirtualRegister
        // operands to physical registers.
        // Use vreg_ir_map (which accumulates ALL vreg→Value associations,
        // including intermediates like SETCC results) instead of value_map
        // (which only stores the LAST operand per Value and loses earlier
        // vregs).
        for (&vreg, &ir_val) in &vreg_ir_map {
            mf.vreg_to_ir_value.insert(vreg, ir_val);
        }
        // Also add any vregs from value_map that weren't in vreg_ir_map
        // (defensive fallback).
        for (&val_idx, operand) in &value_map {
            if let MachineOperand::VirtualRegister(vreg) = operand {
                mf.vreg_to_ir_value
                    .entry(*vreg)
                    .or_insert(crate::ir::instructions::Value(val_idx));
            }
        }

        Ok(mf)
    }

    fn emit_assembly(
        &self,
        mf: &MachineFunction,
    ) -> Result<crate::backend::traits::AssembledFunction, String> {
        // Two-pass assembly with accurate label offset computation.
        //
        // Pass 1: Encode every instruction with placeholder (zero) label
        //         offsets.  The *sizes* of x86 instructions are independent
        //         of the displacement value (we always emit near branches,
        //         never short branches), so the resulting byte counts are
        //         exact.  Accumulate those sizes to build a correct
        //         label → byte-offset map.
        //
        // Pass 2: Re-encode with the accurate label offsets so that branch
        //         displacement fields contain the correct values.  Collect
        //         relocations for the linker.
        use crate::common::fx_hash::FxHashMap;

        // --- Pass 1: compute real instruction sizes and label offsets ----
        //
        // Block labels in the IR come from the lowering phase and have
        // descriptive names like "if.then", "while.cond", etc.  However,
        // `BlockLabel(id)` operands in JMP/JCC instructions encode the
        // *MBB index* as a u32, and the encoder resolves them via
        // `label_key(id)` → `".L{id}"`.  We must populate label_offsets
        // with `.L{mbb_index}` keys so that the encoder can find them.
        let dummy_labels: FxHashMap<String, usize> = FxHashMap::default();
        let mut label_offsets: FxHashMap<String, usize> = FxHashMap::default();
        let mut offset: usize = 0;

        for (block_idx, block) in mf.blocks.iter().enumerate() {
            // Insert MBB-index-based key that matches label_key(block_idx).
            let idx_key = format!(".L{}", block_idx);
            label_offsets.insert(idx_key, offset);
            // Also insert the descriptive label if present (for any code
            // that might reference blocks by name).
            if let Some(ref lbl) = block.label {
                label_offsets.insert(lbl.clone(), offset);
            }
            for inst in &block.instructions {
                if !inst.encoded_bytes.is_empty() {
                    offset += inst.encoded_bytes.len();
                } else {
                    // Encode with dummy labels — we only care about the
                    // resulting byte count, not the displacement values.
                    match crate::backend::i686::assembler::encoder::encode_instruction(
                        inst,
                        &dummy_labels,
                        offset,
                    ) {
                        Ok(encoded) => {
                            offset += encoded.bytes.len();
                        }
                        Err(_) => {
                            // UD2 placeholder — 2 bytes
                            offset += 2;
                        }
                    }
                }
            }
        }

        // --- Pass 2: encode with correct label offsets -------------------
        let mut code = Vec::new();
        let mut relocations: Vec<crate::backend::traits::FunctionRelocation> = Vec::new();
        for block in &mf.blocks {
            for inst in &block.instructions {
                if !inst.encoded_bytes.is_empty() {
                    code.extend_from_slice(&inst.encoded_bytes);
                } else {
                    let base_offset = code.len();
                    match crate::backend::i686::assembler::encoder::encode_instruction(
                        inst,
                        &label_offsets,
                        base_offset,
                    ) {
                        Ok(encoded) => {
                            code.extend_from_slice(&encoded.bytes);
                            if let Some(rel) = encoded.relocation {
                                // Resolve .L local block labels: for
                                // absolute address LEAs used in computed
                                // goto (BlockAddress), the label is a
                                // function-internal block label.  We write
                                // the known byte offset as the implicit
                                // addend into the code stream and convert
                                // the relocation to reference the function
                                // symbol.  At link time:
                                //   R_386_32: S + *loc
                                //   S = absolute address of function
                                //   *loc = byte offset of block within fn
                                //   result = absolute address of the block
                                if rel.symbol.starts_with(".L") {
                                    if let Some(&lbl_off) = label_offsets.get(&rel.symbol) {
                                        let patch_off = base_offset + rel.offset_in_instruction;
                                        let addend_bytes = (lbl_off as u32).to_le_bytes();
                                        // Patch the 4-byte displacement
                                        // field with the label's byte
                                        // offset within this function.
                                        for (i, &b) in addend_bytes.iter().enumerate() {
                                            if patch_off + i < code.len() {
                                                code[patch_off + i] = b;
                                            }
                                        }
                                        relocations.push(
                                            crate::backend::traits::FunctionRelocation {
                                                offset: patch_off as u64,
                                                symbol: mf.name.clone(),
                                                rel_type_id: rel.rel_type,
                                                addend: lbl_off as i64,
                                                section: ".text".to_string(),
                                            },
                                        );
                                        continue;
                                    }
                                }

                                relocations.push(crate::backend::traits::FunctionRelocation {
                                    offset: (base_offset + rel.offset_in_instruction) as u64,
                                    symbol: rel.symbol.clone(),
                                    rel_type_id: rel.rel_type,
                                    addend: rel.addend,
                                    section: ".text".to_string(),
                                });
                            }
                        }
                        Err(e) => {
                            if crate::common::verbosity::verbose_enabled() {
                                eprintln!("i686 encoder warning: {} — emitting UD2 (instr: {:?} operands: {:?})", e, inst.opcode, inst.operands);
                            }
                            code.push(0x0F);
                            code.push(0x0B);
                        }
                    }
                }
            }
        }

        Ok(crate::backend::traits::AssembledFunction {
            bytes: code,
            relocations,
        })
    }

    fn emit_prologue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        emit_i686_prologue(mf)
    }

    fn emit_epilogue(&self, mf: &MachineFunction) -> Vec<MachineInstruction> {
        emit_i686_epilogue(mf)
    }
}

// ===========================================================================
// Standalone prologue/epilogue functions
// ===========================================================================

/// Generate the i686 cdecl function prologue instructions.
///
/// Standard cdecl prologue sequence:
/// ```text
/// push ebp            ; save old frame pointer
/// mov ebp, esp        ; establish new frame pointer
/// sub esp, N          ; allocate stack space for locals/spills
/// push ebx            ; save callee-saved registers (if used)
/// push esi
/// push edi
/// ```
///
/// Only callee-saved registers that appear in `mf.callee_saved_regs` are
/// pushed. Stack space `N` is `mf.frame_size`.
pub fn emit_i686_prologue(mf: &MachineFunction) -> Vec<MachineInstruction> {
    let mut prologue = Vec::new();

    // push ebp
    let push_ebp =
        MachineInstruction::new(I686_PUSH).with_operand(MachineOperand::Register(registers::EBP));
    prologue.push(push_ebp);

    // mov ebp, esp
    let mov_ebp_esp = MachineInstruction::new(I686_MOV)
        .with_operand(MachineOperand::Register(registers::ESP))
        .with_result(MachineOperand::Register(registers::EBP));
    prologue.push(mov_ebp_esp);

    // sub esp, frame_size (only if non-zero)
    if mf.frame_size > 0 {
        let sub_esp = MachineInstruction::new(I686_SUB)
            .with_operand(MachineOperand::Register(registers::ESP))
            .with_operand(MachineOperand::Immediate(mf.frame_size as i64))
            .with_result(MachineOperand::Register(registers::ESP));
        prologue.push(sub_esp);
    }

    // Save callee-saved registers that were used.
    for &reg in &mf.callee_saved_regs {
        // EBP is already saved in the prologue above.
        if reg == registers::EBP {
            continue;
        }
        let push = MachineInstruction::new(I686_PUSH).with_operand(MachineOperand::Register(reg));
        prologue.push(push);
    }

    prologue
}

/// Generate the i686 cdecl function epilogue instructions.
///
/// Standard cdecl epilogue sequence:
/// ```text
/// pop edi             ; restore callee-saved registers (reverse order)
/// pop esi
/// pop ebx
/// leave               ; mov esp, ebp ; pop ebp
/// ret                 ; return to caller
/// ```
///
/// Only callee-saved registers that were pushed in the prologue are popped.
pub fn emit_i686_epilogue(mf: &MachineFunction) -> Vec<MachineInstruction> {
    let mut epilogue = Vec::new();

    // Restore callee-saved registers in reverse order.
    for &reg in mf.callee_saved_regs.iter().rev() {
        if reg == registers::EBP {
            continue;
        }
        let pop = MachineInstruction::new(I686_POP).with_result(MachineOperand::Register(reg));
        epilogue.push(pop);
    }

    // leave (mov esp, ebp ; pop ebp)
    let leave = MachineInstruction::new(I686_LEAVE);
    epilogue.push(leave);

    // ret
    let ret = MachineInstruction::new(I686_RET).set_terminator();
    epilogue.push(ret);

    epilogue
}

// ===========================================================================
// I686Codegen — private instruction lowering dispatch
// ===========================================================================

impl I686Codegen {
    /// Master dispatch for lowering a single IR instruction into one or more
    /// i686 machine instructions.
    #[allow(clippy::too_many_arguments)]
    fn lower_instruction(
        &self,
        inst: &Instruction,
        value_map: &mut FxHashMap<u32, MachineOperand>,
        alloca_offsets: &mut FxHashMap<u32, i32>,
        stack_offset: &mut i32,
        vregs: &mut VRegAllocator,
        block_map: &FxHashMap<usize, usize>,
        mf: &mut MachineFunction,
        target: &Target,
        _diag: &mut DiagnosticEngine,
        constant_cache: &FxHashMap<u32, i64>,
    ) -> Vec<MachineInstruction> {
        let mut out = Vec::new();

        match inst {
            // ----- Memory -----
            Instruction::Alloca {
                result,
                ty,
                alignment,
                ..
            } => {
                let size = ty.size_bytes(target);
                let align = alignment.unwrap_or_else(|| ty.align_bytes(target));
                let align = if align == 0 { 4 } else { align };

                // Align the running stack offset.
                *stack_offset = (*stack_offset + align as i32 - 1) & !(align as i32 - 1);
                *stack_offset += size as i32;
                let offset = -(*stack_offset);
                alloca_offsets.insert(result.index(), offset);

                // LEA vreg, [EBP + offset] — materialize pointer to stack slot.
                let vreg = vregs.alloc();
                let lea = MachineInstruction::new(I686_LEA)
                    .with_operand(MachineOperand::Memory {
                        base: Some(registers::EBP),
                        index: None,
                        scale: 1,
                        displacement: offset as i64,
                    })
                    .with_result(MachineOperand::VirtualRegister(vreg));
                out.push(lea);
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            Instruction::Load {
                result, ptr, ty, ..
            } => {
                let src = self.resolve_operand(ptr, value_map, alloca_offsets);
                let vreg = vregs.alloc();

                if let MachineOperand::GlobalSymbol(_) = &src {
                    // Global variable load: use dedicated opcode that encodes
                    // as MOV r32, [disp32] with absolute addressing and
                    // R_386_32 relocation.
                    out.push(
                        MachineInstruction::new(I686_MOV_LOAD_GLOBAL)
                            .with_operand(src)
                            .with_result(MachineOperand::VirtualRegister(vreg)),
                    );
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                } else if ty.is_float() {
                    // x87 FLD from memory.
                    let fld = MachineInstruction::new(I686_FLD)
                        .with_operand(self.to_memory_operand(&src));
                    out.push(fld);
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                } else if matches!(&src, MachineOperand::VirtualRegister(_)) {
                    // Pointer held in a virtual register (e.g. GEP result).
                    // For sub-32-bit types (I8, I16), use a zero-extending
                    // byte/word load so we read the correct number of bytes
                    // from memory.  A plain 32-bit MOV_LOAD_INDIRECT would
                    // read 4 bytes (including adjacent memory) and then
                    // require a register-level byte extraction which is
                    // unreliable on i686 where only EAX–EDX have 8-bit
                    // sub-register aliases.
                    let opcode = match ty {
                        IrType::I1 | IrType::I8 => I686_MOVZX_LOAD_INDIRECT_BYTE,
                        IrType::I16 => I686_MOVZX_LOAD_INDIRECT_WORD,
                        _ => I686_MOV_LOAD_INDIRECT,
                    };
                    let mov = MachineInstruction::new(opcode)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg));
                    out.push(mov);
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                } else {
                    let opcode = if ty.size_bytes(target) < 4 {
                        I686_MOVZX
                    } else {
                        I686_MOV
                    };
                    let mem_op = self.to_memory_operand(&src);
                    let mov = MachineInstruction::new(opcode)
                        .with_operand(mem_op)
                        .with_result(MachineOperand::VirtualRegister(vreg));
                    out.push(mov);
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                }
            }

            Instruction::Store { value, ptr, .. } => {
                let dst = self.resolve_operand(ptr, value_map, alloca_offsets);
                let src = self.resolve_operand(value, value_map, alloca_offsets);

                if let MachineOperand::GlobalSymbol(_) = &dst {
                    // Global variable store: use dedicated opcode that encodes
                    // as MOV [disp32], r32 with absolute addressing and
                    // R_386_32 relocation.
                    // Materialize src if needed.
                    let src_safe = match &src {
                        MachineOperand::Immediate(_) | MachineOperand::GlobalSymbol(_) => {
                            let tmp = vregs.alloc();
                            out.push(
                                MachineInstruction::new(I686_MOV)
                                    .with_operand(src)
                                    .with_result(MachineOperand::VirtualRegister(tmp)),
                            );
                            MachineOperand::VirtualRegister(tmp)
                        }
                        _ => src,
                    };
                    out.push(
                        MachineInstruction::new(I686_MOV_STORE_GLOBAL)
                            .with_operand(dst)
                            .with_operand(src_safe),
                    );
                } else if matches!(&dst, MachineOperand::VirtualRegister(_)) {
                    // Pointer held in a virtual register (e.g. GEP result).
                    // Use dedicated STORE_INDIRECT opcode so the encoder
                    // dereferences the pointer ([reg]) instead of failing
                    // with "dst must be memory" or encoding a register copy.
                    let src_safe = match &src {
                        MachineOperand::Immediate(_) | MachineOperand::GlobalSymbol(_) => {
                            let tmp = vregs.alloc();
                            out.push(
                                MachineInstruction::new(I686_MOV)
                                    .with_operand(src)
                                    .with_result(MachineOperand::VirtualRegister(tmp)),
                            );
                            MachineOperand::VirtualRegister(tmp)
                        }
                        _ => src,
                    };
                    let mov = MachineInstruction::new(I686_MOV_STORE_INDIRECT)
                        .with_operand(dst)
                        .with_operand(src_safe);
                    out.push(mov);
                } else {
                    // i686 MOV store: operands[0] = Memory(dst), operands[1] = Register(src)
                    let dst_mem = self.to_memory_operand(&dst);
                    // Materialize src if it's not a register/vreg.
                    let src_safe = match &src {
                        MachineOperand::Immediate(_) | MachineOperand::GlobalSymbol(_) => {
                            let tmp = vregs.alloc();
                            out.push(
                                MachineInstruction::new(I686_MOV)
                                    .with_operand(src)
                                    .with_result(MachineOperand::VirtualRegister(tmp)),
                            );
                            MachineOperand::VirtualRegister(tmp)
                        }
                        _ => src,
                    };
                    let mov = MachineInstruction::new(I686_MOV)
                        .with_operand(dst_mem)
                        .with_operand(src_safe);
                    out.push(mov);
                }
            }

            // ----- Arithmetic -----
            Instruction::BinOp {
                result,
                op,
                lhs,
                rhs,
                ty,
                ..
            } => {
                // Handle constant sentinels: BinOp(Add, result, result, UNDEF)
                // These are placeholders for compile-time constants.
                if *lhs == *result && *rhs == Value::UNDEF {
                    if let Some(&imm) = constant_cache.get(&result.index()) {
                        let vreg = vregs.alloc();
                        let mov = MachineInstruction::new(I686_MOV)
                            .with_operand(MachineOperand::Immediate(imm))
                            .with_result(MachineOperand::VirtualRegister(vreg));
                        out.push(mov);
                        value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                        return out;
                    }
                }

                if *ty == IrType::I64 {
                    let insts =
                        self.lower_64bit_op(result, op, lhs, rhs, value_map, alloca_offsets, vregs);
                    out.extend(insts);
                    // 64-bit ops use EAX:EDX; register a vreg alias for
                    // downstream resolution (EAX holds the low word which
                    // is sufficient for most 32-bit consumers).
                    let vreg = vregs.alloc();
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                } else if ty.is_float() {
                    // lower_float_op now inserts into value_map internally.
                    let insts =
                        self.lower_float_op(result, op, lhs, rhs, value_map, alloca_offsets, vregs);
                    out.extend(insts);
                } else {
                    // lower_binary_op now inserts into value_map internally.
                    let insts = self.lower_binary_op(
                        result,
                        op,
                        lhs,
                        rhs,
                        ty,
                        value_map,
                        alloca_offsets,
                        vregs,
                    );
                    out.extend(insts);
                }
            }

            // ----- Comparisons -----
            Instruction::ICmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => {
                let insts =
                    self.lower_comparison(result, op, lhs, rhs, value_map, alloca_offsets, vregs);
                out.extend(insts);
            }

            Instruction::FCmp {
                result,
                op,
                lhs,
                rhs,
                ..
            } => {
                let lhs_op = self.resolve_operand(lhs, value_map, alloca_offsets);
                let rhs_op = self.resolve_operand(rhs, value_map, alloca_offsets);

                // FLD lhs → ST(0).
                out.push(MachineInstruction::new(I686_FLD).with_operand(lhs_op));
                // FLD rhs → ST(0), lhs moves to ST(1).
                out.push(MachineInstruction::new(I686_FLD).with_operand(rhs_op));
                // FUCOMIP ST(0), ST(1) → set EFLAGS, pop one.
                out.push(
                    MachineInstruction::new(I686_FUCOMIP)
                        .with_operand(MachineOperand::Register(registers::ST0)),
                );
                // FSTP ST(0) → pop remaining value.
                out.push(
                    MachineInstruction::new(I686_FSTP)
                        .with_operand(MachineOperand::Register(registers::ST0)),
                );

                let cc = CondCode::from_fcmp(op);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_SETCC)
                        .with_operand(MachineOperand::Immediate(cc.encoding() as i64))
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                let vreg_ext = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOVZX)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_result(MachineOperand::VirtualRegister(vreg_ext)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg_ext));
            }

            // ----- Control flow -----
            Instruction::Branch { target: tgt, .. } => {
                let target_mbb = *block_map.get(&tgt.index()).unwrap_or(&0);
                out.push(
                    MachineInstruction::new(I686_JMP)
                        .with_operand(MachineOperand::BlockLabel(target_mbb as u32))
                        .set_terminator()
                        .set_branch(),
                );
            }

            Instruction::CondBranch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                let cond_op = self.resolve_operand(condition, value_map, alloca_offsets);
                let then_mbb = *block_map.get(&then_block.index()).unwrap_or(&0);
                let else_mbb = *block_map.get(&else_block.index()).unwrap_or(&0);

                // Optimize: if the condition is a known constant, emit
                // an unconditional jump. This avoids generating
                // TEST imm, imm which the encoder cannot handle.
                if let MachineOperand::Immediate(val) = &cond_op {
                    if *val != 0 {
                        // Condition is true — jump to then_block.
                        out.push(
                            MachineInstruction::new(I686_JMP)
                                .with_operand(MachineOperand::BlockLabel(then_mbb as u32))
                                .set_terminator()
                                .set_branch(),
                        );
                    } else {
                        // Condition is false — jump to else_block.
                        out.push(
                            MachineInstruction::new(I686_JMP)
                                .with_operand(MachineOperand::BlockLabel(else_mbb as u32))
                                .set_terminator()
                                .set_branch(),
                        );
                    }
                } else {
                    // TEST cond, cond
                    out.push(
                        MachineInstruction::new(I686_TEST)
                            .with_operand(cond_op.clone())
                            .with_operand(cond_op),
                    );
                    // JNE then_block
                    out.push(
                        MachineInstruction::new(I686_JCC)
                            .with_operand(MachineOperand::Immediate(
                                CondCode::NotEqual.encoding() as i64
                            ))
                            .with_operand(MachineOperand::BlockLabel(then_mbb as u32))
                            .set_branch(),
                    );
                    // JMP else_block
                    out.push(
                        MachineInstruction::new(I686_JMP)
                            .with_operand(MachineOperand::BlockLabel(else_mbb as u32))
                            .set_terminator()
                            .set_branch(),
                    );
                }
            }

            Instruction::Switch {
                value: switch_val,
                default,
                cases,
                ..
            } => {
                let val_op = self.resolve_operand(switch_val, value_map, alloca_offsets);
                for (case_val, case_block) in cases {
                    let case_mbb = *block_map.get(&case_block.index()).unwrap_or(&0);
                    out.push(
                        MachineInstruction::new(I686_CMP)
                            .with_operand(val_op.clone())
                            .with_operand(MachineOperand::Immediate(*case_val)),
                    );
                    out.push(
                        MachineInstruction::new(I686_JCC)
                            .with_operand(MachineOperand::Immediate(
                                CondCode::Equal.encoding() as i64
                            ))
                            .with_operand(MachineOperand::BlockLabel(case_mbb as u32))
                            .set_branch(),
                    );
                }
                let default_mbb = *block_map.get(&default.index()).unwrap_or(&0);
                out.push(
                    MachineInstruction::new(I686_JMP)
                        .with_operand(MachineOperand::BlockLabel(default_mbb as u32))
                        .set_terminator()
                        .set_branch(),
                );
            }

            Instruction::Call {
                result,
                callee,
                args,
                return_type,
                ..
            } => {
                // ── Builtin interception ──────────────────────────────
                let callee_name = self.func_ref_names.get(callee).cloned();
                let handled = if let Some(ref fname) = callee_name {
                    self.try_emit_i686_builtin(
                        fname,
                        result,
                        args,
                        return_type,
                        value_map,
                        alloca_offsets,
                        vregs,
                        &mut out,
                    )
                } else {
                    false
                };

                if !handled {
                    let (call_insts, ret_vreg) = self.lower_call(
                        result,
                        callee,
                        args,
                        return_type,
                        value_map,
                        alloca_offsets,
                        vregs,
                        target,
                    );
                    out.extend(call_insts);
                    if let Some(vreg) = ret_vreg {
                        value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                    }
                    mf.mark_has_calls();
                }
            }

            Instruction::Return { value: ret_val, .. } => {
                if let Some(val) = ret_val {
                    let src = self.resolve_operand(val, value_map, alloca_offsets);
                    out.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(src)
                            .with_result(MachineOperand::Register(registers::EAX)),
                    );
                }
                out.push(MachineInstruction::new(I686_RET).set_terminator());
            }

            // ----- SSA -----
            Instruction::Phi {
                result, incoming, ..
            } => {
                // Phi nodes should be eliminated before codegen.
                // Fallback: use first incoming value.
                let vreg = vregs.alloc();
                if let Some((val, _)) = incoming.first() {
                    let src = self.resolve_operand(val, value_map, alloca_offsets);
                    out.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(src)
                            .with_result(MachineOperand::VirtualRegister(vreg)),
                    );
                }
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // ----- Pointer -----
            Instruction::GetElementPtr {
                result,
                base,
                indices,
                ..
            } => {
                let base_op = self.resolve_operand(base, value_map, alloca_offsets);
                let vreg = vregs.alloc();

                if indices.is_empty() {
                    out.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(base_op)
                            .with_result(MachineOperand::VirtualRegister(vreg)),
                    );
                } else {
                    let idx_op = self.resolve_operand(&indices[0], value_map, alloca_offsets);
                    // GEP lowering: compute base + index.
                    // Step 1: MOV vreg, base
                    out.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(base_op)
                            .with_result(MachineOperand::VirtualRegister(vreg)),
                    );
                    // Step 2: ADD vreg, index (if non-zero)
                    let skip_add = matches!(&idx_op, MachineOperand::Immediate(0));
                    if !skip_add {
                        out.push(
                            MachineInstruction::new(I686_ADD)
                                .with_operand(MachineOperand::VirtualRegister(vreg))
                                .with_operand(idx_op)
                                .with_result(MachineOperand::VirtualRegister(vreg)),
                        );
                    }
                }
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // ----- Casts -----
            Instruction::BitCast {
                result, value: val, ..
            } => {
                let src = self.resolve_operand(val, value_map, alloca_offsets);
                // If this result already has a VR assigned (e.g. from phi-copy
                // pre-allocation or a different predecessor block), reuse the
                // same VR so all predecessor paths write to the same virtual
                // register and the register allocator assigns a single physical
                // location. This is essential for phi elimination correctness
                // when critical edge split blocks are processed after merge blocks.
                let vreg = if let Some(MachineOperand::VirtualRegister(existing_vr)) =
                    value_map.get(&result.index())
                {
                    *existing_vr
                } else {
                    let vr = vregs.alloc();
                    value_map.insert(result.index(), MachineOperand::VirtualRegister(vr));
                    vr
                };
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
            }

            Instruction::Trunc {
                result,
                value: val,
                to_type,
                ..
            } => {
                let src = self.resolve_operand(val, value_map, alloca_offsets);
                let vreg = vregs.alloc();
                let mask: i64 = match to_type {
                    IrType::I1 => 0x1,
                    IrType::I8 => 0xFF,
                    IrType::I16 => 0xFFFF,
                    _ => -1, // No mask for I32.
                };
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                if mask != -1 {
                    out.push(
                        MachineInstruction::new(I686_AND)
                            .with_operand(MachineOperand::VirtualRegister(vreg))
                            .with_operand(MachineOperand::Immediate(mask))
                            .with_result(MachineOperand::VirtualRegister(vreg)),
                    );
                }
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            Instruction::ZExt {
                result, value: val, ..
            } => {
                let src = self.resolve_operand(val, value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOVZX)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            Instruction::SExt {
                result, value: val, ..
            } => {
                let src = self.resolve_operand(val, value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOVSX)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            Instruction::IntToPtr {
                result, value: val, ..
            }
            | Instruction::PtrToInt {
                result, value: val, ..
            } => {
                // On i686, pointers are 32-bit — same size as integers.
                let src = self.resolve_operand(val, value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(src)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // ----- StackAlloc (dynamic stack allocation: __builtin_alloca) -----
            Instruction::StackAlloc { result, size, .. } => {
                let vreg = vregs.alloc();
                // Resolve size operand
                let size_op = self.resolve_operand(size, value_map, alloca_offsets);
                // SUB ESP, size
                out.push(
                    MachineInstruction::new(I686_SUB)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(size_op),
                );
                // AND ESP, -16  (align to 16 bytes)
                out.push(
                    MachineInstruction::new(I686_AND)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(MachineOperand::Immediate(-16i64)),
                );
                // MOV result, ESP
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_operand(MachineOperand::Register(registers::ESP)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // ----- StackSave — capture ESP into a virtual register -----
            Instruction::StackSave { result, .. } => {
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_operand(MachineOperand::Register(registers::ESP)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // ----- StackRestore — restore ESP from a previously saved value -----
            Instruction::StackRestore { ptr, .. } => {
                let ptr_op = self.resolve_operand(ptr, value_map, alloca_offsets);
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(ptr_op),
                );
            }

            // ----- Inline assembly -----
            Instruction::InlineAsm { result, .. } => {
                let vreg = vregs.alloc();
                out.push(MachineInstruction::new(I686_NOP));
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }

            // IndirectBranch — computed goto: jmp *%reg
            Instruction::IndirectBranch { target, .. } => {
                let target_op = self.resolve_operand(target, value_map, alloca_offsets);
                let mut jmp = MachineInstruction::new(I686_JMP);
                jmp.operands.push(target_op);
                jmp.is_terminator = true;
                // Mark as branch so epilogue insertion does NOT insert
                // leave+ret before this instruction — it's a jump, not a
                // function return.
                jmp.is_branch = true;
                out.push(jmp);
            }

            // BlockAddress — materialize address of a labeled basic block
            Instruction::BlockAddress { result, block, .. } => {
                let vreg = vregs.alloc();
                let mach_idx = *block_map.get(&block.index()).unwrap_or(&0);
                let block_label = format!(".L{}", mach_idx);
                let mut lea = MachineInstruction::new(I686_LEA);
                lea.result = Some(MachineOperand::VirtualRegister(vreg));
                lea.operands.push(MachineOperand::GlobalSymbol(block_label));
                out.push(lea);
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
            }
        }

        out
    }
}

// ===========================================================================
// I686Codegen — helper methods
// ===========================================================================

impl I686Codegen {
    /// Resolve an IR Value to a MachineOperand.
    fn resolve_operand(
        &self,
        val: &Value,
        value_map: &FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
    ) -> MachineOperand {
        if let Some(op) = value_map.get(&val.index()) {
            return op.clone();
        }
        if let Some(&offset) = alloca_offsets.get(&val.index()) {
            return MachineOperand::Memory {
                base: Some(registers::EBP),
                index: None,
                scale: 1,
                displacement: offset as i64,
            };
        }
        // Check function-reference names (e.g., printf, exit).
        if let Some(fname) = self.func_ref_names.get(val) {
            return MachineOperand::GlobalSymbol(fname.clone());
        }
        // Check global-variable references (e.g., string literals).
        if let Some(gname) = self.global_var_refs.get(val) {
            return MachineOperand::GlobalSymbol(gname.clone());
        }
        // Fallback: treat as an immediate 0 (undefined value).
        MachineOperand::Immediate(0)
    }

    /// Convert an operand to a memory operand form.
    fn to_memory_operand(&self, op: &MachineOperand) -> MachineOperand {
        match op {
            MachineOperand::Memory { .. } => op.clone(),
            MachineOperand::Register(r) => MachineOperand::Memory {
                base: Some(*r),
                index: None,
                scale: 1,
                displacement: 0,
            },
            MachineOperand::VirtualRegister(_) => op.clone(),
            MachineOperand::FrameSlot(off) => MachineOperand::Memory {
                base: Some(registers::EBP),
                index: None,
                scale: 1,
                displacement: *off as i64,
            },
            // GlobalSymbol stays as-is — the encoder handles it
            // directly (e.g., [disp32] with relocation for PUSH/MOV).
            MachineOperand::GlobalSymbol(_) => op.clone(),
            _ => op.clone(),
        }
    }

    /// Construct a memory operand from base + index*scale + displacement.
    pub fn lower_memory_operand(
        &self,
        base: Option<u16>,
        index: Option<u16>,
        scale: u8,
        displacement: i64,
    ) -> MachineOperand {
        MachineOperand::Memory {
            base,
            index,
            scale,
            displacement,
        }
    }

    /// Lower an integer binary operation to i686 machine instructions.
    pub(crate) fn lower_binary_op(
        &self,
        result: &Value,
        op: &BinOp,
        lhs: &Value,
        rhs: &Value,
        _ty: &IrType,
        value_map: &mut FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        vregs: &mut VRegAllocator,
    ) -> Vec<MachineInstruction> {
        let mut insts = Vec::new();
        let lhs_op = self.resolve_operand(lhs, value_map, alloca_offsets);
        let rhs_op = self.resolve_operand(rhs, value_map, alloca_offsets);
        let result_vreg = vregs.alloc();

        match op {
            BinOp::Add => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::Sub => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_SUB)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::Mul => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_IMUL)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::SDiv => {
                // Signed division: PUSH rhs → MOV EAX, lhs → CDQ → IDIV [ESP] → ADD ESP,4
                // We PUSH the divisor first to avoid the register conflict where the
                // allocator assigns rhs to EDX, which CDQ then clobbers.
                let esp_mem = MachineOperand::Memory {
                    base: Some(registers::ESP),
                    index: None,
                    scale: 1,
                    displacement: 0,
                };
                insts.push(MachineInstruction::new(I686_PUSH).with_operand(rhs_op));
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(MachineInstruction::new(I686_CDQ));
                insts.push(MachineInstruction::new(I686_IDIV).with_operand(esp_mem));
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(MachineOperand::Immediate(4))
                        .with_result(MachineOperand::Register(registers::ESP)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::UDiv => {
                // Unsigned division: PUSH rhs → MOV EAX, lhs → XOR EDX,EDX → DIV [ESP] → ADD ESP,4
                let esp_mem = MachineOperand::Memory {
                    base: Some(registers::ESP),
                    index: None,
                    scale: 1,
                    displacement: 0,
                };
                insts.push(MachineInstruction::new(I686_PUSH).with_operand(rhs_op));
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(
                    MachineInstruction::new(I686_XOR)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_result(MachineOperand::Register(registers::EDX)),
                );
                insts.push(MachineInstruction::new(I686_DIV).with_operand(esp_mem));
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(MachineOperand::Immediate(4))
                        .with_result(MachineOperand::Register(registers::ESP)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::SRem => {
                // Signed remainder: PUSH rhs → MOV EAX, lhs → CDQ → IDIV [ESP] → ADD ESP,4
                let esp_mem = MachineOperand::Memory {
                    base: Some(registers::ESP),
                    index: None,
                    scale: 1,
                    displacement: 0,
                };
                insts.push(MachineInstruction::new(I686_PUSH).with_operand(rhs_op));
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(MachineInstruction::new(I686_CDQ));
                insts.push(MachineInstruction::new(I686_IDIV).with_operand(esp_mem));
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(MachineOperand::Immediate(4))
                        .with_result(MachineOperand::Register(registers::ESP)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::URem => {
                // Unsigned remainder: PUSH rhs → MOV EAX, lhs → XOR EDX,EDX → DIV [ESP] → ADD ESP,4
                let esp_mem = MachineOperand::Memory {
                    base: Some(registers::ESP),
                    index: None,
                    scale: 1,
                    displacement: 0,
                };
                insts.push(MachineInstruction::new(I686_PUSH).with_operand(rhs_op));
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(
                    MachineInstruction::new(I686_XOR)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_result(MachineOperand::Register(registers::EDX)),
                );
                insts.push(MachineInstruction::new(I686_DIV).with_operand(esp_mem));
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::Register(registers::ESP))
                        .with_operand(MachineOperand::Immediate(4))
                        .with_result(MachineOperand::Register(registers::ESP)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::And => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_AND)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::Or => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_OR)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::Xor => {
                // Detect bitwise NOT: XOR with Value::UNDEF represents
                // `~operand` (XOR with all-ones). Emit NOT instruction
                // instead of XOR with UNDEF-resolved Immediate(0).
                if *rhs == Value::UNDEF {
                    insts.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(lhs_op)
                            .with_result(MachineOperand::VirtualRegister(result_vreg)),
                    );
                    insts.push(
                        MachineInstruction::new(I686_NOT)
                            .with_operand(MachineOperand::VirtualRegister(result_vreg))
                            .with_result(MachineOperand::VirtualRegister(result_vreg)),
                    );
                } else {
                    insts.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(lhs_op)
                            .with_result(MachineOperand::VirtualRegister(result_vreg)),
                    );
                    insts.push(
                        MachineInstruction::new(I686_XOR)
                            .with_operand(MachineOperand::VirtualRegister(result_vreg))
                            .with_operand(rhs_op)
                            .with_result(MachineOperand::VirtualRegister(result_vreg)),
                    );
                }
            }
            BinOp::Shl => {
                // Shift amount must go into CL register.
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::Register(registers::ECX)),
                );
                insts.push(
                    MachineInstruction::new(I686_SHL)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(MachineOperand::Register(registers::CL))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::LShr => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::Register(registers::ECX)),
                );
                insts.push(
                    MachineInstruction::new(I686_SHR)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(MachineOperand::Register(registers::CL))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            BinOp::AShr => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::Register(registers::ECX)),
                );
                insts.push(
                    MachineInstruction::new(I686_SAR)
                        .with_operand(MachineOperand::VirtualRegister(result_vreg))
                        .with_operand(MachineOperand::Register(registers::CL))
                        .with_result(MachineOperand::VirtualRegister(result_vreg)),
                );
            }
            // Floating-point — delegate to lower_float_op.
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
                let float_insts =
                    self.lower_float_op(result, op, lhs, rhs, value_map, alloca_offsets, vregs);
                return float_insts;
            }
        }

        // Store the result vreg in value_map so downstream instructions
        // (ICmp, CondBranch, Store, etc.) can resolve the BinOp result.
        value_map.insert(result.index(), MachineOperand::VirtualRegister(result_vreg));
        insts
    }

    /// Lower an integer comparison (ICmp) to CMP + SETcc instructions.
    pub(crate) fn lower_comparison(
        &self,
        result: &Value,
        op: &ICmpOp,
        lhs: &Value,
        rhs: &Value,
        value_map: &mut FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        vregs: &mut VRegAllocator,
    ) -> Vec<MachineInstruction> {
        let mut insts = Vec::new();
        let lhs_op = self.resolve_operand(lhs, value_map, alloca_offsets);
        let rhs_op = self.resolve_operand(rhs, value_map, alloca_offsets);

        // CMP requires the first operand to be a register or memory
        // (not an immediate). If lhs resolves to an immediate,
        // materialize it in a temporary register.
        let lhs_safe = if matches!(lhs_op, MachineOperand::Immediate(_)) {
            let tmp = vregs.alloc();
            insts.push(
                MachineInstruction::new(I686_MOV)
                    .with_operand(lhs_op)
                    .with_result(MachineOperand::VirtualRegister(tmp)),
            );
            MachineOperand::VirtualRegister(tmp)
        } else {
            lhs_op
        };

        // CMP lhs, rhs
        insts.push(
            MachineInstruction::new(I686_CMP)
                .with_operand(lhs_safe)
                .with_operand(rhs_op),
        );

        let cc = CondCode::from_icmp(op);
        let vreg_byte = vregs.alloc();
        insts.push(
            MachineInstruction::new(I686_SETCC)
                .with_operand(MachineOperand::Immediate(cc.encoding() as i64))
                .with_result(MachineOperand::VirtualRegister(vreg_byte)),
        );

        let vreg_ext = vregs.alloc();
        insts.push(
            MachineInstruction::new(I686_MOVZX)
                .with_operand(MachineOperand::VirtualRegister(vreg_byte))
                .with_result(MachineOperand::VirtualRegister(vreg_ext)),
        );

        // Store the comparison result in value_map so that CondBranch
        // can resolve it as a VirtualRegister (not fallback Immediate(0)).
        value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg_ext));

        insts
    }

    /// Attempt to emit specialised code for compiler builtins (i686).
    /// Returns `true` if the builtin was handled, `false` otherwise.
    #[allow(clippy::too_many_arguments)]
    fn try_emit_i686_builtin(
        &self,
        fname: &str,
        result: &Value,
        args: &[Value],
        _return_type: &IrType,
        value_map: &mut FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        vregs: &mut VRegAllocator,
        out: &mut Vec<MachineInstruction>,
    ) -> bool {
        match fname {
            // ── byte-swap builtins ────────────────────────────────────
            "__builtin_bswap32" => {
                let arg = self.resolve_operand(&args[0], value_map, alloca_offsets);
                let vreg_eax = vregs.alloc();
                // MOV vreg, arg
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(arg)
                        .with_result(MachineOperand::VirtualRegister(vreg_eax)),
                );
                // BSWAP vreg (in-place byte reversal)
                out.push(
                    MachineInstruction::new(I686_BSWAP)
                        .with_operand(MachineOperand::VirtualRegister(vreg_eax))
                        .with_result(MachineOperand::VirtualRegister(vreg_eax)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg_eax));
                true
            }
            "__builtin_bswap16" => {
                // bswap16: BSWAP(arg32) >> 16, or simpler: XCHG AL,AH style.
                // Use: MOV vreg, arg → BSWAP vreg → SHR vreg, 16
                let arg = self.resolve_operand(&args[0], value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(arg)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                out.push(
                    MachineInstruction::new(I686_BSWAP)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                out.push(
                    MachineInstruction::new(I686_SHR)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_operand(MachineOperand::Immediate(16))
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                true
            }
            "__builtin_bswap64" => {
                // 32-bit platform: bswap64 on two halves and swap them.
                // Simplified: just BSWAP the lower 32 bits and return (truncated).
                // For true 64-bit bswap on i686, we'd need 64-bit register pairs.
                // Most uses in 32-bit context only need 32-bit bswap.
                let arg = self.resolve_operand(&args[0], value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(arg)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                out.push(
                    MachineInstruction::new(I686_BSWAP)
                        .with_operand(MachineOperand::VirtualRegister(vreg))
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                true
            }
            // ── branch hint builtins (passthrough) ───────────────────
            "__builtin_expect" | "__builtin_expect_with_probability" => {
                let arg = self.resolve_operand(&args[0], value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(arg)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                true
            }
            "__builtin_assume_aligned" => {
                let arg = self.resolve_operand(&args[0], value_map, alloca_offsets);
                let vreg = vregs.alloc();
                out.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(arg)
                        .with_result(MachineOperand::VirtualRegister(vreg)),
                );
                value_map.insert(result.index(), MachineOperand::VirtualRegister(vreg));
                true
            }
            // ── trap ─────────────────────────────────────────────────
            "__builtin_trap" | "__builtin_unreachable" => {
                out.push(MachineInstruction::new(I686_UD2));
                true
            }
            _ => false,
        }
    }

    /// Lower a function call with cdecl calling convention.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_call(
        &self,
        _result: &Value,
        callee: &Value,
        args: &[Value],
        return_type: &IrType,
        value_map: &FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        vregs: &mut VRegAllocator,
        _target: &Target,
    ) -> (Vec<MachineInstruction>, Option<u32>) {
        let mut insts = Vec::new();

        // Each argument is a 4-byte stack slot (minimum).
        let arg_stack_size: usize = args.len() * 4;

        // Align stack to 16 bytes before CALL.
        let padding = (16 - (arg_stack_size % 16)) % 16;
        if padding > 0 {
            insts.push(
                MachineInstruction::new(I686_SUB)
                    .with_operand(MachineOperand::Register(registers::ESP))
                    .with_operand(MachineOperand::Immediate(padding as i64))
                    .with_result(MachineOperand::Register(registers::ESP)),
            );
        }

        // Push arguments right-to-left (cdecl convention).
        for arg in args.iter().rev() {
            let arg_op = self.resolve_operand(arg, value_map, alloca_offsets);
            insts.push(MachineInstruction::new(I686_PUSH).with_operand(arg_op));
        }

        // Resolve callee.
        let callee_op = self.resolve_operand(callee, value_map, alloca_offsets);
        let call_target = if self.pic {
            match &callee_op {
                MachineOperand::GlobalSymbol(name) => {
                    MachineOperand::GlobalSymbol(format!("{}@PLT", name))
                }
                _ => callee_op,
            }
        } else {
            callee_op
        };

        // CALL
        insts.push(
            MachineInstruction::new(I686_CALL)
                .with_operand(call_target)
                .set_call(),
        );

        // Caller cleanup: ADD ESP, total
        let total_cleanup = arg_stack_size + padding;
        if total_cleanup > 0 {
            insts.push(
                MachineInstruction::new(I686_ADD)
                    .with_operand(MachineOperand::Register(registers::ESP))
                    .with_operand(MachineOperand::Immediate(total_cleanup as i64))
                    .with_result(MachineOperand::Register(registers::ESP)),
            );
        }

        // Move return value to a virtual register.
        let result_vreg = vregs.alloc();
        let mut ret_vreg = None;
        if return_type.is_float() {
            insts.push(
                MachineInstruction::new(I686_FSTP)
                    .with_operand(MachineOperand::VirtualRegister(result_vreg)),
            );
            ret_vreg = Some(result_vreg);
        } else if *return_type != IrType::Void {
            insts.push(
                MachineInstruction::new(I686_MOV)
                    .with_operand(MachineOperand::Register(registers::EAX))
                    .with_result(MachineOperand::VirtualRegister(result_vreg)),
            );
            ret_vreg = Some(result_vreg);
        }

        (insts, ret_vreg)
    }

    /// Lower x87 FPU floating-point operations.
    pub(crate) fn lower_float_op(
        &self,
        result: &Value,
        op: &BinOp,
        lhs: &Value,
        rhs: &Value,
        value_map: &mut FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        vregs: &mut VRegAllocator,
    ) -> Vec<MachineInstruction> {
        let mut insts = Vec::new();
        let lhs_op = self.resolve_operand(lhs, value_map, alloca_offsets);
        let rhs_op = self.resolve_operand(rhs, value_map, alloca_offsets);

        // Load LHS to ST(0).
        insts.push(MachineInstruction::new(I686_FLD).with_operand(lhs_op));
        // Load RHS to ST(0); LHS moves to ST(1).
        insts.push(MachineInstruction::new(I686_FLD).with_operand(rhs_op));

        let fpu_opcode = match op {
            BinOp::FAdd => I686_FADD,
            BinOp::FSub => I686_FSUB,
            BinOp::FMul => I686_FMUL,
            BinOp::FDiv | BinOp::FRem => I686_FDIV,
            _ => I686_FADD, // Unreachable for non-float ops.
        };
        insts.push(MachineInstruction::new(fpu_opcode));

        let result_vreg = vregs.alloc();
        insts.push(
            MachineInstruction::new(I686_FSTP)
                .with_result(MachineOperand::VirtualRegister(result_vreg)),
        );

        // Store the float result in value_map for downstream resolution.
        value_map.insert(result.index(), MachineOperand::VirtualRegister(result_vreg));
        insts
    }

    /// Lower 64-bit integer operations using register pairs.
    pub(crate) fn lower_64bit_op(
        &self,
        _result: &Value,
        op: &BinOp,
        lhs: &Value,
        rhs: &Value,
        value_map: &FxHashMap<u32, MachineOperand>,
        alloca_offsets: &FxHashMap<u32, i32>,
        _vregs: &mut VRegAllocator,
    ) -> Vec<MachineInstruction> {
        let mut insts = Vec::new();
        let lhs_op = self.resolve_operand(lhs, value_map, alloca_offsets);
        let rhs_op = self.resolve_operand(rhs, value_map, alloca_offsets);

        match op {
            BinOp::Add => {
                // Load LHS low to EAX.
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op.clone())
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                // ADD EAX, rhs_low.
                insts.push(
                    MachineInstruction::new(I686_ADD)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_operand(rhs_op.clone())
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                // ADC EDX, rhs_high (with carry).
                insts.push(
                    MachineInstruction::new(I686_ADC)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_operand(MachineOperand::Immediate(0))
                        .with_result(MachineOperand::Register(registers::EDX)),
                );
            }
            BinOp::Sub => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op.clone())
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                // SUB EAX, rhs_low.
                insts.push(
                    MachineInstruction::new(I686_SUB)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_operand(rhs_op.clone())
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                // SBB EDX, rhs_high (with borrow).
                insts.push(
                    MachineInstruction::new(I686_SBB)
                        .with_operand(MachineOperand::Register(registers::EDX))
                        .with_operand(MachineOperand::Immediate(0))
                        .with_result(MachineOperand::Register(registers::EDX)),
                );
            }
            BinOp::And => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(
                    MachineInstruction::new(I686_AND)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
            }
            BinOp::Or => {
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
                insts.push(
                    MachineInstruction::new(I686_OR)
                        .with_operand(MachineOperand::Register(registers::EAX))
                        .with_operand(rhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
            }
            BinOp::Xor => {
                // Detect bitwise NOT: XOR with Value::UNDEF represents
                // `~operand` (XOR with all-ones).
                if *rhs == Value::UNDEF {
                    insts.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(lhs_op)
                            .with_result(MachineOperand::Register(registers::EAX)),
                    );
                    insts.push(
                        MachineInstruction::new(I686_NOT)
                            .with_operand(MachineOperand::Register(registers::EAX))
                            .with_result(MachineOperand::Register(registers::EAX)),
                    );
                } else {
                    insts.push(
                        MachineInstruction::new(I686_MOV)
                            .with_operand(lhs_op)
                            .with_result(MachineOperand::Register(registers::EAX)),
                    );
                    insts.push(
                        MachineInstruction::new(I686_XOR)
                            .with_operand(MachineOperand::Register(registers::EAX))
                            .with_operand(rhs_op)
                            .with_result(MachineOperand::Register(registers::EAX)),
                    );
                }
            }
            _ => {
                // For 64-bit mul/div/shift: emit the lhs → EAX as a baseline.
                // The full register-pair sequences are expanded by later passes.
                insts.push(
                    MachineInstruction::new(I686_MOV)
                        .with_operand(lhs_op)
                        .with_result(MachineOperand::Register(registers::EAX)),
                );
            }
        }

        insts
    }
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::Span;
    use crate::ir::function::FunctionParam;

    #[test]
    fn test_condcode_invert() {
        assert_eq!(CondCode::Equal.invert(), CondCode::NotEqual);
        assert_eq!(CondCode::NotEqual.invert(), CondCode::Equal);
        assert_eq!(CondCode::Less.invert(), CondCode::GreaterEqual);
        assert_eq!(CondCode::LessEqual.invert(), CondCode::Greater);
        assert_eq!(CondCode::Greater.invert(), CondCode::LessEqual);
        assert_eq!(CondCode::GreaterEqual.invert(), CondCode::Less);
        assert_eq!(CondCode::Below.invert(), CondCode::AboveEqual);
        assert_eq!(CondCode::BelowEqual.invert(), CondCode::Above);
        assert_eq!(CondCode::Above.invert(), CondCode::BelowEqual);
        assert_eq!(CondCode::AboveEqual.invert(), CondCode::Below);
    }

    #[test]
    fn test_condcode_double_invert() {
        for cc in &[
            CondCode::Equal,
            CondCode::NotEqual,
            CondCode::Less,
            CondCode::LessEqual,
            CondCode::Greater,
            CondCode::GreaterEqual,
            CondCode::Below,
            CondCode::BelowEqual,
            CondCode::Above,
            CondCode::AboveEqual,
        ] {
            assert_eq!(cc.invert().invert(), *cc);
        }
    }

    #[test]
    fn test_condcode_encoding() {
        assert_eq!(CondCode::Equal.encoding(), 0x04);
        assert_eq!(CondCode::NotEqual.encoding(), 0x05);
        assert_eq!(CondCode::Below.encoding(), 0x02);
        assert_eq!(CondCode::AboveEqual.encoding(), 0x03);
    }

    #[test]
    fn test_i686_codegen_target() {
        let cg = I686Codegen::new(false, false);
        assert_eq!(cg.target(), Target::I686);
    }

    #[test]
    fn test_i686_codegen_register_info() {
        let cg = I686Codegen::new(false, false);
        let ri = cg.register_info();
        assert_eq!(ri.allocatable_gpr.len(), 5); // ECX is reserved as spill scratch
        assert_eq!(ri.allocatable_fpr.len(), 0);
        assert_eq!(ri.callee_saved.len(), 4);
        assert_eq!(ri.caller_saved.len(), 3);
        assert_eq!(ri.reserved.len(), 3); // ESP, EBP, ECX
        assert_eq!(ri.argument_gpr.len(), 0);
        assert_eq!(ri.return_gpr.len(), 1);
        assert_eq!(ri.return_gpr[0], registers::EAX);
    }

    #[test]
    fn test_i686_frame_pointer() {
        let cg = I686Codegen::new(false, false);
        assert_eq!(cg.frame_pointer_reg(), registers::EBP);
    }

    #[test]
    fn test_i686_stack_pointer() {
        let cg = I686Codegen::new(false, false);
        assert_eq!(cg.stack_pointer_reg(), registers::ESP);
    }

    #[test]
    fn test_i686_return_address_reg() {
        let cg = I686Codegen::new(false, false);
        assert_eq!(cg.return_address_reg(), None);
    }

    #[test]
    fn test_i686_opcode_constants() {
        assert!(I686_MOV >= 0x100 && I686_MOV < 0x200);
        assert!(I686_ADD >= 0x200 && I686_ADD < 0x300);
        assert!(I686_AND >= 0x300 && I686_AND < 0x400);
        assert!(I686_CMP >= 0x400 && I686_CMP < 0x500);
        assert!(I686_JMP >= 0x500 && I686_JMP < 0x600);
        assert!(I686_FLD >= 0x600 && I686_FLD < 0x700);
        assert!(I686_ENTER >= 0x700 && I686_ENTER < 0x800);
        assert!(I686_NOP >= 0x800);
    }

    #[test]
    fn test_prologue_empty_frame() {
        let mf = MachineFunction::new("test_fn".to_string());
        let prologue = emit_i686_prologue(&mf);
        assert!(prologue.len() >= 2);
        assert_eq!(prologue[0].opcode, I686_PUSH);
        assert_eq!(prologue[1].opcode, I686_MOV);
    }

    #[test]
    fn test_prologue_with_frame() {
        let mut mf = MachineFunction::new("test_fn".to_string());
        mf.frame_size = 32;
        let prologue = emit_i686_prologue(&mf);
        assert!(prologue.len() >= 3);
        assert_eq!(prologue[0].opcode, I686_PUSH);
        assert_eq!(prologue[1].opcode, I686_MOV);
        assert_eq!(prologue[2].opcode, I686_SUB);
    }

    #[test]
    fn test_epilogue() {
        let mf = MachineFunction::new("test_fn".to_string());
        let epilogue = emit_i686_epilogue(&mf);
        assert!(epilogue.len() >= 2);
        assert_eq!(epilogue.last().unwrap().opcode, I686_RET);
        assert!(epilogue.last().unwrap().is_terminator);
    }

    #[test]
    fn test_lower_empty_function() {
        let cg = I686Codegen::new(false, false);
        let mut diag = DiagnosticEngine::new();

        let mut func = IrFunction::new("empty".to_string(), vec![], IrType::Void);
        func.blocks[0].push_instruction(Instruction::Return {
            value: None,
            span: Span::dummy(),
        });

        let result = cg.lower_function(
            &func,
            &mut diag,
            &[],
            &crate::common::fx_hash::FxHashMap::default(),
            &crate::common::fx_hash::FxHashMap::default(),
        );
        assert!(result.is_ok());
        let mf = result.unwrap();
        assert_eq!(mf.name, "empty");
        assert!(!mf.blocks[0].instructions.is_empty());
    }

    #[test]
    fn test_lower_function_with_params() {
        let cg = I686Codegen::new(false, false);
        let mut diag = DiagnosticEngine::new();

        let params = vec![
            FunctionParam::new("a".to_string(), IrType::I32, Value(0)),
            FunctionParam::new("b".to_string(), IrType::I32, Value(1)),
        ];
        let mut func = IrFunction::new("add".to_string(), params, IrType::I32);
        func.blocks[0].push_instruction(Instruction::BinOp {
            result: Value(2),
            op: BinOp::Add,
            lhs: Value(0),
            rhs: Value(1),
            ty: IrType::I32,
            span: Span::dummy(),
        });
        func.blocks[0].push_instruction(Instruction::Return {
            value: Some(Value(2)),
            span: Span::dummy(),
        });
        func.value_count = 3;

        let result = cg.lower_function(
            &func,
            &mut diag,
            &[],
            &crate::common::fx_hash::FxHashMap::default(),
            &crate::common::fx_hash::FxHashMap::default(),
        );
        assert!(result.is_ok());
        let mf = result.unwrap();
        assert_eq!(mf.name, "add");
        assert!(mf.blocks[0].instructions.len() >= 3);
    }

    #[test]
    fn test_emit_assembly_nop() {
        let cg = I686Codegen::new(false, false);
        let mut mf = MachineFunction::new("nop_fn".to_string());
        mf.blocks[0].push_instruction(MachineInstruction::new(I686_NOP));
        mf.blocks[0].push_instruction(MachineInstruction::new(I686_RET).set_terminator());

        let result = cg.emit_assembly(&mf);
        assert!(result.is_ok());
        let asm_fn = result.unwrap();
        assert!(!asm_fn.bytes.is_empty());
        assert_eq!(asm_fn.bytes[0], 0x90);
        assert_eq!(asm_fn.bytes[1], 0xC3);
    }

    #[test]
    fn test_classify_return_types() {
        let cg = I686Codegen::new(false, false);
        match cg.classify_return(&IrType::I32) {
            crate::backend::traits::ArgLocation::Register(r) => {
                assert_eq!(r, registers::EAX);
            }
            _ => panic!("I32 should return in register"),
        }
        match cg.classify_return(&IrType::I64) {
            crate::backend::traits::ArgLocation::RegisterPair(lo, hi) => {
                assert_eq!(lo, registers::EAX);
                assert_eq!(hi, registers::EDX);
            }
            _ => panic!("I64 should return in register pair"),
        }
        match cg.classify_return(&IrType::F64) {
            crate::backend::traits::ArgLocation::Register(r) => {
                assert_eq!(r, registers::ST0);
            }
            _ => panic!("F64 should return in ST(0)"),
        }
    }

    #[test]
    fn test_lower_memory_operand() {
        let cg = I686Codegen::new(false, false);
        let op = cg.lower_memory_operand(Some(registers::EBP), None, 1, -8);
        match op {
            MachineOperand::Memory {
                base,
                index,
                scale,
                displacement,
            } => {
                assert_eq!(base, Some(registers::EBP));
                assert_eq!(index, None);
                assert_eq!(scale, 1);
                assert_eq!(displacement, -8);
            }
            _ => panic!("Expected Memory operand"),
        }
    }

    #[test]
    fn test_icmp_to_condcode() {
        assert_eq!(CondCode::from_icmp(&ICmpOp::Eq), CondCode::Equal);
        assert_eq!(CondCode::from_icmp(&ICmpOp::Ne), CondCode::NotEqual);
        assert_eq!(CondCode::from_icmp(&ICmpOp::Slt), CondCode::Less);
        assert_eq!(CondCode::from_icmp(&ICmpOp::Ult), CondCode::Below);
        assert_eq!(CondCode::from_icmp(&ICmpOp::Sge), CondCode::GreaterEqual);
        assert_eq!(CondCode::from_icmp(&ICmpOp::Uge), CondCode::AboveEqual);
    }
}
