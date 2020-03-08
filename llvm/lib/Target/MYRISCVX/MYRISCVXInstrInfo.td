//===- MYRISCVXInstrInfo.td - Target Description for MYRISCVX Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

//@{MYRISCVXInstrInfo_MYRISCVXRet
// Return
def MYRISCVXRet : SDNode<"MYRISCVXISD::Ret", SDTNone,
                         [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
//@}MYRISCVXInstrInfo_MYRISCVXRet

//===----------------------------------------------------------------------===//
// MYRISCVX profiles and nodes
//===----------------------------------------------------------------------===//

//@{MYRISCVXInstrInfo_Include_td
//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
include "MYRISCVXInstrFormats.td"
//@}MYRISCVXInstrInfo_Include_td

//@{MYRISCVXInstrInfo_Operand_Definition
//===----------------------------------------------------------------------===//
// MYRISCVX Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//
def simm12 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<12>(Imm);}]>;
def simm20u : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<32>(Imm) && !(Imm & 0x0fff);}]>;
def simm32: Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<32>(Imm);}]>;
//@}MYRISCVXInstrInfo_Operand_Definition


// @{MYRISCVXInstrInfo_LO12_HI20
// Transformation Function - get the lower 12 bits.
def LO12sext : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(SignExtend64<12>(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Transformation Function - get the higher 20 bits.
def HI20 : SDNodeXForm<imm, [{
  return getImm(N, ((N->getZExtValue()+0x800) >> 12) & 0xfffff);
}]>;
// @}MYRISCVXInstrInfo_LO12_HI20


// shamt field must fit in 5 bits.
def immZExt5 : ImmLeaf<XLenVT, [{return Imm == (Imm & 0x1f);}]>;
def shamt : Operand<i32>;

def calltarget : Operand<iPTR> {
  let EncoderMethod = "getJumpTargetOpValue";
}

// JAL
def brtarget20    : Operand<OtherVT> {
  let EncoderMethod = "getBranch20TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranch20Target";
}


// BEQ, BNE
def brtarget12    : Operand<OtherVT> {
  let EncoderMethod = "getBranch12TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}


// MYRISCVX Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr_fi : ComplexPattern<iPTR, 1, "SelectAddrFI", [frameindex], []>;


//@{MYRISCVXInstrInfo_Instruction_Format
//@{MYRISCVXInstrInfo_ArithLogicR
//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
// Arithmetic and logical instructions with 2 register operands.
class ArithLogicR<bits<7> opcode, bits<3> funct3, bits<7>funct7,
                  string instr_asm, SDNode OpNode,
                  RegisterClass RC> :
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
  !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
  [(set RC:$rd, (OpNode RC:$rs1, RC:$rs2))], IIAlu> {
    let isReMaterializable = 1;
}
//@}MYRISCVXInstrInfo_ArithLogicR


//@{MYRISCVXInstrInfo_ArithLogicI
// Arithmetic and logical instructions with 2 register operands.
class ArithLogicI<bits<7> opcode, bits<3> funct3,
                  string instr_asm, SDNode OpNode,
                  Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
  !strconcat(instr_asm, "\t$rd, $rs1, $simm12"),
  [(set RC:$rd, (OpNode RC:$rs1, Od:$simm12))], IIAlu> {
    let isReMaterializable = 1;
}
//@}MYRISCVXInstrInfo_ArithLogicI

// Arithmetic and logical instructions with 2 register and one immediate.
class ArithLogicU<bits<7> opcode,
                  string instr_asm, RegisterClass RC, Operand Od> :
  MYRISCVX_U<opcode, (outs RC:$rd), (ins Od:$imm20),
    !strconcat(instr_asm, "\t$rd, $imm20"), [], IIAlu> {
      let isReMaterializable = 1;
}
//@}MYRISCVXInstrInfo_Instruction_Format


//@{MYRISCVXInstrInfo_ShiftRotate
// Shifts
class ShiftRotateI<bits<7> opcode, bits<3> funct3, bit arithshift, string instr_asm,
                   SDNode OpNode, PatFrag PF, Operand ImmOpnd,
                   RegisterClass RC>:
  MYRISCVX_ISHIFT<opcode, funct3, arithshift, (outs RC:$rd), (ins RC:$rs1, ImmOpnd:$shamt),
                  !strconcat(instr_asm, "\t$rd, $rs1, $shamt"),
                  [(set GPR:$rd, (OpNode RC:$rs1, PF:$shamt))], IIAlu> {
}


// 32-bit shift instructions.
class ShiftRotateI32<bits<7> opcode, bits<3> funct3, bit arithshift, string instr_asm,
                     SDNode OpNode>:
  ShiftRotateI<opcode, funct3, arithshift, instr_asm, OpNode, immZExt5, shamt, GPR>;


class ShiftRotateR<bits<7> opcode, bits<7> funct7, bits<3> funct3,
                   bits<4> isRotate, string instr_asm,
                   SDNode OpNode, RegisterClass RC>:
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
          !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
          [(set GPR:$rd, (OpNode RC:$rs1, RC:$rs2))], IIAlu> {
}
//@}MYRISCVXInstrInfo_ShiftRotate


//@{MYRISCVXInstrInfo_jalr_variation
def JALR : ArithLogicI<0b1100111, 0b000, "jalr", add, simm12, GPR>;
def: InstAlias<"jr $rs",      (JALR ZERO, GPR:$rs, 0)>;
def: InstAlias<"jalr $rs",    (JALR RA,   GPR:$rs, 0)>;
def: InstAlias<"ret",         (JALR ZERO,      RA, 0), 2>;
//@}MYRISCVXInstrInfo_jalr_variation


//@{MYRISCVXInstrInfo_LoadMemory_StoreMemory
// Memory Load/Store
let canFoldAsLoad = 1 in
class LoadMemory<bits<7> opcode, bits<3> funct3, string instr_asm, RegisterClass RC, PatFrag OpNode,
                 bit Pseudo>:
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins GPR:$rs1, simm12:$simm12),
     !strconcat(instr_asm, "\t$rd, ${simm12}(${rs1})"),
     [], IILoad> {
  let isPseudo = Pseudo;
}

class StoreMemory<bits<7> opcode, bits<3> funct3, string instr_asm, RegisterClass RC, PatFrag OpNode,
                  bit Pseudo>:
  MYRISCVX_S<opcode, funct3, (outs), (ins RC:$rs2, GPR:$rs1, simm12:$simm12),
     !strconcat(instr_asm, "\t$rs2, ${simm12}(${rs1})"),
     [], IIStore> {
  let isPseudo = Pseudo;
}
//@}MYRISCVXInstrInfo_LoadMemory_StoreMemory

//===----------------------------------------------------------------------===//
// Pattern fragment for load/store
//===----------------------------------------------------------------------===//

//@{MYRISCVXInstrInfo_AlignedLoad_AlignedStore
class AlignedLoad<PatFrag Node> :
  PatFrag<(ops node:$ptr), (Node node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getMemoryVT().getSizeInBits()/8 <= LD->getAlignment();
}]>;

class AlignedStore<PatFrag Node> :
  PatFrag<(ops node:$val, node:$ptr), (Node node:$val, node:$ptr), [{
  StoreSDNode *SD = cast<StoreSDNode>(N);
  return SD->getMemoryVT().getSizeInBits()/8 <= SD->getAlignment();
}]>;
//@}MYRISCVXInstrInfo_AlignedLoad_AlignedStore

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
//@{MYRISCVXInstrInfo_Basic_Definition
//===----------------------------------------------------------------------===//
// MYRISCVX Instructions
//===----------------------------------------------------------------------===//
//@{MYRISCVXInstrInfo_Arithmetic
//@{MYRISCVXInstrInfo_Arithmetic_ADDI
def ADDI : ArithLogicI<0b0010011, 0b000, "addi", add, simm12, GPR>;
//@}MYRISCVXInstrInfo_Arithmetic_ADDI
def XORI : ArithLogicI<0b0010011, 0b100, "xori", xor, simm12, GPR>;
def ORI  : ArithLogicI<0b0010011, 0b110, "ori",  or,  simm12, GPR>;
def ANDI : ArithLogicI<0b0010011, 0b111, "andi", and, simm12, GPR>;

def LUI  : ArithLogicU<0b0110111, "lui", GPR, simm20u>;
def ADD  : ArithLogicR<0b0110011, 0b000, 0b0000000, "add", add, GPR>;
def SUB  : ArithLogicR<0b0110011, 0b000, 0b0100000, "sub", sub, GPR>;
//@}MYRISCVXInstrInfo_Basic_Definition

def AND  : ArithLogicR     <0b0110011, 0b111, 0b0000000, "and", and, GPR>;
def OR   : ArithLogicR     <0b0110011, 0b110, 0b0000000, "or",  or,  GPR>;
def XOR  : ArithLogicR     <0b0110011, 0b100, 0b0000000, "xor", xor, GPR>;
//@}MYRISCVXInstrInfo_Arithmetic

// @{MYRISCVXInstrInfo_ShiftInsts
def SLL  : ShiftRotateR<0b0110011, 0b0000000, 0b001, 0x0, "sll", shl, GPR>;
def SRL  : ShiftRotateR<0b0110011, 0b0000000, 0b101, 0x0, "srl", srl, GPR>;
def SRA  : ShiftRotateR<0b0110011, 0b0100000, 0b101, 0x0, "sra", sra, GPR>;

def SRLI : ShiftRotateI32 <0b0010011, 0b101, 0, "srli", srl>;
def SLLI : ShiftRotateI32 <0b0010011, 0b001, 0, "slli", shl>;
def SRAI : ShiftRotateI32 <0b0010011, 0b101, 1, "srai", sra>;
// @}MYRISCVXInstrInfo_ShiftInsts

// @{MYRISCVXInstrInfo_MulDivInsts
def MUL   : ArithLogicR<0b0110011, 0b000, 0b0000001, "mul",    mul,   GPR>;
def MULH  : ArithLogicR<0b0110011, 0b001, 0b0000001, "mulh",   mulhs, GPR>;
def MULHSU: ArithLogicR<0b0110011, 0b010, 0b0000001, "mulhsu", mulhs, GPR>;
def MULHU : ArithLogicR<0b0110011, 0b011, 0b0000001, "mulhu",  mulhu, GPR>;
def DIV   : ArithLogicR<0b0110011, 0b100, 0b0000001, "div",    sdiv,  GPR>;
def DIVU  : ArithLogicR<0b0110011, 0b101, 0b0000001, "divu",   udiv,  GPR>;
def REM   : ArithLogicR<0b0110011, 0b110, 0b0000001, "rem",    srem,  GPR>;
def REMU  : ArithLogicR<0b0110011, 0b111, 0b0000001, "remu",   urem,  GPR>;
// @}MYRISCVXInstrInfo_MulDivInsts


//@{MYRISCVXInstrInfo_RetRA
//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//
let isReturn=1, isTerminator=1, hasDelaySlot=0, isBarrier=1, hasCtrlDep=1 in
  def RetRA : MYRISCVXPseudo<(outs), (ins), "", [(MYRISCVXRet)]>;
//@}MYRISCVXInstrInfo_RetRA


//@{MYRISCVXInstrInfo_LoadStore_PatFrags
// Load/Store PatFrags.
def load_a          : AlignedLoad<load>;
def sextloadi8_a    : AlignedLoad<sextloadi8>;
def zextloadi8_a    : AlignedLoad<zextloadi8>;
def sextloadi16_a   : AlignedLoad<sextloadi16>;
def zextloadi16_a   : AlignedLoad<zextloadi16>;
def extloadi16_a    : AlignedLoad<extloadi16>;
def sextloadi32_a   : AlignedLoad<sextloadi32>;
def zextloadi32_a   : AlignedLoad<zextloadi32>;
def extloadi32_a    : AlignedLoad<extloadi32>;

def store_a         : AlignedStore<store>;
def truncstorei8_a  : AlignedStore<truncstorei8>;
def truncstorei16_a : AlignedStore<truncstorei16>;
def truncstorei32_a : AlignedStore<truncstorei32>;

def truncstoref32_a : AlignedStore<truncstoref32>;
def truncstoref64_a : AlignedStore<truncstoref64>;
//@}MYRISCVXInstrInfo_LoadStore_PatFrags


// @{MYRISCVXInstrInfo_td_MemoryInsts
/// Aligned Load and Store Instructions
def LW  : LoadMemory <0b0000011, 0b010, "lw",  GPR, sextloadi32_a  , 0>;
def SW  : StoreMemory<0b0100011, 0b010, "sw",  GPR, truncstorei32_a, 0>;
def LB  : LoadMemory <0b0000011, 0b000, "lb",  GPR, sextloadi8_a   , 0>;
def LBU : LoadMemory <0b0000011, 0b100, "lbu", GPR, zextloadi8_a   , 0>;
def SB  : StoreMemory<0b0100011, 0b000, "sb",  GPR, truncstorei8_a , 0>;
def LH  : LoadMemory <0b0000011, 0b001, "lh",  GPR, sextloadi16_a  , 0>;
def LHU : LoadMemory <0b0000011, 0b101, "lhu", GPR, zextloadi16_a  , 0>;
def SH  : StoreMemory<0b0100011, 0b001, "sh",  GPR, truncstorei16_a, 0>;
// @}MYRISCVXInstrInfo_td_MemoryInsts


// Small immediates

// @{MYRISCVXInstrInfo_Pat_simm12
def : Pat<(simm12:$in), (ADDI ZERO, simm12:$in)>;
// @}MYRISCVXInstrInfo_Pat_simm12

// @{MYRISCVXInstrInfo_Pat_simm20u
def : Pat<(simm20u:$in), (LUI (HI20 imm:$in))>;
// @}MYRISCVXInstrInfo_Pat_simm20u
// @{MYRISCVXInstrInfo_Pat_simm32
def : Pat<(simm32:$imm),
          (ADDI (LUI (HI20 imm:$imm)), (LO12sext imm:$imm))>;
// @}MYRISCVXInstrInfo_Pat_simm32


// Addr for Frame
def : Pat<(add (XLenVT addr_fi:$rs1), simm12:$simm12),
          (ADDI (XLenVT addr_fi:$rs1), simm12:$simm12)>;

// @{MYRISCVXInstrInfo_td_MemoryPattern_Define
// Memory Acccess
multiclass LoadPattern<PatFrag LoadOp, MYRISCVXInst Inst> {
  def : Pat<(LoadOp GPR:$rs1)                          , (Inst GPR:$rs1, 0)                  >;
  def : Pat<(LoadOp addr_fi:$rs1)                      , (Inst addr_fi:$rs1, 0)              >;
  def : Pat<(LoadOp (add GPR:$rs1, simm12:$simm12))    , (Inst GPR:$rs1, simm12:$simm12)     >;
  def : Pat<(LoadOp (add addr_fi:$rs1, simm12:$simm12)), (Inst addr_fi:$rs1, simm12:$simm12) >;
}

multiclass StorePattern<PatFrag StoreOp, MYRISCVXInst Inst> {
  def : Pat<(StoreOp GPR:$rs2, GPR:$rs1)                          , (Inst GPR:$rs2, GPR:$rs1, 0)                 >;
  def : Pat<(StoreOp GPR:$rs2, addr_fi:$rs1)                      , (Inst GPR:$rs2, addr_fi:$rs1, 0)             >;
  def : Pat<(StoreOp GPR:$rs2, (add GPR:$rs1, simm12:$simm12))    , (Inst GPR:$rs2, GPR:$rs1, simm12:$simm12)    >;
  def : Pat<(StoreOp GPR:$rs2, (add addr_fi:$rs1, simm12:$simm12)), (Inst GPR:$rs2, addr_fi:$rs1, simm12:$simm12)>;
}
// @}MYRISCVXInstrInfo_td_MemoryPattern_Define

// @{MYRISCVXInstrInfo_td_MemoryPattern_Impl
defm : LoadPattern<sextloadi8  , LB>;
defm : LoadPattern<extloadi8   , LB>;
defm : LoadPattern<sextloadi16 , LH>;
defm : LoadPattern<extloadi16  , LH>;
defm : LoadPattern<load        , LW>, Requires<[IsRV32]>;
defm : LoadPattern<zextloadi8  , LBU>;
defm : LoadPattern<zextloadi16 , LHU>;

defm : StorePattern<truncstorei8  , SB>;
defm : StorePattern<truncstorei16 , SH>;
defm : StorePattern<store         , SW>, Requires<[IsRV32]>;
// @}MYRISCVXInstrInfo_td_MemoryPattern_Impl


// @{MYRISCVXInstrInfo_td_MemoryPattern64
let Predicates = [IsRV64] in {
  def LD  : LoadMemory <0b0000011, 0b011, "ld",  GPR, load_a         , 0>;
  def SD  : StoreMemory<0b0100011, 0b011, "sd",  GPR, store_a        , 0>;
  def LWU : LoadMemory <0b0000011, 0b110, "lwu",  GPR, sextloadi32_a  , 0>;

  /// Loads
  defm : LoadPattern<sextloadi32 , LW>;
  defm : LoadPattern<extloadi32  , LW>;
  defm : LoadPattern<zextloadi32 , LWU>;
  defm : LoadPattern<load        , LD>;

  /// Stores
  defm : StorePattern<truncstorei32 , SW>;
  defm : StorePattern<store         , SD>;
} // Predicates = [IsRV64]
// @}MYRISCVXInstrInfo_td_MemoryPattern64
