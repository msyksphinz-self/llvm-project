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


//@{MYRISCVXInstrInfo_brtarget20
def brtarget20    : Operand<OtherVT> {
  let EncoderMethod = "getBranch20TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranch20Target";
}
//@}MYRISCVXInstrInfo_brtarget20


//@{MYRISCVXInstrInfo_brtarget12
// BEQ, BNE
def brtarget12    : Operand<OtherVT> {
  let EncoderMethod = "getBranch12TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}
//@}MYRISCVXInstrInfo_brtarget12


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


//@{MYRISCVXInstrInfo_MYRISCVXHiLo
// Hi and Lo nodes are used to handle global addresses. Used on
// MYRISCVXISelLowering to lower stuff like GlobalAddress, ExternalSymbol
// static model. (nothing to do with MYRISCVX Registers Hi and Lo)
def MYRISCVXHi    : SDNode<"MYRISCVXISD::Hi", SDTIntUnaryOp>;
def MYRISCVXLo    : SDNode<"MYRISCVXISD::Lo", SDTIntUnaryOp>;
//@}MYRISCVXInstrInfo_MYRISCVXHiLo
//@{MYRISCVXInstrInfo_MYRISCVXGPRel
def MYRISCVXGPRel : SDNode<"MYRISCVXISD::GPRel", SDTIntUnaryOp>;
//@}MYRISCVXInstrInfo_MYRISCVXGPRel

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
class ArithLogicI_JALR<bits<7> opcode, bits<3> funct3,
                       string instr_asm,
                       Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
  !strconcat(instr_asm, "\t$rd, $rs1, $simm12"),
  [], IIAlu> {
    let isReMaterializable = 1;
}
def JALR : ArithLogicI_JALR<0b1100111, 0b000, "jalr", simm12, GPR>;
def: InstAlias<"jr $rs",      (JALR ZERO, GPR:$rs, 0)>;
def: InstAlias<"jalr $rs",    (JALR RA,   GPR:$rs, 0)>;
def: InstAlias<"ret",         (JALR ZERO,      RA, 0), 2>;
//@}MYRISCVXInstrInfo_jalr_variation


//@{MYRISCVXInstrInfo_JumpLink
// Jump and Link (Call)
let isCall = 1 in
class JumpLink<bits<7> opcode, string opstr, DAGOperand opnd> :
  MYRISCVX_J<opcode, (outs GPR:$rd), (ins opnd:$imm20), !strconcat(opstr, "\t$rd, $imm20"),
                 [], IIAlu> {
  let DecoderMethod = "DecodeJumpTarget";
}
//@}MYRISCVXInstrInfo_JumpLink

//@{MYRISCVXInstrInfo_JAL
def JAL : JumpLink<0b1101111, "jal", brtarget20>;
def : InstAlias<"j $offset",   (JAL  ZERO, brtarget20:$offset)>;
def : InstAlias<"jal $offset", (JAL  RA,   brtarget20:$offset)>;
//@}MYRISCVXInstrInfo_JAL

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


//@{MYRISCVXInstrInfo_MYRISCVXWrapper
def MYRISCVXWrapper : SDNode<"MYRISCVXISD::Wrapper", SDTIntBinOp>;
//@}MYRISCVXInstrInfo_MYRISCVXWrapper

def RelocPIC : Predicate<"TM.getRelocationModel() == Reloc::PIC_">;

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


//@{MYRISCVXInstrInfo_SetCC
// SetCC
class SetCC_R<bits<7> opcode, bits<3> funct3, bits<7> funct7,
              string instr_asm, PatFrag cond_op,
              RegisterClass RC> :
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$ra), (ins RC:$rb, RC:$rc),
             !strconcat(instr_asm, "\t$ra, $rb, $rc"),
             [(set GPR:$ra, (cond_op RC:$rb, RC:$rc))], IIAlu> {
               let isReMaterializable = 1;
}


class SetCC_I<bits<7> opcode, bits<3> funct3,
              string instr_asm, PatFrag cond_op,
              Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$ra), (ins RC:$rb, Od:$simm12),
             !strconcat(instr_asm, "\t$ra, $rb, $simm12"),
             [(set GPR:$ra, (cond_op RC:$rb, Od:$simm12))], IIAlu> {
               let isReMaterializable = 1;
}
//@}MYRISCVXInstrInfo_SetCC


//@{MYRISCVXInstrInfo_CBranch12
// Branch instructions with 2 register operands.
class CBranch12<bits<7> opcode, bits<3> funct3,
                string instr_asm,
                PatFrag cond_op, RegisterClass RC> :
  MYRISCVX_B<opcode, funct3, (outs), (ins RC:$rs1, RC:$rs2, brtarget12:$imm12),
  !strconcat(instr_asm, "\t$rs1, $rs2, $imm12"),
  [], IIAlu> {
    let isBranch = 1;
    let isTerminator = 1;
}
//@}MYRISCVXInstrInfo_CBranch12


//@{MYRISCVXInstrInfo_BEQ
def BEQ     : CBranch12<0b1100011, 0b000, "beq" , seteq,  GPR>;
def BNE     : CBranch12<0b1100011, 0b001, "bne" , setne,  GPR>;
def BLT     : CBranch12<0b1100011, 0b100, "blt" , setlt,  GPR>;
def BGE     : CBranch12<0b1100011, 0b101, "bge" , setge,  GPR>;
def BLTU    : CBranch12<0b1100011, 0b110, "bltu", setult, GPR>;
def BGEU    : CBranch12<0b1100011, 0b111, "bgeu", setuge, GPR>;
//@}MYRISCVXInstrInfo_BEQ

//@{MYRISCVXInstrInfo_PseudoBR
let isBarrier = 1, isBranch = 1, isTerminator = 1 in
  def PseudoBR : MYRISCVXPseudo<(outs), (ins brtarget20:$simm20), "", [(br bb:$simm20)]>,
                 PseudoInstExpansion<(JAL ZERO, brtarget20:$simm20)>;
//@}MYRISCVXInstrInfo_PseudoBR


//@{MYRISCVXInstrInfo_PseudoBRIND
let isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in
def PseudoBRIND : MYRISCVXPseudo<(outs), (ins GPR:$rs1, simm12:$simm12), "", []>,
                  PseudoInstExpansion<(JALR ZERO, GPR:$rs1, simm12:$simm12)>;
//@}MYRISCVXInstrInfo_PseudoBRIND

//@{MYRISCVXInstrInfo_PseudoBRIND_Pat
def : Pat<(brind GPR:$rs1), (PseudoBRIND GPR:$rs1, 0)>;
def : Pat<(brind (add GPR:$rs1, simm12:$simm12)),
          (PseudoBRIND GPR:$rs1, simm12:$simm12)>;
//@}MYRISCVXInstrInfo_PseudoBRIND_Pat

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

//@{MYRISCVXInstrInfo_SLTInsts
def SLTI  : SetCC_I<0b0010011, 0b010, "slti",  setlt,  simm12, GPR>;
def SLTIU : SetCC_I<0b0010011, 0b011, "sltiu", setult, simm12, GPR>;
def SLT   : SetCC_R<0b0110011, 0b010, 0b0000000, "slt",  setlt,  GPR>;
def SLTU  : SetCC_R<0b0110011, 0b011, 0b0000000, "sltu", setult, GPR>;
//@}MYRISCVXInstrInfo_SLTInsts

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


//@{MYRISCVXInstrInfo_HiLoPattern
// hi/lo relocs
def : Pat<(MYRISCVXHi tglobaladdr:$in), (LUI tglobaladdr:$in)>;
def : Pat<(MYRISCVXLo tglobaladdr:$in), (ADDI ZERO, tglobaladdr:$in)>;
def : Pat<(add GPR:$hi, (MYRISCVXLo tglobaladdr:$lo)),
          (ADDI GPR:$hi, tglobaladdr:$lo)>;
//@}MYRISCVXInstrInfo_HiLoPattern
// gp_rel relocs
def : Pat<(add GPR:$gp, (MYRISCVXGPRel tglobaladdr:$in)),
          (ADDI GPR:$gp, tglobaladdr:$in)>;


//@{MYRISCVXInstrInfo_WrapperPat
class WrapperPat<SDNode node, Instruction ORiOp, RegisterClass RC>:
      Pat<(MYRISCVXWrapper RC:$gp, node:$in),
          (ORiOp RC:$gp, node:$in)>;

def : WrapperPat<tglobaladdr, ADDI, GPR>;
def : WrapperPat<texternalsym, ADDI, GPR>;
//@}MYRISCVXInstrInfo_WrapperPat


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


//@{MYRISCVXInstrInfo_SLTPatterns
// Define pattern expansions for setcc operations that aren't directly
// handled by a RISC-V instruction.
def : Pat<(seteq  GPR:$rs1, GPR:$rs2), (SLTIU (XOR GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setne  GPR:$rs1, GPR:$rs2), (SLTU ZERO, (XOR GPR:$rs1, GPR:$rs2))>;
def : Pat<(setugt GPR:$rs1, GPR:$rs2), (SLTU GPR:$rs2, GPR:$rs1)>;
def : Pat<(setuge GPR:$rs1, GPR:$rs2), (XORI (SLTU GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setule GPR:$rs1, GPR:$rs2), (XORI (SLTU GPR:$rs2, GPR:$rs1), 1)>;
def : Pat<(setgt  GPR:$rs1, GPR:$rs2), (SLT GPR:$rs2, GPR:$rs1)>;
def : Pat<(setge  GPR:$rs1, GPR:$rs2), (XORI (SLT GPR:$rs1, GPR:$rs2), 1)>;
def : Pat<(setle  GPR:$rs1, GPR:$rs2), (XORI (SLT GPR:$rs2, GPR:$rs1), 1)>;
//@}MYRISCVXInstrInfo_SLTPatterns


//@{MYRISCVXInstrInfo_BrcondPatsSlt
// brcond for slt instruction
multiclass BrcondPatsSlt<RegisterClass RC, Instruction BEQOp, Instruction BNEOp,
                      Instruction SLTOp, Instruction SLTuOp, Instruction SLTiOp,
                      Instruction SLTiuOp, Register ZEROReg> {
  def : Pat<(brcond (XLenVT (setne RC:$lhs, 0)), bb:$dst),
                (BNEOp RC:$lhs, ZEROReg, bb:$dst)>;
  def : Pat<(brcond (XLenVT (seteq RC:$lhs, 0)), bb:$dst),
                (BEQOp RC:$lhs, ZEROReg, bb:$dst)>;

  def : Pat<(brcond (XLenVT (seteq RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQOp RC:$lhs, RC:$rhs, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setueq RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQOp RC:$lhs, RC:$rhs, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setne RC:$lhs, RC:$rhs)), bb:$dst),
                (BNEOp RC:$lhs, RC:$rhs, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setune RC:$lhs, RC:$rhs)), bb:$dst),
                (BNEOp RC:$lhs, RC:$rhs, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setlt RC:$lhs, RC:$rhs)), bb:$dst),
                (BNE (SLTOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setult RC:$lhs, RC:$rhs)), bb:$dst),
                (BNE (SLTuOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setgt RC:$lhs, RC:$rhs)), bb:$dst),
                (BNE (SLTOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setugt RC:$lhs, RC:$rhs)), bb:$dst),
                (BNE (SLTuOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setle RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQ (SLTOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setule RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQ (SLTuOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setge RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQ (SLTOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
  def : Pat<(brcond (XLenVT (setuge RC:$lhs, RC:$rhs)), bb:$dst),
                (BEQ (SLTuOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
  def : Pat<(brcond RC:$cond, bb:$dst),
                (BNEOp RC:$cond, ZEROReg, bb:$dst)>;
}

defm : BrcondPatsSlt<GPR, BEQ, BNE, SLT, SLTU, SLTI, SLTIU, ZERO>;
//@}MYRISCVXInstrInfo_BrcondPatsSlt
