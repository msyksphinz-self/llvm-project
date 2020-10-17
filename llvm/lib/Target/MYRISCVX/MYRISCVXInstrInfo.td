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

//@{ MYRISCVXInstrInfo_MYRISCVXRet
// MYRISCVXRetノードを定義する
def MYRISCVXRet : SDNode<"MYRISCVXISD::Ret", SDTNone,
                         [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
//@} MYRISCVXInstrInfo_MYRISCVXRet

//===----------------------------------------------------------------------===//
// MYRISCVX profiles and nodes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
//@{ MYRISCVXInstrInfo_Include_td
include "MYRISCVXInstrFormats.td"   // 命令フォーマットを定義したMYRISCVXInstrFormats.tdをincludeする
//@} MYRISCVXInstrInfo_Include_td

//@{ MYRISCVXInstrInfo_Operand_Definition
//===----------------------------------------------------------------------===//
// MYRISCVX のオペランド定義
//===----------------------------------------------------------------------===//
// @{ MYRISCVXInstrInfo_Operand_Definition_simm12
// 12ビットの即値オペランド
def simm12 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<12>(Imm);}]>;
// @} MYRISCVXInstrInfo_Operand_Definition_simm12
// @{ MYRISCVXInstrInfo_Operand_Definition_simm20u
// 上位20ビットで下位12ビットが0の即値オペランド
def simm20u : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<32>(Imm) && !(Imm & 0x0fff);}]>;
// @} MYRISCVXInstrInfo_Operand_Definition_simm20u
// @{ MYRISCVXInstrInfo_Operand_Definition_simm32
// 32ビットの即値オペランド
def simm32: Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<32>(Imm);}]>;
// @} MYRISCVXInstrInfo_Operand_Definition_simm32
//@} MYRISCVXInstrInfo_Operand_Definition


def BareSymbol : AsmOperandClass {
  let Name = "BareSymbol";
}

// A bare symbol.
def bare_symbol : Operand<XLenVT> {
  let ParserMatchClass = BareSymbol;
}


// @{MYRISCVXInstrInfo_LO12_HI20
// 変換関数を定義する：定数のうち下位の12ビットを抽出する関数
def LO12sext : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(SignExtend64<12>(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

// 変換関数を定義する：定数のうち上位の20ビットを抽出する関数
def HI20 : SDNodeXForm<imm, [{
  return getImm(N, ((N->getZExtValue()+0x800) >> 12) & 0xfffff);
}]>;
// @}MYRISCVXInstrInfo_LO12_HI20


// shamt field must fit in 5 bits.
def immZExt_Xlen : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  return Subtarget->is64Bit() ? isUInt<6>(Imm) : isUInt<5>(Imm);
}]>;

//@{ MYRISCVXInstrInfo_calltarget
def calltarget : Operand<iPTR> {
  let EncoderMethod = "getJumpTargetOpValue";
}
//@} MYRISCVXInstrInfo_calltarget

// JAL
def brtarget20    : Operand<OtherVT> {
  let EncoderMethod = "getBranch20TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}


// BEQ, BNE
def brtarget12    : Operand<OtherVT> {
  let EncoderMethod = "getBranch12TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}


// MYRISCVX Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr_fi : ComplexPattern<iPTR, 1, "SelectAddrFI", [frameindex], []>;


//@{ MYRISCVXInstrInfo_Instruction_Format
//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//
//@{ MYRISCVXInstrInfo_ArithLogicR
// 2レジスタオペランドを持つ算術論理演算命令
class ArithLogicR<bits<7> opcode, bits<3> funct3, bits<7>funct7,
                  string instr_asm, SDNode OpNode,
                  RegisterClass RC> :
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
  !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
  [(set RC:$rd, (OpNode RC:$rs1, RC:$rs2))], IIAlu> {
    let isReMaterializable = 1;
}
//@} MYRISCVXInstrInfo_ArithLogicR


//@{ MYRISCVXInstrInfo_ArithLogicI
// 1レジスタオペランドと即値オペランドを持つ算術論理演算命令
class ArithLogicI<bits<7> opcode, bits<3> funct3,
                  string instr_asm, SDNode OpNode,
                  Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
  !strconcat(instr_asm, "\t$rd, $rs1, $simm12"),
  [(set RC:$rd, (OpNode RC:$rs1, Od:$simm12))], IIAlu> {
    let isReMaterializable = 1;
}
//@} MYRISCVXInstrInfo_ArithLogicI

// JALRはI-typeの命令だがOpNodeを使用しない
//@{ MYRISCVXInstrInfo_ArithLogicI_J
class ArithLogicI_J<bits<7> opcode, bits<3> funct3,
                  string instr_asm,
                  Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
  !strconcat(instr_asm, "\t$rd, $rs1, $simm12"), [], IIAlu> {
}
//@} MYRISCVXInstrInfo_ArithLogicI_J


//@{ MYRISCVXInstrInfo_ArithLogicU
// 1つの即値オペランドを持つ算術論理演算命令
class ArithLogicU<bits<7> opcode,
                  string instr_asm, RegisterClass RC, Operand Od> :
  MYRISCVX_U<opcode, (outs RC:$rd), (ins Od:$imm20),
    !strconcat(instr_asm, "\t$rd, $imm20"), [], IIAlu> {
      let isReMaterializable = 1;
}
//@} MYRISCVXInstrInfo_ArithLogicU
//@} MYRISCVXInstrInfo_Instruction_Format


//@{ MYRISCVXInstrInfo_Shift
// ShiftIは即値シフト命令を生成するためのテンプレート
// opcode, funct3フィールドに加えてarithshiftを指定する
class ShiftI<bits<7> opcode, bits<3> funct3, bit arithshift, string instr_asm,
                   SDNode OpNode>:
  // ShiftIはMYRISCVX_ISHIFTを継承している
  // immZExt_XlenはRV32/RV63で長さが変わるオペランド
  MYRISCVX_ISHIFT<opcode, funct3, arithshift, (outs GPR:$rd), (ins GPR:$rs1, immZExt_Xlen:$shamt),
                  !strconcat(instr_asm, "\t$rd, $rs1, $shamt"),
                  [(set GPR:$rd, (OpNode GPR:$rs1, immZExt_Xlen:$shamt))], IIAlu> {
}

// ShiftRはレジスタオペランドシフト命令を生成するためのテンプレート
// MYRISCVX_Rとほぼ同じ
class ShiftR<bits<7> opcode, bits<7> funct7, bits<3> funct3,
                   bits<4> isRotate, string instr_asm,
                   SDNode OpNode, RegisterClass RC>:
  // ShiftRはMYRISCVX_Rを継承している
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
          !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
          [(set GPR:$rd, (OpNode RC:$rs1, RC:$rs2))], IIAlu> {
}
//@} MYRISCVXInstrInfo_Shift


//@{ MYRISCVXInstrInfo_jalr_variation
// JALR命令とそのエイリアスを定義する
def JALR : ArithLogicI_J<0b1100111, 0b000, "jalr", simm12, GPR>;
// エイリアスはjr / jalr / retの3つを定義する
def: InstAlias<"jr\t$rs",      (JALR ZERO, GPR:$rs, 0)>;
def: InstAlias<"jalr\t$rs",    (JALR RA,   GPR:$rs, 0)>;
def: InstAlias<"ret",          (JALR ZERO,      RA, 0), 2>;
//@} MYRISCVXInstrInfo_jalr_variation


//@{ MYRISCVXInstrInfo_LoadMemory_StoreMemory
// メモリロードストアのためのテンプレートを作成
let canFoldAsLoad = 1 in
class LoadMemory<bits<7> opcode, bits<3> funct3, string instr_asm, RegisterClass RC>:
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins GPR:$rs1, simm12:$simm12),
     !strconcat(instr_asm, "\t$rd, ${simm12}(${rs1})"),
     [], IILoad>;

class StoreMemory<bits<7> opcode, bits<3> funct3, string instr_asm, RegisterClass RC>:
  MYRISCVX_S<opcode, funct3, (outs), (ins RC:$rs2, GPR:$rs1, simm12:$simm12),
     !strconcat(instr_asm, "\t$rs2, ${simm12}(${rs1})"),
     [], IIStore>;
//@} MYRISCVXInstrInfo_LoadMemory_StoreMemory

//===----------------------------------------------------------------------===//
// Pattern fragment for load/store
//===----------------------------------------------------------------------===//

//@{ MYRISCVXInstrInfo_AlignedLoad_AlignedStore
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
//@} MYRISCVXInstrInfo_AlignedLoad_AlignedStore


// @{ MYRISCVXInstrInfo_SetCC
// SetCC
class SetCC_R<bits<7> opcode, bits<3> funct3, bits<7> funct7,
              string instr_asm, PatFrag cond_op,
              RegisterClass RC> :
  MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
             !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
             [(set GPR:$rd, (cond_op RC:$rs1, RC:$rs2))], IIAlu> {
               let isReMaterializable = 1;
}


class SetCC_I<bits<7> opcode, bits<3> funct3,
              string instr_asm, PatFrag cond_op,
              Operand Od, RegisterClass RC> :
  MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
             !strconcat(instr_asm, "\t$rd, $rs1, $simm12"),
             [(set GPR:$rd, (cond_op RC:$rs1, Od:$simm12))], IIAlu> {
               let isReMaterializable = 1;
}
// @} MYRISCVXInstrInfo_SetCC


//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
//@{ MYRISCVXInstrInfo_Basic_Definition
//===----------------------------------------------------------------------===//
// MYRISCVX Instructions
//===----------------------------------------------------------------------===//
//@{ MYRISCVXInstrInfo_Arithmetic
//@{ MYRISCVXInstrInfo_Arithmetic_ADDI
def ADDI : ArithLogicI<0b0010011, 0b000, "addi", add, simm12, GPR>;
//@} MYRISCVXInstrInfo_Arithmetic_ADDI
def XORI : ArithLogicI<0b0010011, 0b100, "xori", xor, simm12, GPR>;
def ORI  : ArithLogicI<0b0010011, 0b110, "ori",  or,  simm12, GPR>;
def ANDI : ArithLogicI<0b0010011, 0b111, "andi", and, simm12, GPR>;

// @{ MYRISCVXInstrInfo_AUIPC
def AUIPC: ArithLogicU<0b0010111, "auipc", GPR, simm20u>;
// @} MYRISCVXInstrInfo_AUIPC
def LUI  : ArithLogicU<0b0110111, "lui", GPR, simm20u>;
def ADD  : ArithLogicR<0b0110011, 0b000, 0b0000000, "add", add, GPR>;
def SUB  : ArithLogicR<0b0110011, 0b000, 0b0100000, "sub", sub, GPR>;
//@} MYRISCVXInstrInfo_Basic_Definition

def AND  : ArithLogicR     <0b0110011, 0b111, 0b0000000, "and", and, GPR>;
def OR   : ArithLogicR     <0b0110011, 0b110, 0b0000000, "or",  or,  GPR>;
def XOR  : ArithLogicR     <0b0110011, 0b100, 0b0000000, "xor", xor, GPR>;
//@} MYRISCVXInstrInfo_Arithmetic

// @{ MYRISCVXInstrInfo_ShiftInsts
def SLL  : ShiftR<0b0110011, 0b0000000, 0b001, 0x0, "sll", shl, GPR>;
def SRL  : ShiftR<0b0110011, 0b0000000, 0b101, 0x0, "srl", srl, GPR>;
def SRA  : ShiftR<0b0110011, 0b0100000, 0b101, 0x0, "sra", sra, GPR>;

def SRLI : ShiftI <0b0010011, 0b101, 0, "srli", srl>;
def SLLI : ShiftI <0b0010011, 0b001, 0, "slli", shl>;
def SRAI : ShiftI <0b0010011, 0b101, 1, "srai", sra>;
// @} MYRISCVXInstrInfo_ShiftInsts

// @{ MYRISCVXInstrInfo_MulDivInsts
def MUL   : ArithLogicR<0b0110011, 0b000, 0b0000001, "mul",    mul,   GPR>;
def MULH  : ArithLogicR<0b0110011, 0b001, 0b0000001, "mulh",   mulhs, GPR>;
def MULHSU: ArithLogicR<0b0110011, 0b010, 0b0000001, "mulhsu", mulhs, GPR>;
def MULHU : ArithLogicR<0b0110011, 0b011, 0b0000001, "mulhu",  mulhu, GPR>;
def DIV   : ArithLogicR<0b0110011, 0b100, 0b0000001, "div",    sdiv,  GPR>;
def DIVU  : ArithLogicR<0b0110011, 0b101, 0b0000001, "divu",   udiv,  GPR>;
def REM   : ArithLogicR<0b0110011, 0b110, 0b0000001, "rem",    srem,  GPR>;
def REMU  : ArithLogicR<0b0110011, 0b111, 0b0000001, "remu",   urem,  GPR>;
// @} MYRISCVXInstrInfo_MulDivInsts

// @{ MYRISCVXInstrInfo_SLTInsts
def SLTI  : SetCC_I<0b0010011, 0b010, "slti",  setlt,  simm12, GPR>;
def SLTIU : SetCC_I<0b0010011, 0b011, "sltiu", setult, simm12, GPR>;
def SLT   : SetCC_R<0b0110011, 0b010, 0b0000000, "slt",  setlt,  GPR>;
def SLTU  : SetCC_R<0b0110011, 0b011, 0b0000000, "sltu", setult, GPR>;
// @} MYRISCVXInstrInfo_SLTInsts

def assertzexti32 : PatFrag<(ops node:$src),
                            (assertzext node:$src),
                            [{ return cast<VTSDNode>(N->getOperand(1))->getVT() == MVT::i32; }]>;

def zexti32 : PatFrags<(ops node:$src),
                       [(and node:$src, 0xffffffff),
                        (assertzexti32 node:$src)]>;

// @{ MYRISCVXInstrInfo_td_32bitsTemplate
let Predicates = [IsRV64] in {
  class ArithLogicI_W<bits<7> opcode, bits<3> funct3,
                    string instr_asm, Operand Od, RegisterClass RC> :
    MYRISCVX_I<opcode, funct3, (outs RC:$rd), (ins RC:$rs1, Od:$simm12),
    !strconcat(instr_asm, "\t$rd, $rs1, $simm12"),
    [], IIAlu> {
      let isReMaterializable = 1;
  }

  class ArithLogicR_W<bits<7> opcode, bits<3> funct3, bits<7>funct7,
                    string instr_asm, RegisterClass RC> :
    MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
    !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
    [], IIAlu> {
      let isReMaterializable = 1;
  }

  class ShiftI32_W<bits<7> opcode, bits<3> funct3, bit arithshift, string instr_asm>:
    MYRISCVX_ISHIFT<opcode, funct3, arithshift, (outs GPR:$rd), (ins GPR:$rs1, immZExt_Xlen:$shamt),
                    !strconcat(instr_asm, "\t$rd, $rs1, $shamt"),
                    [], IIAlu> {
  }

  class ShiftR_W<bits<7> opcode, bits<7> funct7, bits<3> funct3,
                     bits<4> isRotate, string instr_asm,
                     RegisterClass RC>:
    MYRISCVX_R<opcode, funct3, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
            !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
            [], IIAlu> {
  }
// @} MYRISCVXInstrInfo_td_32bitsTemplate

// @{ MYRISCVXInstrInfo_td_ArithInsts32
  def SLLIW : ShiftI32_W <0b0011011, 0b001, 0, "slliw">;
  def SRLIW : ShiftI32_W <0b0011011, 0b101, 0, "srliw">;
  def SRAIW : ShiftI32_W <0b0011011, 0b101, 1, "sraiw">;
  def ADDIW : ArithLogicI_W    <0b0011011, 0b000, "addiw", simm12, GPR>;
  def ADDW  : ArithLogicR_W    <0b0111011, 0b000, 0b0000000, "addw", GPR>;
  def SUBW  : ArithLogicR_W    <0b0111011, 0b000, 0b0100000, "subw", GPR>;
  def SLLW  : ShiftR_W   <0b0111011, 0b0000000, 0b001, 0x0, "sllw", GPR>;
  def SRLW  : ShiftR_W   <0b0111011, 0b0000000, 0b101, 0x0, "srlw", GPR>;
  def SRAW  : ShiftR_W   <0b0111011, 0b0100000, 0b101, 0x0, "sraw", GPR>;
// @} MYRISCVXInstrInfo_td_ArithInsts32


  // @{ MYRISCVXInstrInfo_td_MulDivInsts32
  def MULW  : ArithLogicR_W<0b0111011, 0b000, 0b0000001, "mulw",  GPR>;
  def DIVW  : ArithLogicR_W<0b0111011, 0b100, 0b0000001, "divw",  GPR>;
  def DIVUW : ArithLogicR_W<0b0111011, 0b101, 0b0000001, "divuw", GPR>;
  def REMW  : ArithLogicR_W<0b0111011, 0b110, 0b0000001, "remw",  GPR>;
  def REMUW : ArithLogicR_W<0b0111011, 0b111, 0b0000001, "remuw", GPR>;
  // @} MYRISCVXInstrInfo_td_MulDivInsts32

// @{ MYRISCVXInstrInfo_td_ArithInsts32_Pattern
  def : Pat<(sext_inreg GPR:$rs1, i32), (ADDIW GPR:$rs1, 0)>;
  def : Pat<(and GPR:$rs1, 0xffffffff), (SRLI (SLLI GPR:$rs1, 32), 32)>;

  def : Pat<(sext_inreg (add GPR:$rs1, GPR:$rs2), i32),
            (ADDW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (add GPR:$rs1, simm12:$imm12), i32),
            (ADDIW GPR:$rs1, simm12:$imm12)>;
  def : Pat<(sext_inreg (sub GPR:$rs1, GPR:$rs2), i32),
            (SUBW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (srl GPR:$rs1, GPR:$rs2), i32),
            (SRLW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (shl GPR:$rs1, GPR:$rs2), i32),
            (SLLW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (sra GPR:$rs1, GPR:$rs2), i32),
            (SRAW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (shl GPR:$rs1, immZExt_Xlen:$shamt), i32),
            (SLLIW GPR:$rs1, immZExt_Xlen:$shamt)>;
  def : Pat<(sext_inreg (sra GPR:$rs1, immZExt_Xlen:$shamt), i32),
            (SRAIW GPR:$rs1, immZExt_Xlen:$shamt)>;
  def : Pat<(sext_inreg (srl GPR:$rs1, immZExt_Xlen:$shamt), i32),
            (SRLIW GPR:$rs1, immZExt_Xlen:$shamt)>;
// @} MYRISCVXInstrInfo_td_ArithInsts32_Pattern

  // @{ MYRISCVXInstrInfo_td_MulDivInsts32_Pattern
  def : Pat<(sext_inreg (mul GPR:$rs1, GPR:$rs2), i32),
            (MULW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (sdiv GPR:$rs1, GPR:$rs2), i32),
            (DIVW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(zexti32 (udiv (zexti32 GPR:$rs1), (zexti32 GPR:$rs2))),
            (DIVUW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(sext_inreg (srem GPR:$rs1, GPR:$rs2), i32),
            (REMW GPR:$rs1, GPR:$rs2)>;
  def : Pat<(zexti32 (urem (zexti32 GPR:$rs1), (zexti32 GPR:$rs2))),
            (REMUW GPR:$rs1, GPR:$rs2)>;
  // @} MYRISCVXInstrInfo_td_MulDivInsts32_Pattern
}


//@{ MYRISCVXInstrInfo_RetRA
// RetRAノードの定義 RetRAは具体的な命令を持たない疑似ノード
// MYRISCVXRetはRetRAに変換される
let isReturn=1, isTerminator=1, hasDelaySlot=0, isBarrier=1, hasCtrlDep=1 in
  def RetRA : MYRISCVXPseudo<(outs), (ins), "", [(MYRISCVXRet)]>;
//@} MYRISCVXInstrInfo_RetRA

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// @{ MYRISCVXInstrInfo_LoadStore_PatFrags
// AlignedLoad / AlignStoreを継承してロードストアのためのOpNodeを生成する
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
// @} MYRISCVXInstrInfo_LoadStore_PatFrags


// @{ MYRISCVXInstrInfo_td_MemoryInsts
/// メモリアクセス命令の定義
def LW  : LoadMemory <0b0000011, 0b010, "lw",  GPR>;
def SW  : StoreMemory<0b0100011, 0b010, "sw",  GPR>;
def LB  : LoadMemory <0b0000011, 0b000, "lb",  GPR>;
def LBU : LoadMemory <0b0000011, 0b100, "lbu", GPR>;
def SB  : StoreMemory<0b0100011, 0b000, "sb",  GPR>;
def LH  : LoadMemory <0b0000011, 0b001, "lh",  GPR>;
def LHU : LoadMemory <0b0000011, 0b101, "lhu", GPR>;
def SH  : StoreMemory<0b0100011, 0b001, "sh",  GPR>;
// @} MYRISCVXInstrInfo_td_MemoryInsts


// Small immediates

// @{MYRISCVXInstrInfo_Pat_simm12
// 12ビットの定数を生成するパタン
def : Pat<(simm12:$in), (ADDI ZERO, simm12:$in)>;
// @}MYRISCVXInstrInfo_Pat_simm12

// @{MYRISCVXInstrInfo_Pat_simm20u
// 下位12ビットが0の定数を生成するパタン
def : Pat<(simm20u:$in), (LUI (HI20 imm:$in))>;
// @}MYRISCVXInstrInfo_Pat_simm20u
// @{MYRISCVXInstrInfo_Pat_simm32
// 一般的な32ビットの定数を生成するパタン
def : Pat<(simm32:$imm),
          (ADDI (LUI (HI20 imm:$imm)), (LO12sext imm:$imm))>;
// @}MYRISCVXInstrInfo_Pat_simm32


// Addr for Frame
def : Pat<(add (XLenVT addr_fi:$rs1), simm12:$simm12),
          (ADDI (XLenVT addr_fi:$rs1), simm12:$simm12)>;

// @{MYRISCVXInstrInfo_td_MemoryPattern_Define
// メモリアクセスのためのパタン定義
// multiclassは複数のパタン定義をまとめ上げることができる
multiclass LoadPattern<PatFrag LoadOp, MYRISCVXInst Inst> {
  def : Pat<(LoadOp GPR:$rs1)                          , (Inst GPR:$rs1, 0)                  >;
  def : Pat<(LoadOp addr_fi:$rs1)                      , (Inst addr_fi:$rs1, 0)              >;
  def : Pat<(LoadOp (add GPR:$rs1, simm12:$simm12))    , (Inst GPR:$rs1, simm12:$simm12)     >;
  def : Pat<(LoadOp (add addr_fi:$rs1, simm12:$simm12)), (Inst addr_fi:$rs1, simm12:$simm12) >;
}

multiclass StorePattern<PatFrag StoreOp, MYRISCVXInst Inst, RegisterClass RC> {
  def : Pat<(StoreOp RC:$rs2, GPR:$rs1)                          , (Inst RC:$rs2, GPR:$rs1, 0)                 >;
  def : Pat<(StoreOp RC:$rs2, addr_fi:$rs1)                      , (Inst RC:$rs2, addr_fi:$rs1, 0)             >;
  def : Pat<(StoreOp RC:$rs2, (add GPR:$rs1, simm12:$simm12))    , (Inst RC:$rs2, GPR:$rs1, simm12:$simm12)    >;
  def : Pat<(StoreOp RC:$rs2, (add addr_fi:$rs1, simm12:$simm12)), (Inst RC:$rs2, addr_fi:$rs1, simm12:$simm12)>;
}
// @}MYRISCVXInstrInfo_td_MemoryPattern_Define

// @{ MYRISCVXInstrInfo_td_MemoryPattern_Impl
// メモリアクセスのためのノードとMYRISCVXの命令を関連付ける
defm : LoadPattern<sextloadi8  , LB>;
defm : LoadPattern<extloadi8   , LB>;
defm : LoadPattern<sextloadi16 , LH>;
defm : LoadPattern<extloadi16  , LH>;
defm : LoadPattern<load        , LW>, Requires<[IsRV32]>;
defm : LoadPattern<zextloadi8  , LBU>;
defm : LoadPattern<zextloadi16 , LHU>;

defm : StorePattern<truncstorei8  , SB, GPR>;
defm : StorePattern<truncstorei16 , SH, GPR>;
defm : StorePattern<store         , SW, GPR>, Requires<[IsRV32]>;
// @} MYRISCVXInstrInfo_td_MemoryPattern_Impl


// @{ MYRISCVXInstrInfo_td_MemoryPattern64
// RV64のときは以下の命令を追加で定義する
let Predicates = [IsRV64] in {
  def LD  : LoadMemory <0b0000011, 0b011, "ld",  GPR>;
  def SD  : StoreMemory<0b0100011, 0b011, "sd",  GPR>;
  def LWU : LoadMemory <0b0000011, 0b110, "lwu", GPR>;

  /// Loads
  defm : LoadPattern<sextloadi32 , LW>;
  defm : LoadPattern<extloadi32  , LW>;
  defm : LoadPattern<zextloadi32 , LWU>;
  defm : LoadPattern<load        , LD>;

  /// Stores
  defm : StorePattern<truncstorei32 , SW, GPR>;
  defm : StorePattern<store         , SD, GPR>;
} // Predicates = [IsRV64]
// @} MYRISCVXInstrInfo_td_MemoryPattern64


// @{ MYRISCVXInstrInfo_SLTPatterns
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
// @} MYRISCVXInstrInfo_SLTPatterns


// @{ MYRISCVXInstrInfo_PseudoLA_LLA
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLLA : MYRISCVXPseudo<(outs GPR:$dst), (ins bare_symbol:$src), "lla\t$dst, $src", []>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 0,
    isAsmParserOnly = 1 in
def PseudoLA : MYRISCVXPseudo<(outs GPR:$dst), (ins bare_symbol:$src), "la\t$dst, $src", []>;
// @} MYRISCVXInstrInfo_PseudoLA_LLA
