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


//@{ MYRISCVXInstrInfo_jalr_variation
// JALR命令とそのエイリアスを定義する
def JALR : ArithLogicI_J<0b1100111, 0b000, "jalr", simm12, GPR>;
// エイリアスはjr / jalr / retの3つを定義する
def: InstAlias<"jr\t$rs",      (JALR ZERO, GPR:$rs, 0)>;
def: InstAlias<"jalr\t$rs",    (JALR RA,   GPR:$rs, 0)>;
def: InstAlias<"ret",          (JALR ZERO,      RA, 0), 2>;
//@} MYRISCVXInstrInfo_jalr_variation


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

def LUI  : ArithLogicU<0b0110111, "lui", GPR, simm20u>;
def ADD  : ArithLogicR<0b0110011, 0b000, 0b0000000, "add", add, GPR>;
//@} MYRISCVXInstrInfo_Basic_Definition

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//
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
