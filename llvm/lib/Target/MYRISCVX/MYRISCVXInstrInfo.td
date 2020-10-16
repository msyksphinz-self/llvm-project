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


// @{ MYRISCVXInstrInfo_SDT_MYRISCVXCall
def SDT_MYRISCVXCall       : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;
// @} MYRISCVXInstrInfo_SDT_MYRISCVXCall
// @{ MYRISCVXInstrInfo_MYRISCVXSelectCC
def SDT_MYRISCVXSelectCC   : SDTypeProfile<1, 5, [SDTCisSameAs<1, 2>,
                                                  SDTCisSameAs<0, 4>,
                                                  SDTCisSameAs<4, 5>]>;
// @} MYRISCVXInstrInfo_MYRISCVXSelectCC

//@{ MYRISCVXInstrInfo_MYRISCVXRet
// MYRISCVXRetノードを定義する
def MYRISCVXRet : SDNode<"MYRISCVXISD::Ret", SDTNone,
                         [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
//@} MYRISCVXInstrInfo_MYRISCVXRet

// @{ MYRISCVXInstrInfo_MYRISCVXTailCall
// Tail call
def MYRISCVXTailCall : SDNode<"MYRISCVXISD::TailCall", SDT_MYRISCVXCall,
                               [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;
// @} MYRISCVXInstrInfo_MYRISCVXTailCall

// @{ MYRISCVXInstrInfo_SelectCC
def SelectCC : SDNode<"MYRISCVXISD::SELECT_CC", SDT_MYRISCVXSelectCC,
                      [SDNPInGlue]>;
// @} MYRISCVXInstrInfo_SelectCC

// @{ MYRISCVXInstrInfo_SDT_MYRISCVXCallSeq
def SDT_MYRISCVXCallSeqStart : SDCallSeqStart<[SDTCisVT<0, XLenVT>, SDTCisVT<1, XLenVT>]>;
def SDT_MYRISCVXCallSeqEnd   : SDCallSeqEnd  <[SDTCisVT<0, XLenVT>, SDTCisVT<1, XLenVT>]>;
// @} MYRISCVXInstrInfo_SDT_MYRISCVXCallSeq


// @{ MYRISCVXInstrInfo_MYRISCVXCall
def MYRISCVXCall : SDNode<"MYRISCVXISD::CALL", SDT_MYRISCVXCall,
                          [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                           SDNPVariadic]>;
// @} MYRISCVXInstrInfo_MYRISCVXCall

// @{ MYRISCVXInstrInfo_MYRISCVXCallSeqStart
// These are target-independent nodes, but have target-specific formats.
def MYRISCVXCallSeqStart : SDNode<"ISD::CALLSEQ_START", SDT_MYRISCVXCallSeqStart,
                                  [SDNPHasChain, SDNPOutGlue]>;
def MYRISCVXCallSeqEnd : SDNode<"ISD::CALLSEQ_END", SDT_MYRISCVXCallSeqEnd,
                                [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
// @} MYRISCVXInstrInfo_MYRISCVXCallSeqStart


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

def CallSymbol : AsmOperandClass {
  let Name = "CallSymbol";
  let RenderMethod = "addImmOperands";
}

// A bare symbol used in call/tail only.
def call_symbol : Operand<XLenVT> {
  let ParserMatchClass = CallSymbol;
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
// @{ MYRISCVXInstrInfo_brtarget20
def brtarget20    : Operand<OtherVT> {
  let EncoderMethod = "getBranch20TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}
// @} MYRISCVXInstrInfo_brtarget20


// @{ MYRISCVXInstrInfo_brtarget12
// BEQ, BNE
def brtarget12    : Operand<OtherVT> {
  let EncoderMethod = "getBranch12TargetOpValue";
  let OperandType   = "OPERAND_PCREL";
}
// @} MYRISCVXInstrInfo_brtarget12


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


// @{ MYRISCVXInstrInfo_JumpLink
// Jump and Link (Call)
let isCall = 1 in
class JumpLink<bits<7> opcode, string opstr, DAGOperand opnd> :
  MYRISCVX_J<opcode, (outs GPR:$rd), (ins opnd:$imm20), !strconcat(opstr, "\t$rd, $imm20"),
                 [], IIAlu>;
// @} MYRISCVXInstrInfo_JumpLink

// @{ MYRISCVXInstrInfo_JAL
// JAL命令の定義
def JAL : JumpLink<0b1101111, "jal", brtarget20>;
def : InstAlias<"j $offset",   (JAL  ZERO, brtarget20:$offset)>;  // JALのエイリアス: "j命令"
def : InstAlias<"jal $offset", (JAL  RA,   brtarget20:$offset)>;  // JAL命令のレジスタ省略版：暗黙的にRAレジスタが指定される
// @} MYRISCVXInstrInfo_JAL


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

//@{ MYRISCVXInstrInfo_PseudoTailCall
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, hasDelaySlot = 0,
    hasExtraSrcRegAllocReq = 1 in {
  def PseudoTAILCALL : MYRISCVXPseudo<(outs), (ins call_symbol:$target), "tail\t$target", []>;
}
//@} MYRISCVXInstrInfo_PseudoTailCall


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


// @{ MYRISCVXInstrInfo_CBranch12
// 2レジスタオペランドをとる条件分岐命令のテンプレートクラス
class CBranch12<bits<7> opcode, bits<3> funct3,
                string instr_asm,
                PatFrag cond_op, RegisterClass RC> :
  MYRISCVX_B<opcode, funct3, (outs), (ins RC:$rs1, RC:$rs2, brtarget12:$imm12),
  !strconcat(instr_asm, "\t$rs1, $rs2, $imm12"),
  [], IIAlu> {
    let isBranch = 1;
    let isTerminator = 1;
}
// @} MYRISCVXInstrInfo_CBranch12


//@{ MYRISCVXInstrInfo_BEQ
// 条件分岐命令の定義
def BEQ     : CBranch12<0b1100011, 0b000, "beq" , seteq,  GPR>;
def BNE     : CBranch12<0b1100011, 0b001, "bne" , setne,  GPR>;
def BLT     : CBranch12<0b1100011, 0b100, "blt" , setlt,  GPR>;
def BGE     : CBranch12<0b1100011, 0b101, "bge" , setge,  GPR>;
def BLTU    : CBranch12<0b1100011, 0b110, "bltu", setult, GPR>;
def BGEU    : CBranch12<0b1100011, 0b111, "bgeu", setuge, GPR>;
//@} MYRISCVXInstrInfo_BEQ

// @{ MYRISCVXInstrInfo_BRANCH_PseudoInstructions
def: InstAlias<"beqz $rs1, $simm12", (BEQ GPR:$rs1, ZERO,    brtarget12:$simm12)>;
def: InstAlias<"bnez $rs1, $simm12", (BNE GPR:$rs1, ZERO,    brtarget12:$simm12)>;
def: InstAlias<"blez $rs1, $simm12", (BGE ZERO,    GPR:$rs1, brtarget12:$simm12)>;
def: InstAlias<"bgez $rs1, $simm12", (BGE GPR:$rs1, ZERO,    brtarget12:$simm12)>;
def: InstAlias<"bltz $rs1, $simm12", (BLT GPR:$rs1, ZERO,    brtarget12:$simm12)>;
def: InstAlias<"bgtz $rs1, $simm12", (BLT ZERO,    GPR:$rs1, brtarget12:$simm12)>;

def: InstAlias<"bgt  $rs1, $rs2, $simm12", (BLT  GPR:$rs2, GPR:$rs1, brtarget12:$simm12)>;
def: InstAlias<"ble  $rs1, $rs2, $simm12", (BGE  GPR:$rs2, GPR:$rs1, brtarget12:$simm12)>;
def: InstAlias<"bgtu $rs1, $rs2, $simm12", (BLTU GPR:$rs2, GPR:$rs1, brtarget12:$simm12)>;
def: InstAlias<"bleu $rs1, $rs2, $simm12", (BGEU GPR:$rs2, GPR:$rs1, brtarget12:$simm12)>;
// @} MYRISCVXInstrInfo_BRANCH_PseudoInstructions


//@{ MYRISCVXInstrInfo_PseudoBR
// ブロックへのジャンプを検出すると, JAL命令への置き換えを行うパタン
let isBarrier = 1, isBranch = 1, isTerminator = 1 in
  def PseudoBR : MYRISCVXPseudo<(outs), (ins brtarget20:$simm20u), "", [(br bb:$simm20u)]>,
                 PseudoInstExpansion<(JAL ZERO, brtarget20:$simm20u)>;
//@} MYRISCVXInstrInfo_PseudoBR


//@{ MYRISCVXInstrInfo_PseudoBRIND
// PseudoBRとは違い、 汎用レジスタの値と即値を使ってジャンプするケース
let isBarrier = 1, isBranch = 1, isIndirectBranch = 1, isTerminator = 1 in
def PseudoBRIND : MYRISCVXPseudo<(outs), (ins GPR:$rs1, simm12:$simm12), "", []>,
                  PseudoInstExpansion<(JALR ZERO, GPR:$rs1, simm12:$simm12)>;
//@} MYRISCVXInstrInfo_PseudoBRIND

//@{ MYRISCVXInstrInfo_PseudoBRIND_Pat
// PseudoBRINDの生成パタン。 汎用レジスタと即値の組み合わせで生成する
def : Pat<(brind GPR:$rs1), (PseudoBRIND GPR:$rs1, 0)>;
def : Pat<(brind (add GPR:$rs1, simm12:$simm12)),
          (PseudoBRIND GPR:$rs1, simm12:$simm12)>;
//@} MYRISCVXInstrInfo_PseudoBRIND_Pat


//@{ MYRISCVXInstrInfo_SelectCC_rrirr
// カスタムのSelectCCノードを使用するクラス
// useCustomInserter=1とすることで, EmitInstrWithCustomInserter()での生成対象となる
let usesCustomInserter = 1 in
class SelectCC_rrirr<RegisterClass RC, RegisterClass cmpty>
    : MYRISCVXPseudo<(outs RC:$dst),
             (ins cmpty:$lhs, cmpty:$rhs, simm12:$imm,
              RC:$truev, RC:$falsev),
              "",
             [(set RC:$dst,
               (SelectCC cmpty:$lhs,
                         cmpty:$rhs,
                         (XLenVT imm:$imm),
                         RC:$truev,
                         RC:$falsev))]>;

def Select_GPR_Using_CC_GPR : SelectCC_rrirr<GPR, GPR>;
//@} MYRISCVXInstrInfo_SelectCC_rrirr

// @{ MYRISCVXInstrInfo_ADJCALLSTACK
let Defs = [SP], Uses = [SP], hasSideEffects = 1 in {
  // As stack alignment is always done with addi, we need a 12-bit immediate
  def ADJCALLSTACKDOWN : MYRISCVXPseudo<(outs), (ins simm12:$amt1, simm12:$amt2), "",
                                        [(MYRISCVXCallSeqStart timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP   : MYRISCVXPseudo<(outs), (ins simm12:$amt1, simm12:$amt2), "",
                                        [(MYRISCVXCallSeqEnd timm:$amt1, timm:$amt2)]>;
}
// @} MYRISCVXInstrInfo_ADJCALLSTACK


//}
//@def CPRESTORE {
// When handling PIC code the assembler needs .cpload and .cprestore
// directives. If the real instructions corresponding these directives
// are used, we have the same behavior, but get also a bunch of warnings
// from the assembler.
let hasSideEffects = 0 in
def CPRESTORE : MYRISCVXPseudo<(outs), (ins simm12:$loc, GPR:$gp),
                            ".cprestore\t$loc", []>;

// @{ MYRISCVXInstrInfo_MYRISCVXTailCall_Convert
// カスタムTail CallノードはTail疑似命令に変換される
def : Pat<(MYRISCVXTailCall tglobaladdr:$dst),  (PseudoTAILCALL tglobaladdr:$dst)>;
def : Pat<(MYRISCVXTailCall texternalsym:$dst), (PseudoTAILCALL texternalsym:$dst)>;
// @} MYRISCVXInstrInfo_MYRISCVXTailCall_Convert

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


// @{ MYRISCVXInstrInfo_PseudoCall
let isCall = 1, Defs = [RA], isCodeGenOnly = 0 in {
  def PseudoCALL : MYRISCVXPseudo<(outs), (ins call_symbol:$target), "call\t$target", []>;
}

def : Pat<(MYRISCVXCall tglobaladdr:$func),  (PseudoCALL tglobaladdr:$func)>;
def : Pat<(MYRISCVXCall texternalsym:$func), (PseudoCALL texternalsym:$func)>;
// @} MYRISCVXInstrInfo_PseudoCall

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

// @{ MYRISCVXInstrInfo_BrcondPatsSlt
// 条件分岐命令の生成パタン一覧
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
// @} MYRISCVXInstrInfo_BrcondPatsSlt


// @{ MYRISCVXInstrInfo_LoadImm32
class LoadImm32< string instr_asm, Operand Od, RegisterClass RC> :
  MYRISCVXAsmPseudoInst<(outs RC:$rd), (ins Od:$imm32),
                     !strconcat(instr_asm, "\t$rd, $imm32")> ;
def LoadImm32Reg : LoadImm32<"li", simm32, GPR>;
// @} MYRISCVXInstrInfo_LoadImm32

// PseudoInstructions
// @{ MYRISCVXInstrInfo_PseudoInstructions
def: InstAlias<"nop",            (ADDI ZERO,    ZERO,      0      )>;
def: InstAlias<"mv $rd, $rs1",   (ADDI GPR:$rd, GPR:$rs1,  0      )>;
def: InstAlias<"not $rd, $rs1",  (XORI GPR:$rd, GPR:$rs1, -1      )>;
def: InstAlias<"neg $rd, $rs1",  (SUB  GPR:$rd, ZERO,     GPR:$rs1)>;

let Predicates = [IsRV64] in {
  def: InstAlias<"negw $rd, $rs1",   (SUBW  GPR:$rd, ZERO,    GPR:$rs1)>;
  def: InstAlias<"sext.w $rd, $rs1", (ADDIW GPR:$rd, GPR:$rs1,       0)>;
} // Predicates = [IsRV64]

def: InstAlias<"seqz $rd, $rs1", (SLTIU GPR:$rd, GPR:$rs1, 1       )>;
def: InstAlias<"snez $rd, $rs1", (SLTU  GPR:$rd, ZERO,     GPR:$rs1)>;
def: InstAlias<"sltz $rd, $rs1", (SLT   GPR:$rd, GPR:$rs1, ZERO    )>;
def: InstAlias<"sgtz $rd, $rs1", (SLT   GPR:$rd, ZERO,     GPR:$rs1)>;
// @} MYRISCVXInstrInfo_PseudoInstructions
