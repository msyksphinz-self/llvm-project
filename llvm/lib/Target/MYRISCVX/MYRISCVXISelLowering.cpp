//===-- MYRISCVXISelLowering.cpp - MYRISCVX DAG Lowering Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MYRISCVX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//
#include "MYRISCVXISelLowering.h"

#include "MYRISCVXMachineFunction.h"
#include "MYRISCVXTargetMachine.h"
#include "MYRISCVXTargetObjectFile.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

const char *MYRISCVXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    case MYRISCVXISD::CALL:              return "MYRISCVXISD::CALL";
    case MYRISCVXISD::TailCall:          return "MYRISCVXISD::TailCall";
    case MYRISCVXISD::Hi:                return "MYRISCVXISD::Hi";
    case MYRISCVXISD::Lo:                return "MYRISCVXISD::Lo";
    case MYRISCVXISD::GPRel:             return "MYRISCVXISD::GPRel";
    case MYRISCVXISD::Ret:               return "MYRISCVXISD::Ret";
    case MYRISCVXISD::EH_RETURN:         return "MYRISCVXISD::EH_RETURN";
    case MYRISCVXISD::DivRem:            return "MYRISCVXISD::DivRem";
    case MYRISCVXISD::DivRemU:           return "MYRISCVXISD::DivRemU";
    case MYRISCVXISD::Wrapper:           return "MYRISCVXISD::Wrapper";
    case MYRISCVXISD::SELECT_CC:         return "MYRISCVXISD::SELECT_CC";
    default:                             return NULL;
  }
}


// @{ MYRISCVXTargetLowering
//@{ MYRISCVXTargetLowering_setOperationAction_DontGenerate
//@{ MYRISCVXTargetLowering_setOperationAction_GlobalAddress
// @{ MYRISCVXTargetLowering_setOperationAction_Select
// @{ MYRISCVXTargetLowering_setOperationAction_Branch
// @{ MYRISCVXTargetLowering_setOperationAction_vararg
// @{ MYRISCVXISelLowering_addRegisterClass_FPR_SRegClass
MYRISCVXTargetLowering::MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                               const MYRISCVXSubtarget &STI)
  // @{ MYRISCVXISelLowering_addRegisterClass_FPR_SRegClass ...
    : TargetLowering(TM), Subtarget(STI), ABI(TM.getABI()) {
  //@{ MYRISCVXTargetLowering_setOperationAction_DontGenerate ...
  //@{ MYRISCVXTargetLowering_setOperationAction_GlobalAddress ...
  // @{ MYRISCVXTargetLowering_setOperationAction_Branch ...
  // @{ MYRISCVXTargetLowering_setOperationAction_Select ...
  // @{ MYRISCVXTargetLowering_setOperationAction_vararg ...

  MVT XLenVT = Subtarget.getXLenVT();

  // @} MYRISCVXISelLowering_addRegisterClass_FPR_SRegClass ...
  // レジスタクラスをセットアップする
  addRegisterClass(XLenVT, &MYRISCVX::GPRRegClass);

  addRegisterClass(MVT::f32, &MYRISCVX::FPR_SRegClass);
  addRegisterClass(MVT::f64, &MYRISCVX::FPR_DRegClass);
  // @} MYRISCVXISelLowering_addRegisterClass_FPR_SRegClass

  // 関数の配置アライメント 関数は4バイトアラインに配置する
  setMinFunctionAlignment(Align(4));

  // 全てのレジスタを宣言すると、computeRegisterProperties()を呼び出さなければならない
  computeRegisterProperties(STI.getRegisterInfo());

  //@} MYRISCVXTargetLowering_setOperationAction_DontGenerate ...
  setOperationAction(ISD::ROTL, XLenVT, Expand);
  setOperationAction(ISD::ROTR, XLenVT, Expand);
  setOperationAction(ISD::CTLZ,  XLenVT, Expand);
  setOperationAction(ISD::CTPOP, XLenVT, Expand);
  //@} MYRISCVXTargetLowering_setOperationAction_DontGenerate

  //@} MYRISCVXTargetLowering_setOperationAction_GlobalAddress ...
  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  //@} MYRISCVXTargetLowering_setOperationAction_GlobalAddress

  // @} MYRISCVXTargetLowering_setOperationAction_Branch ...
  // Branch Instructions
  setOperationAction(ISD::BR_CC,     XLenVT,     Expand);
  setOperationAction(ISD::BR_JT,     MVT::Other, Expand);
  // @} MYRISCVXTargetLowering_setOperationAction_Branch

  // @} MYRISCVXTargetLowering_setOperationAction_Select ...
  setOperationAction(ISD::SELECT,    XLenVT,     Custom);   // SELECTはカスタム関数を定義して生成する
  setOperationAction(ISD::SELECT_CC, XLenVT,     Expand);   // SELECT_CCは生成を抑制する
  // @} MYRISCVXTargetLowering_setOperationAction_Select

  // @} MYRISCVXTargetLowering_setOperationAction_vararg ...
  // 可変長引数のためのノード設定
  setOperationAction(ISD::VASTART,   MVT::Other, Custom);

  setOperationAction(ISD::VAARG,     MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,    MVT::Other, Expand);
  setOperationAction(ISD::VAEND,     MVT::Other, Expand);
  // @} MYRISCVXTargetLowering_setOperationAction_vararg

  // @{ MYRISCVXTargetLowering_setOperationAction_stack
  // Use the default for now
  setOperationAction(ISD::STACKSAVE,    MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  // @} MYRISCVXTargetLowering_setOperationAction_stack

  // @{ MYRISCVXTargetLowering_setMinimumJumpTableEntries
  // テーブルジャンプの生成を抑制するのには, 生成条件のエントリ数を最大にしておくと
  // 常に生成されなくなる
  setMinimumJumpTableEntries(INT_MAX);
  // @} MYRISCVXTargetLowering_setMinimumJumpTableEntries

  // Operations not directly supported by RISC-V
  setOperationAction(ISD::BR_CC,             MVT::f32,   Expand);
  setOperationAction(ISD::BR_CC,             MVT::f64,   Expand);
  setOperationAction(ISD::SELECT_CC,         MVT::f32,   Expand);
  setOperationAction(ISD::SELECT_CC,         MVT::f64,   Expand);

  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool, MVT::f32, Custom);
  setOperationAction(ISD::ConstantPool, MVT::f64, Custom);

  // RISCV doesn't have extending float->double load/store.  Set LoadExtAction
  // for f32, f16
  for (MVT VT : MVT::fp_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f32, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::f16, Expand);
  }
}
// @} MYRISCVXTargetLowering

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

// addLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned
addLiveIn(MachineFunction &MF, unsigned PReg, const TargetRegisterClass *RC)
{
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

#include "MYRISCVXGenCallingConv.inc"

// @{ MYRISCVXTargetLowering_LowerOperation
// @{ MYRISCVXTargetLowering_LowerOperation_SELECT
// @{ MYRISCVXTargetLowering_LowerOperation_VASTART
SDValue MYRISCVXTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  // SelectionDAGのノード種類をチェック
  switch (Op.getOpcode())
  {
    case ISD::ConstantPool : return lowerConstantPool(Op, DAG);
    // SELECTノードはカスタム関数で処理する
    case ISD::SELECT       : return lowerSELECT(Op, DAG);
// @} MYRISCVXTargetLowering_LowerOperation_SELECT
    // GlobalAddressノードであれば, lowerGlobalAddress()を呼び出す
    // GlobalAddressノードはあらかじめsetOperationAction()でカスタム処理を呼び出すように設定してある
    case ISD::GlobalAddress: return lowerGlobalAddress(Op, DAG);
    // VASTARTノードはカスタム関数で処理する
    case ISD::VASTART      : return lowerVASTART       (Op, DAG);
// @} MYRISCVXTargetLowering_LowerOperation_VASTART
  }
  return SDValue();
}
// @} MYRISCVXTargetLowering_LowerOperation


// @{ MYRISCVXTargetLowering_lowerGlobalAddress
// @{ MYRISCVXTargetLowering_lowerGlobalAddress_PIC
SDValue MYRISCVXTargetLowering::lowerGlobalAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  // @{ MYRISCVXTargetLowering_lowerGlobalAddress_Static
  if (!isPositionIndependent()) {
    // @{ MYRISCVXTargetLowering_lowerGlobalAddress_PIC ...
    // Staticモードの場合: %hi/%loへを挿入する
    SDValue Addr = getAddrStatic(N, Ty, DAG);
    if (Offset) {
      return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                         DAG.getConstant(Offset, DL, XLenVT));
    } else {
      return Addr;
    }
    // @} MYRISCVXTargetLowering_lowerGlobalAddress_PIC ...
  }
  // @} MYRISCVXTargetLowering_lowerGlobalAddress_Static
  // PICモードの場合：LA疑似命令を発行する
  SDValue Addr = getTargetNode(N, Ty, DAG, 0);
  return SDValue(DAG.getMachineNode(MYRISCVX::PseudoLA, DL, Ty, Addr), 0);
}
// @} MYRISCVXTargetLowering_lowerGlobalAddress_PIC
// @} MYRISCVXTargetLowering_lowerGlobalAddress

// @{ MYRISCVXTargetLowering_lowerSELECT
SDValue MYRISCVXTargetLowering::
lowerSELECT(SDValue Op, SelectionDAG &DAG) const
{
  // やっていることは, SELECTノードをMYRISCVXカスタムのSELECT_CCノードに置き換えている
  // LLVMでデフォルトで定義されているSELECT_CCノードとは異なるので注意
  // MYRISCVXISD::SELECT_CCから独自の生成パタンで命令に落とし込むための下準備
  SDValue CondV = Op.getOperand(0);    // 条件判定のための値
  SDValue TrueV = Op.getOperand(1);    // 条件がTrueの場合に選択される値
  SDValue FalseV = Op.getOperand(2);   // 条件がFalseの場合に選択される値
  SDLoc DL(Op);

  MVT XLenVT = Subtarget.getXLenVT();

  // (select condv, truev, falsev)
  // -> (MYRISCVXISD::SELECT_CC condv, zero, setne, truev, falsev)
  SDValue Zero = DAG.getConstant(0, DL, XLenVT);              // 定数ゼロのためのSelectionDAGノード
  SDValue SetNE = DAG.getConstant(ISD::SETNE, DL, XLenVT);    // NEQ演算のためのSelectionDAGノード

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  // (NEQ(条件値, ZERO), True値, False値)という引数リストが作成される
  SDValue Ops[] = {CondV, Zero, SetNE, TrueV, FalseV};

  // MYRISCVXISD::SELECT_CCの引数としてOpsが設定されこのDAGノードが返される
  return DAG.getNode(MYRISCVXISD::SELECT_CC, DL, VTs, Ops);
}
// @} MYRISCVXTargetLowering_lowerSELECT


SDValue MYRISCVXTargetLowering::lowerConstantPool(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  int64_t Offset = N->getOffset();
  MVT XLenVT = Subtarget.getXLenVT();

  if (!isPositionIndependent()) {
    SDValue Addr = getAddrStatic(N, Ty, DAG);
    if (Offset) {
      return DAG.getNode(ISD::ADD, DL, Ty, Addr,
                         DAG.getConstant(Offset, DL, XLenVT));
    } else {
      return Addr;
    }
  } else {
    SDValue Addr = getTargetNode(N, Ty, DAG, 0);
    return SDValue(DAG.getMachineNode(MYRISCVX::PseudoLA, DL, Ty, Addr), 0);
  }
}


// @{ MYRISCVXTargetLowering_getTargetNode_Global
SDValue MYRISCVXTargetLowering::getTargetNode(GlobalAddressSDNode *N, EVT Ty,
                                              SelectionDAG &DAG,
                                              unsigned Flag) const {
  return DAG.getTargetGlobalAddress(N->getGlobal(), SDLoc(N), Ty, 0, Flag);
}
// @} MYRISCVXTargetLowering_getTargetNode


// @{ MYRISCVXTargetLowering_getTargetNode_External
SDValue MYRISCVXTargetLowering::getTargetNode(ExternalSymbolSDNode *N, EVT Ty,
                                              SelectionDAG &DAG,
                                              unsigned Flag) const {
  return DAG.getTargetExternalSymbol(N->getSymbol(), Ty, Flag);
}
// @} MYRISCVXTargetLowering_getTargetNode_External


SDValue MYRISCVXTargetLowering::getTargetNode(ConstantPoolSDNode *N, EVT Ty,
                                              SelectionDAG &DAG,
                                              unsigned Flags) const {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flags);
}


// Return the RISC-V branch opcode that matches the given DAG integer
// condition code. The CondCode must be one of those supported by the RISC-V
// ISA (see normaliseSetCC).
// @{ MYRISCVXISelLowering_getBranchOpcodeForIntCondCode
unsigned MYRISCVXTargetLowering::getBranchOpcodeForIntCondCode (ISD::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unsupported CondCode");
  case ISD::SETEQ:
    return MYRISCVX::BEQ;
  case ISD::SETNE:
    return MYRISCVX::BNE;
  case ISD::SETLT:
    return MYRISCVX::BLT;
  case ISD::SETGE:
    return MYRISCVX::BGE;
  case ISD::SETULT:
    return MYRISCVX::BLTU;
  case ISD::SETUGE:
    return MYRISCVX::BGEU;
  }
}
// @} MYRISCVXISelLowering_getBranchOpcodeForIntCondCode


// @{ MYRISCVXISelLowering_emitSelectPseudo
MachineBasicBlock *MYRISCVXTargetLowering::
emitSelectPseudo(MachineInstr &MI,
                 MachineBasicBlock *BB) {
  // SELECT IRを変換するため関数
  // MIの入力オペランドには以下のデータが揃っている
  // MI.getOperand(0) : SELECT結果を書き込むレジスタ
  // MI.getOperand(1) : LHS比較用データ
  // MI.getOperand(2) : RHS比較用データ
  // MI.getOperand(3) : 比較演算
  // MI.getOperand(4) : 比較結果がTrue時のデータ
  // MI.getOperand(5) : 比較結果がFalse時のデータ

  // @{ MYRISCVXISelLowering_emitSelectPseudo ...

  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  DebugLoc DL = MI.getDebugLoc();
  MachineFunction::iterator I = ++BB->getIterator();

  MachineBasicBlock *HeadMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TailMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *IfFalseMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, IfFalseMBB);
  F->insert(I, TailMBB);

  TailMBB->splice(TailMBB->begin(), HeadMBB,
                  std::next(MachineBasicBlock::iterator(MI)), HeadMBB->end());

  // @} MYRISCVXISelLowering_emitSelectPseudo ...

  TailMBB->transferSuccessorsAndUpdatePHIs(HeadMBB);
  // HeadMBBはIfFalseMBBとTailMBBに繋がっている
  HeadMBB->addSuccessor(IfFalseMBB);
  HeadMBB->addSuccessor(TailMBB);

  // 比較演算の準備
  unsigned LHS = MI.getOperand(1).getReg();
  unsigned RHS = MI.getOperand(2).getReg();
  auto CC = static_cast<ISD::CondCode>(MI.getOperand(3).getImm());
  unsigned Opcode = getBranchOpcodeForIntCondCode(CC);

  // @{ MYRISCVXISelLowering_emitSelectPseudo_BuildMI
  // 比較演算用のMachineInstrを作る
  // LHSとrHSをOpcodeに基づき比較し, 比較が成立すればTailMBBに飛ぶ
  // HeadMBBにこの比較演算を取り付ける
  BuildMI(HeadMBB, DL, TII.get(Opcode))
    .addReg(LHS)
    .addReg(RHS)
    .addMBB(TailMBB);
  // @} MYRISCVXISelLowering_emitSelectPseudo_BuildMI

  // @{ MYRISCVXISelLowering_emitSelectPseudo_addSuccessor
  // IfFalseMBBに到達すると単純にTailMBBに再度ジャンプする
  IfFalseMBB->addSuccessor(TailMBB);
  // @} MYRISCVXISelLowering_emitSelectPseudo_addSuccessor

  // @{ MYRISCVXISelLowering_emitSelectPseudo_BuildMI2
  // PHI演算のポイント：HeadMBBからやってきた場合はTrueV(GetOperand(4)のデータ)を選択する
  //                    IfFalseBBからやってきた場合はFalseV(GetOperand(5)のデータ)を選択する
  //
  // %Result = phi [ %TrueValue, HeadMBB ], [ %FalseValue, IfFalseMBB ]
  BuildMI(*TailMBB, TailMBB->begin(), DL, TII.get(MYRISCVX::PHI),
          MI.getOperand(0).getReg())
      .addReg(MI.getOperand(4).getReg())
      .addMBB(HeadMBB)
      .addReg(MI.getOperand(5).getReg())
      .addMBB(IfFalseMBB);
  // @} MYRISCVXISelLowering_emitSelectPseudo_BuildMI2
  MI.eraseFromParent();
  return TailMBB;
}
// @} MYRISCVXISelLowering_emitSelectPseudo


// @{ MYRISCVXTargetLowering_getRegForInlineAsmConstraint
// インラインアセンブリにおいて、どの制約文字がレジスタを使用するのかを返す
std::pair<unsigned, const TargetRegisterClass *>
MYRISCVXTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                     StringRef Constraint,
                                                     MVT VT) const {
  // MYRISCVXのレジスタクラスを使用するものがあればそれを返す
  // MYRISCVX register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      // 識別文字'r'はMYRISCVXのGPRRegClassを使用する
      return std::make_pair(0U, &MYRISCVX::GPRRegClass);
    default:
      break;
    }
  }

  // それ以外の制約はデフォルトのものを使用する
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}
// @} MYRISCVXTargetLowering_getRegForInlineAsmConstraint


unsigned
MYRISCVXTargetLowering::getInlineAsmMemConstraint(StringRef ConstraintCode) const {
  // Currently only support length 1 constraints.
  if (ConstraintCode.size() == 1) {
    switch (ConstraintCode[0]) {
    case 'A':
      return InlineAsm::Constraint_A;
    default:
      break;
    }
  }

  return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
}

void MYRISCVXTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, std::string &Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  // Currently only support length 1 constraints.
  if (Constraint.length() == 1) {
    switch (Constraint[0]) {
    case 'I':
      // Validate & create a 12-bit signed immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getSExtValue();
        if (isInt<12>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    case 'J':
      // Validate & create an integer zero operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op))
        if (C->getZExtValue() == 0)
          Ops.push_back(
              DAG.getTargetConstant(0, SDLoc(Op), Subtarget.getXLenVT()));
      return;
    case 'K':
      // Validate & create a 5-bit unsigned immediate operand.
      if (auto *C = dyn_cast<ConstantSDNode>(Op)) {
        uint64_t CVal = C->getZExtValue();
        if (isUInt<5>(CVal))
          Ops.push_back(
              DAG.getTargetConstant(CVal, SDLoc(Op), Subtarget.getXLenVT()));
      }
      return;
    default:
      break;
    }
  }
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}


// @{ MYRISCVXISelLowering_EmitInstrWithCustomInserter
MachineBasicBlock *
MYRISCVXTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                    MachineBasicBlock *BB) const {

  switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected instr type to insert");
    case MYRISCVX::Select_GPR_Using_CC_GPR:
      return emitSelectPseudo(MI, BB);
  }
}
// @} MYRISCVXISelLowering_EmitInstrWithCustomInserter


// @{ MYRISCVXISelLowering_LowerCall
// @{ MYRISCVXISelLowering_LowerCall_GlobalAddressSDNode
// @{ MYRISCVXISellowering_LowerCall_TailCall
SDValue
MYRISCVXTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                  SmallVectorImpl<SDValue> &InVals) const {

  SelectionDAG &DAG                     = CLI.DAG;
  SDLoc DL                              = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals     = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  // @} MYRISCVXISellowering_LowerCall_TailCall ...
  bool &IsTailCall                      = CLI.IsTailCall;
  // @{ MYRISCVXISellowering_LowerCall_TailCall ...
  CallingConv::ID CallConv              = CLI.CallConv;
  bool IsVarArg                         = CLI.IsVarArg;

  // @{ MYRISCVXISelLowering_LowerCall_GlobalAddressSDNode ...

  EVT PtrVT = getPointerTy(DAG.getDataLayout());

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
  bool IsPIC = isPositionIndependent();

  // @{ MYRISCVXISelLowering_LowerCall_AnalyzeCallOperands
  // 関数呼び出しの引数解析を行い, 各オペランドの配置方法を決める
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands (Outs, CC_MYRISCVX);
  // @} MYRISCVXISelLowering_LowerCall_AnalyzeCallOperands

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NextStackOffset = CCInfo.getNextStackOffset();

  // 末尾最適化が本当に適用可能かチェックする
  if (IsTailCall)
    IsTailCall = isEligibleForTailCallOptimization(CCInfo, NextStackOffset,
                                                   *MF.getInfo<MYRISCVXFunctionInfo>());

  if (IsTailCall) {
    ++NumTailCalls;
  }
  // @} MYRISCVXISellowering_LowerCall_TailCall

  // Chain is the output chain of the last Load/Store or CopyToReg node.
  // ByValChain is the output chain of the last Memcpy node created for copying
  // byval arguments to the stack.
  unsigned StackAlignment = TFL->getStackAlignment();
  NextStackOffset = alignTo(NextStackOffset, StackAlignment);
  SDValue NextStackOffsetVal = DAG.getIntPtrConstant(NextStackOffset, DL, true);

  // @{ MYRISCVXISelLowering_LowerCall_getCALLSEQ_START
  // 関数呼び出しに必要な処理がここから開始されることを意味する
  if (!IsTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, NextStackOffset, 0, DL);
  // @} MYRISCVXISelLowering_LowerCall_getCALLSEQ_START

  SDValue StackPtr =
      DAG.getCopyFromReg(Chain, DL, MYRISCVX::SP, PtrVT);

  // With EABI is it possible to have 16 args on registers.
  std::deque< std::pair<unsigned, SDValue> > RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // @{ MYRISCVXISelLowering_LowerCall_ArgLocs_Loop
  // @{ MYRISCVXISelLowering_LowerCall_isMemLoc
  // レジスタ・メモリへの引数割り当てを行う
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    SDValue Arg = OutVals[i];
    CCValAssign &VA = ArgLocs[i];
    MVT LocVT = VA.getLocVT();

    // @{ MYRISCVXISelLowering_LowerCall_isMemLoc ...

    // 必要に応じて型を拡張するノードを挿入する
    switch (VA.getLocInfo()) {
      default: llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
      case CCValAssign::SExt:
        Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, LocVT, Arg);
        break;
      case CCValAssign::ZExt:
        Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, LocVT, Arg);
        break;
      case CCValAssign::AExt:
        Arg = DAG.getNode(ISD::ANY_EXTEND, DL, LocVT, Arg);
        break;
    }

    // @{ MYRISCVXISelLowering_LowerCall_isRegLoc
    // AnalyzeCallOperandsの結果レジスタに配置された引数は,
    // RegsToPassキューに積み上げていく
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }
    // @} MYRISCVXISelLowering_LowerCall_isRegLoc

    // @} MYRISCVXISelLowering_LowerCall_isMemLoc ...
    // ここにはメモリを経由する引数しかやってこないはず
    assert(VA.isMemLoc());

    // レジスタに入りきらない引数はMemOpChainsに積み上げていく
    // passArgOnStackでスタックに引数を積み上げるノードを作り上げる
    MemOpChains.push_back(passArgOnStack(StackPtr, VA.getLocMemOffset(),
                                         Chain, Arg, DL, IsTailCall, DAG));
    // @} MYRISCVXISelLowering_LowerCall_isMemLoc
  }
  // @} MYRISCVXISelLowering_LowerCall_ArgLocs_Loop

  // @{ MYRISCVXISelLowering_LowerCall_MemOpChains
  // TokenFactorノードを生成してストアノードを1つにまとめる
  // 引数をメモリに格納するノードを1つにまとめ、それぞれのノードが独立であるということを示す
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);
  // @} MYRISCVXISelLowering_LowerCall_MemOpChains

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  bool IsPICCall = IsPIC; // true if calls are translated to
  bool GlobalOrExternal = false, InternalLinkage = false;

  // @} MYRISCVXISelLowering_LowerCall_GlobalAddressSDNode ...
  if (GlobalAddressSDNode *S = dyn_cast<GlobalAddressSDNode>(Callee)) {
    // 関数呼び出しのターゲットがGlobalAddressSDNodeの場合
    const GlobalValue *GV = S->getGlobal();

    unsigned OpFlags = MYRISCVXII::MO_CALL;
    if (!getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV))
      OpFlags = MYRISCVXII::MO_CALL_PLT;

    // DAG.getTargetGlobalAddress()により関数呼び出し先を示すノードを生成する
    Callee = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0, OpFlags);
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    // 関数呼び出しのターゲットがExternalSymbolSDNodeの場合
    unsigned OpFlags = MYRISCVXII::MO_CALL;

    if (!getTargetMachine().shouldAssumeDSOLocal(*MF.getFunction().getParent(),
                                                 nullptr))
      OpFlags = MYRISCVXII::MO_CALL_PLT;

    // DAG.getTargetExternalSymbol()により関数呼び出し先を示すノードを生成する
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), PtrVT, OpFlags);
  }
  // @} MYRISCVXISelLowering_LowerCall_GlobalAddressSDNode

  SmallVector<SDValue, 8> Ops(1, Chain);
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // 引数をレジスタに置くためのDAGを生成する
  getOpndList(Ops, RegsToPass, IsPICCall, GlobalOrExternal, InternalLinkage,
              CLI, Callee, Chain);

  //@{ MYRISCVXISelLowering_LowerCall_IsTailCall_DAG_getNode
  // カスタムTailCallノードを挿入
  if (IsTailCall) {
    return DAG.getNode(MYRISCVXISD::TailCall, DL, MVT::Other, Ops);
  }
  //@} MYRISCVXISelLowering_LowerCall_IsTailCall_DAG_getNode

  // @{ MYRISCVXISelLowering_LowerCall_MYRISCVXISD_CALL
  Chain = DAG.getNode(MYRISCVXISD::CALL, DL, NodeTys, Ops);
  SDValue InFlag = Chain.getValue(1);

  // @{ MYRISCVXISelLowering_LowerCall_getCALLSEQ_END
  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, NextStackOffsetVal,
                             DAG.getIntPtrConstant(0, DL, true), InFlag, DL);
  // @} MYRISCVXISelLowering_LowerCall_getCALLSEQ_END
  InFlag = Chain.getValue(1);

  // 戻り値を処理するためのノード生成関数を呼び出す
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg,
                         Ins, DL, DAG, InVals, CLI.Callee.getNode(), CLI.RetTy);

  // @} MYRISCVXISelLowering_LowerCall_MYRISCVXISD_CALL
}
// @} MYRISCVXISelLowering_LowerCall


/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
// @{ MYRISCVXISelLowering_LowerCallResult
// @{ MYRISCVXISelLowering_LowerCallResult_Head
SDValue
MYRISCVXTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                        CallingConv::ID CallConv, bool IsVarArg,
                                        const SmallVectorImpl<ISD::InputArg> &Ins,
                                        const SDLoc &DL, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &InVals,
                                        const SDNode *CallNode,
                                        const Type *RetTy) const {
  // @} MYRISCVXISelLowering_LowerCallResult_Head
  // @{ MYRISCVXISelLowering_LowerCallResult_AnalyzeCallResult
  // 戻り値を取得するための場所を確認する. ABIに基づきどのレジスタを使用するか決定する
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 RVLocs, *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_MYRISCVX);
  // @} MYRISCVXISelLowering_LowerCallResult_AnalyzeCallResult

  // Copy all of the result registers out of their specified physreg.
  // @{ MYRISCVXISelLowering_LowerCallResult_Loop
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    // getCopyFromRegを用いて物理的なレジスタからDAGのノードに値を移す
    SDValue Val = DAG.getCopyFromReg(Chain, DL, RVLocs[i].getLocReg(),
                                     RVLocs[i].getLocVT(), InFlag);
    Chain = Val.getValue(1);
    InFlag = Val.getValue(2);

    if (RVLocs[i].getValVT() != RVLocs[i].getLocVT())
      Val = DAG.getNode(ISD::BITCAST, DL, RVLocs[i].getValVT(), Val);

    InVals.push_back(Val);
  }
  // @} MYRISCVXISelLowering_LowerCallResult_Loop

  return Chain;
}
// @} MYRISCVXISelLowering_LowerCallResult


// @{ MYRISCVXISelLowering_passArgOnStack
SDValue
MYRISCVXTargetLowering::passArgOnStack(SDValue StackPtr, unsigned Offset,
                                       SDValue Chain, SDValue Arg, const SDLoc &DL,
                                       bool IsTailCall, SelectionDAG &DAG) const {
  // 関数の引数をスタックに配置するために, オフセットを計算してストア命令を生成する
  if (!IsTailCall) {
    SDValue PtrOff =
        DAG.getNode(ISD::ADD, DL, getPointerTy(DAG.getDataLayout()), StackPtr,
                    DAG.getIntPtrConstant(Offset, DL));
    return DAG.getStore(Chain, DL, Arg, PtrOff, MachinePointerInfo());
  }

  MachineFrameInfo &MFI = DAG.getMachineFunction().getFrameInfo();
  int FI = MFI.CreateFixedObject(Arg.getValueSizeInBits() / 8, Offset, false);
  SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
  return DAG.getStore(Chain, DL, Arg, FIN, MachinePointerInfo(),
                      /* Alignment = */ 0, MachineMemOperand::MOVolatile);
}
// @} MYRISCVXISelLowering_passArgOnStack


// @{ MYRISCVXISelLowering_getOpndList
// @{ MYRISCVXISelLowering_getOpndList_Loop
void MYRISCVXTargetLowering::
getOpndList(SmallVectorImpl<SDValue> &Ops,
            std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
            bool IsPICCall, bool GlobalOrExternal, bool InternalLinkage,
            CallLoweringInfo &CLI, SDValue Callee, SDValue Chain) const {
  // 引数をレジスタに配置するためのDAGを生成する
  // getCopyToRegを使ってひたすらレジスタに置いていく
  // @{ MYRISCVXISelLowering_getOpndList_Loop ...
  Ops.push_back(Callee);

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
  // stuck together.
  SDValue InFlag;

  // @} MYRISCVXISelLowering_getOpndList_Loop ...
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = CLI.DAG.getCopyToReg(Chain, CLI.DL, RegsToPass[i].first,
                                 RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(CLI.DAG.getRegister(RegsToPass[i].first,
                                      RegsToPass[i].second.getValueType()));
  // @} MYRISCVXISelLowering_getOpndList_Loop
  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask =
      TRI->getCallPreservedMask(CLI.DAG.getMachineFunction(), CLI.CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(CLI.DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);
}
// @} MYRISCVXISelLowering_getOpndList


//===----------------------------------------------------------------------===//
//@            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//
// @{ MYRISCVXISelLowering_LowerFormalArguments
// @{ MYRISCVXISelLowering_LowerFormalArguments_Head
// @{ MYRISCVXISelLowering_LowerFormalArguments_IsVarArg
/// LowerFormalArguments()
// 引数渡しにおいて、引数を渡す方法を実装する
SDValue
MYRISCVXTargetLowering::LowerFormalArguments(SDValue Chain,
                                             CallingConv::ID CallConv,
                                             bool IsVarArg,
                                             const SmallVectorImpl<ISD::InputArg> &Ins,
                                             const SDLoc &DL, SelectionDAG &DAG,
                                             SmallVectorImpl<SDValue> &InVals)
// @} MYRISCVXISelLowering_LowerFormalArguments_Head
const {
  // @{ MYRISCVXISelLowering_LowerFormalArguments_IsVarArg ...
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MYRISCVXFunctionInfo *MYRISCVXFI = MF.getInfo<MYRISCVXFunctionInfo>();

  MYRISCVXFI->setVarArgsFrameIndex(0);

  // @{ MYRISCVXISelLowering_LowerFormalArguments_AnalyzeFormalarguments
  // 引数をレジスタに割り当てるのか, スタックに割り当てるのかを決定する
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments (Ins, CC_MYRISCVX);
  // @} MYRISCVXISelLowering_LowerFormalArguments_AnalyzeFormalarguments

  MYRISCVXFI->setFormalArgInfo(CCInfo.getNextStackOffset(),
                               CCInfo.getInRegsParamsCount() > 0);

  Function::const_arg_iterator FuncArg =
      DAG.getMachineFunction().getFunction().arg_begin();

  // スタックに引数を格納するためのストアチェインを格納するためのベクタ
  std::vector<SDValue> OutChains;

  unsigned CurArgIdx = 0;
  CCInfo.rewindByValRegsInfo();

  // @{ MYRISCVXISelLowering_LowerFormalArguments_Loop
  // @{ MYRISCVXISelLowering_LowerFormalArguments_RegLoc
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // @{ MYRISCVXISelLowering_LowerFormalArguments_RegLoc ...
    if (Ins[i].isOrigArg()) {
      std::advance(FuncArg, Ins[i].getOrigArgIndex() - CurArgIdx);
      CurArgIdx = Ins[i].getOrigArgIndex();
    }
    // @} MYRISCVXISelLowering_LowerFormalArguments_RegLoc ...
    // @} MYRISCVXISelLowering_LowerFormalArguments_Loop
    EVT ValVT = VA.getValVT();

    // @{ MYRISCVXISelLowering_LowerFormalArguments_RegLoc
    bool IsRegLoc = VA.isRegLoc();

    if (IsRegLoc) {
      // レジスタに引数を割り当てる場合：
      MVT RegVT = VA.getLocVT();
      unsigned ArgReg = VA.getLocReg();
      const TargetRegisterClass *RC = getRegClassFor(RegVT);

      // レジスタに配置された引数を, 仮想的な変数に乗せ換えるために
      // getCopyFromRegを使用する
      unsigned Reg = addLiveIn(DAG.getMachineFunction(), ArgReg, RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegVT);

      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue,
                                 DAG.getValueType(ValVT));
        ArgValue = DAG.getNode(ISD::TRUNCATE, DL, ValVT, ArgValue);
      }
      InVals.push_back(ArgValue);
      // @} MYRISCVXISelLowering_LowerFormalArguments_RegLoc
      // @{ MYRISCVXISelLowering_LowerFormalArguments_MemLoc
    } else {
      // スタックに引数を割り当てる場合：
      MVT LocVT = VA.getLocVT();

      // sanity check
      assert(VA.isMemLoc());

      // スタックポインタのオフセットはスタックフレームからの相対距離で計算される
      int FI = MFI.CreateFixedObject(ValVT.getSizeInBits()/8,
                                     VA.getLocMemOffset(), true);

      // スタックからの引数をロードするためのロード命令を生成する
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Load = DAG.getLoad(
          LocVT, DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
      InVals.push_back(Load);
      OutChains.push_back(Load.getValue(1));
    }
    // @} MYRISCVXISelLowering_LowerFormalArguments_MemLoc
  }

  // @} MYRISCVXISelLowering_LowerFormalArguments_IsVarArg ...
  // 可変長引数ならば, writeVarArgRegsに飛んで引数をスタックに格納していく
  if (IsVarArg)
    writeVarArgRegs(OutChains, Chain, DL, DAG, CCInfo);
  // @} MYRISCVXISelLowering_LowerFormalArguments_IsVarArg

  // 全ての引数スタック退避命令を1つのグループにまとめ上げる
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}
// @} MYRISCVXISelLowering_LowerFormalArguments


//===----------------------------------------------------------------------===//
//@              Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//
// @{ MYRISCVXISelLowering_LowerReturn
// @{ MYRISCVXISelLowering_LowerReturn_MYRISCVXRet
// LowerReturn()
// LLVM IRのreturn文をどのようにSelectionDAGに置き換えるかをここで指定する
// @{ MYRISCVXISelLowering_LowerReturn_Header
SDValue
MYRISCVXTargetLowering::LowerReturn(SDValue Chain,
                                    CallingConv::ID CallConv, bool IsVarArg,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SDLoc &DL, SelectionDAG &DAG) const {
  // @} MYRISCVXISelLowering_LowerReturn_Header
  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  MachineFunction &MF = DAG.getMachineFunction();


  // @{ MYRISCVXISelLowering_LowerReturn_AnalyzeReturn
  // CCStateにはレジスタとスタックスロットに関する情報が含まれる
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs,
                 *DAG.getContext());
  // AnalyzeReturn()を使ってRetCC_MYRISCVXのルールで戻り値のレジスタ割り当てを行う
  CCInfo.AnalyzeReturn(Outs, RetCC_MYRISCVX);
  // @} MYRISCVXISelLowering_LowerReturn_AnalyzeReturn

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // @{ MYRISCVXISelLowering_LowerReturn_Loop
  // 戻り値の値を戻り値用のレジスタにコピーするためのDAGを生成
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // @{ MYRISCVXISelLowering_LowerReturn_BITCAST
    // 小さなサイズの型の場合は拡張のためのノード挿入
    if (RVLocs[i].getValVT() != RVLocs[i].getLocVT())
      Val = DAG.getNode(ISD::BITCAST, DL, RVLocs[i].getLocVT(), Val);
    // @} MYRISCVXISelLowering_LowerReturn_BITCAST

    // @{ MYRISCVXISelLowering_LowerReturn_getCopyToReg
    // 戻り値のノードをCopyToRegでレジスタと結合
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Flag);
    // @} MYRISCVXISelLowering_LowerReturn_getCopyToReg

    Flag = Chain.getValue(1);
    // RetOpsに戻り値レジスタの情報を追加する
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // RetOpsの先頭はこれまでに作成したgetCopyRegsのチェインを接続する
  // それ以降のRetOpsの要素は戻り値レジスタに関する情報が含まれている
  RetOps[0] = Chain;  // Update chain.

  if (Flag.getNode())
    RetOps.push_back(Flag);
  // @} MYRISCVXISelLowering_LowerReturn_Loop


  // @{ MYRISCVXISelLowering_LowerReturn_RET
  // 戻り値を含むノードRetOpsを引数としたMYRISCVXISD::Retノードを作成する
  return DAG.getNode(MYRISCVXISD::Ret, DL, MVT::Other, RetOps);
  // @} MYRISCVXISelLowering_LowerReturn_RET
}
// @} MYRISCVXISelLowering_LowerReturn_MYRISCVXRet
// @} MYRISCVXISelLowering_LowerReturn


// @{ MYRISCVXISelLowering_lowerVASTART
SDValue MYRISCVXTargetLowering::lowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MYRISCVXFunctionInfo *FuncInfo = MF.getInfo<MYRISCVXFunctionInfo>();

  SDLoc DL = SDLoc(Op);
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy(MF.getDataLayout()));

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), DL, FI, Op.getOperand(1),
                      MachinePointerInfo(SV));
}
// @} MYRISCVXISelLowering_lowerVASTART


// @{ MYRISCVXISelLowering_writeVarArgRegs
// @{ MYRISCVXISelLowering_writeVarArgRegs_Loop
void MYRISCVXTargetLowering::writeVarArgRegs(std::vector<SDValue> &OutChains,
                                             SDValue Chain, const SDLoc &DL,
                                             SelectionDAG &DAG,
                                             CCState &State) const {
  ArrayRef<MCPhysReg> ArgRegs = ABI.GetVarArgRegs();
  unsigned Idx = State.getFirstUnallocated(ArgRegs);
  unsigned RegSizeInBytes = Subtarget.getGPRSizeInBytes();
  MVT RegTy = MVT::getIntegerVT(RegSizeInBytes * 8);
  const TargetRegisterClass *RC = getRegClassFor(RegTy);
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MYRISCVXFunctionInfo *MYRISCVXFI = MF.getInfo<MYRISCVXFunctionInfo>();

  // @{ MYRISCVXISelLowering_writeVarArgRegs_Loop ...

  // Offset of the first variable argument from stack pointer.
  int VaArgOffset;

  if (ArgRegs.size() == Idx)
    VaArgOffset = alignTo(State.getNextStackOffset(), RegSizeInBytes);
  else {
    VaArgOffset =
        (int)ABI.GetCalleeAllocdArgSizeInBytes(State.getCallingConv()) -
        (int)(RegSizeInBytes * (ArgRegs.size() - Idx));
  }

  // Record the frame index of the first variable argument
  // which is a value necessary to VASTART.
  int FI = MFI.CreateFixedObject(RegSizeInBytes, VaArgOffset, true);
  MYRISCVXFI->setVarArgsFrameIndex(FI);

  // @} MYRISCVXISelLowering_writeVarArgRegs_Loop ...
  for (unsigned I = Idx; I < ArgRegs.size();
       ++I, VaArgOffset += RegSizeInBytes) {

    // レジスタを経由して渡された引数をすべてスタックに積み上げていく
    LLVM_DEBUG(dbgs() << "writeVarArgRegs I = " << I << '\n');

    unsigned Reg = addLiveIn(MF, ArgRegs[I], RC);
    SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
    FI = MFI.CreateFixedObject(RegSizeInBytes, VaArgOffset, true);
    SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue Store =
        DAG.getStore(Chain, DL, ArgValue, PtrOff, MachinePointerInfo());
    cast<StoreSDNode>(Store.getNode())->getMemOperand()->setValue(
        (Value *)nullptr);
    OutChains.push_back(Store);
  }
}
// @} MYRISCVXISelLowering_writeVarArgRegs_Loop
// @} MYRISCVXISelLowering_writeVarArgRegs


/// isEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization.
// @{ MYRISCVXISelLowering_isEligibleForTailCallOptimization
bool MYRISCVXTargetLowering::
isEligibleForTailCallOptimization(
    CCState &CCInfo,
    unsigned NextStackOffset, const MYRISCVXFunctionInfo& FI) const {

  // ByVal引数（構造体など）を持っている場合は適用不可能
  if (CCInfo.getInRegsParamsCount() > 0 || FI.hasByvalArg())
    return false;

  // 呼び出し先の関数のスタックサイズが呼び出し元のスタックサイズよりも大きい場合は
  // 適用不可能
  return NextStackOffset <= FI.getIncomingArgSize();
}
// @} MYRISCVXISelLowering_isEligibleForTailCallOptimization
