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

const char *MYRISCVXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    case MYRISCVXISD::TailCall:          return "MYRISCVXISD::TailCall";
    case MYRISCVXISD::Hi:                return "MYRISCVXISD::Hi";
    case MYRISCVXISD::Lo:                return "MYRISCVXISD::Lo";
    case MYRISCVXISD::GPRel:             return "MYRISCVXISD::GPRel";
    case MYRISCVXISD::Ret:               return "MYRISCVXISD::Ret";
    case MYRISCVXISD::EH_RETURN:         return "MYRISCVXISD::EH_RETURN";
    case MYRISCVXISD::DivRem:            return "MYRISCVXISD::DivRem";
    case MYRISCVXISD::DivRemU:           return "MYRISCVXISD::DivRemU";
    case MYRISCVXISD::Wrapper:           return "MYRISCVXISD::Wrapper";
    default:                             return NULL;
  }
}


// @{ MYRISCVXTargetLowering
//@{ MYRISCVXTargetLowering_setOperationAction_DontGenerate
//@{ MYRISCVXTargetLowering_setOperationAction_GlobalAddress
// @{ MYRISCVXTargetLowering_setOperationAction_Select
// @{ MYRISCVXTargetLowering_setOperationAction_Branch
MYRISCVXTargetLowering::MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                               const MYRISCVXSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI), ABI(TM.getABI()) {
  //@{ MYRISCVXTargetLowering_setOperationAction_DontGenerate ...
  //@{ MYRISCVXTargetLowering_setOperationAction_GlobalAddress ...
  // @{ MYRISCVXTargetLowering_setOperationAction_Branch ...
  // @{ MYRISCVXTargetLowering_setOperationAction_Select ...

  MVT XLenVT = Subtarget.getXLenVT();

  // レジスタクラスをセットアップする
  addRegisterClass(XLenVT, &MYRISCVX::GPRRegClass);

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

  // @{ MYRISCVXTargetLowering_setMinimumJumpTableEntries
  // テーブルジャンプの生成を抑制するのには, 生成条件のエントリ数を最大にしておくと
  // 常に生成されなくなる
  setMinimumJumpTableEntries(INT_MAX);
  // @} MYRISCVXTargetLowering_setMinimumJumpTableEntries
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
SDValue MYRISCVXTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  // SelectionDAGのノード種類をチェック
  switch (Op.getOpcode())
  {
    // GlobalAddressノードであれば, lowerGlobalAddress()を呼び出す
    // GlobalAddressノードはあらかじめsetOperationAction()でカスタム処理を呼び出すように設定してある
    case ISD::GlobalAddress: return lowerGlobalAddress(Op, DAG);
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


//===----------------------------------------------------------------------===//
//@            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//
// @{ MYRISCVXISelLowering_LowerFormalArguments
// @{ MYRISCVXISelLowering_LowerFormalArguments_Head
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

  // Copy the result values into the output registers.
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

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  // @{ MYRISCVXISelLowering_LowerReturn_RET
  // 戻り値を含むノードRetOpsを引数としたMYRISCVXISD::Retノードを作成する
  return DAG.getNode(MYRISCVXISD::Ret, DL, MVT::Other, RetOps);
  // @} MYRISCVXISelLowering_LowerReturn_RET
}
// @} MYRISCVXISelLowering_LowerReturn_MYRISCVXRet
// @} MYRISCVXISelLowering_LowerReturn
