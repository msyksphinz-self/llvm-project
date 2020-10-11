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
    default:                         return NULL;
  }
}


// @{ MYRISCVXTargetLowering
MYRISCVXTargetLowering::MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                               const MYRISCVXSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI), ABI(TM.getABI()) {
  MVT XLenVT = Subtarget.getXLenVT();

  // レジスタクラスをセットアップする
  addRegisterClass(XLenVT, &MYRISCVX::GPRRegClass);

  // 関数の配置アライメント 関数は4バイトアラインに配置する
  setMinFunctionAlignment(Align(4));

  // 全てのレジスタを宣言すると、computeRegisterProperties()を呼び出さなければならない
  computeRegisterProperties(STI.getRegisterInfo());
}
// @} MYRISCVXTargetLowering

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

#include "MYRISCVXGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//@            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//
// @{ MYRISCVXISelLowering_LowerFormalArguments
/// LowerFormalArguments()
// 引数渡しにおいて、引数を渡す方法を実装する
SDValue
MYRISCVXTargetLowering::LowerFormalArguments(SDValue Chain,
                                             CallingConv::ID CallConv,
                                             bool IsVarArg,
                                             const SmallVectorImpl<ISD::InputArg> &Ins,
                                             const SDLoc &DL, SelectionDAG &DAG,
                                             SmallVectorImpl<SDValue> &InVals)
const {
  // ここではまだ何も実装しない
  // 多くの場合は、
  //  - 引数を受け取った物理レジスタを仮想的なレジスタに移す作業
  //  - スタックを経由した引数を仮想的なレジスタに移す作業
  // が含まれる
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
SDValue
MYRISCVXTargetLowering::LowerReturn(SDValue Chain,
                                    CallingConv::ID CallConv, bool IsVarArg,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SDLoc &DL, SelectionDAG &DAG) const {
   // MYRISCVXISD::Retノードを生成し、ノードには
   //  - 戻り値を格納しているChain
   //  - 戻りアドレスを格納しているRAレジスタ
   // を接続する
   return DAG.getNode(MYRISCVXISD::Ret, DL, MVT::Other,
                      Chain, DAG.getRegister(MYRISCVX::RA, Subtarget.getXLenVT()));
}
// @} MYRISCVXISelLowering_LowerReturn_MYRISCVXRet
// @} MYRISCVXISelLowering_LowerReturn
