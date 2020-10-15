//===-- MYRISCVXRegisterInfo.cpp - MYRISCVX Register Information -== ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"
#include "MYRISCVXRegisterInfo.h"
#include "MYRISCVXSubtarget.h"
#include "MYRISCVXMachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "MYRISCVXGenRegisterInfo.inc"


MYRISCVXRegisterInfo::MYRISCVXRegisterInfo(const MYRISCVXSubtarget &ST, unsigned HwMode)
    : MYRISCVXGenRegisterInfo(MYRISCVX::RA, /*DwarfFlavour*/0, /*EHFlavor*/0,
                              /*PC*/0, HwMode), Subtarget(ST) {}

const TargetRegisterClass *
MYRISCVXRegisterInfo::intRegClass(unsigned Size) const {
  return &MYRISCVX::GPRRegClass;
}


//===----------------------------------------------------------------------===//
// Callee Saved Registers methods
//===----------------------------------------------------------------------===//
// @{ MYRISCVXRegisterInfo_getCalleeSavedRegs
const MCPhysReg *
MYRISCVXRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  // MYRISCVXCallingConv.tdで定義したCSR_LP32のリストをそのまま返せば良い
  return CSR_LP32_SaveList;
}
// @} MYRISCVXRegisterInfo_getCalleeSavedRegs

// @{ MYRISCVXRegisterInfo_getCallPreservedMask
const uint32_t *
MYRISCVXRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                           CallingConv::ID) const {
  // MYRISCVXCallingConv.tdで定義したCSR_LP32のリストをそのまま返せば良い
  return CSR_LP32_RegMask;
}
// @} MYRISCVXRegisterInfo_getCallPreservedMask

// @{ MYRISCVXRegisterInfo_getReservedRegs
BitVector MYRISCVXRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  // zero, ra, fp, sp, gp, tpはシステムが使用する予約済みレジスタなので
  // レジスタ割り当てに使用しない
  static const uint16_t ReservedCPURegs[] = {
    MYRISCVX::ZERO, MYRISCVX::RA, MYRISCVX::FP, MYRISCVX::SP, MYRISCVX::GP, MYRISCVX::TP
  };
  BitVector Reserved(getNumRegs());

  for (unsigned I = 0; I < array_lengthof(ReservedCPURegs); ++I)
    Reserved.set(ReservedCPURegs[I]);

  return Reserved;
}
// @} MYRISCVXRegisterInfo_getReservedRegs

// @{ MYRISCVXRegisterInfo_eliminateFrameIndex
// @{ MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented
// eliminateFrameIndexでは関数内での変数参照に使用される仮想的なオフセットを
// 具体的なオフセット計算に置き換える
void MYRISCVXRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    unsigned FIOperandNum, RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo MFI = MF.getFrameInfo();

  // @{ eliminateFrameIndex_FindFrameIndex
  // フレームインデックスを使用しているオペランドを探す
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }
  // @} eliminateFrameIndex_FindFrameIndex

  // @{ MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...
  LLVM_DEBUG(errs() << "\nFunction : " << MF.getFunction().getName() << "\n";
             errs() << "<--------->\n" << MI);
  // @} MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...

  // @{ eliminateFrameIndex_getOffset
  int FrameIndex = MI.getOperand(i).getIndex();
  // スタックフレームのサイズと, フレームインデックスからのオフセットを取得
  uint64_t stackSize = MF.getFrameInfo().getStackSize();
  int64_t spOffset = MF.getFrameInfo().getObjectOffset(FrameIndex);
  // @} eliminateFrameIndex_getOffset

  // @{ MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...
  LLVM_DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
             << "spOffset   : " << spOffset << "\n"
             << "stackSize  : " << stackSize << "\n");
  // @} MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...

  // スタックフレーム内の変数はSPレジスタを基準にして参照する
  unsigned FrameReg = MYRISCVX::SP;

  // スタックポインタからのオフセットを計算する
  // @{ eliminateFrameIndex_calcOffset
  int64_t Offset;
  Offset  = spOffset + (int64_t)stackSize;
  Offset += MI.getOperand(i+1).getImm();
  // @} eliminateFrameIndex_calcOffset

  // @{ MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...
  LLVM_DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  if (!MI.isDebugValue() && !isInt<12>(Offset)) {
	assert("(!MI.isDebugValue() && !isInt<16>(Offset))");
  }
  // @} MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented ...

  // @{ eliminateFrameIndex_changeMI
  // スタックフレームの参照をスタックフレームと新たに計算したオフセットに置き換える
  MI.getOperand(i+0).ChangeToRegister(FrameReg, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
  // @} eliminateFrameIndex_changeMI
}
// @} MYRISCVXRegisterInfo_eliminateFrameIndex_Implemented
// @} MYRISCVXRegisterInfo_eliminateFrameIndex

// @{ MYRISCVXRegisterInfo_requiresRegisterScavenging
bool
MYRISCVXRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}
// @} MYRISCVXRegisterInfo_requiresRegisterScavenging


// @{ MYRISCVXRegisterInfo_trackLivenessAfterRegAlloc
bool
MYRISCVXRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  return true;
}
// @} MYRISCVXRegisterInfo_trackLivenessAfterRegAlloc


// @{ MYRISCVXRegisterInfo_getFrameRegister
Register MYRISCVXRegisterInfo::
getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  return TFI->hasFP(MF) ? (MYRISCVX::FP) :
      (MYRISCVX::SP);
}
// @} MYRISCVXRegisterInfo_getFrameRegister
