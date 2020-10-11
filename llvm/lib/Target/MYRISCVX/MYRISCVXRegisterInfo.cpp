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
void MYRISCVXRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    unsigned FIOperandNum, RegScavenger *RS) const {
  // フレームインデックスを削除するための関数
  // ここではまだ実装しない
}
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
