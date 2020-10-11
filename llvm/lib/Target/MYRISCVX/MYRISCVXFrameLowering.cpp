//===-- MYRISCVXFrameLowering.cpp - MYRISCVX Frame Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXFrameLowering.h"

#include "MYRISCVXInstrInfo.h"
#include "MYRISCVXMachineFunction.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

// @{ MYRISCVXFrameLowering_emitPrologue
void MYRISCVXFrameLowering::emitPrologue(MachineFunction &MF,
                                         MachineBasicBlock &MBB) const {
  // 関数のプロローグを生成する
  // ここではまだ何も実装しない
}
// @} MYRISCVXFrameLowering_emitPrologue

// @{ MYRISCVXFrameLowering_emitEpilogue
void MYRISCVXFrameLowering::emitEpilogue(MachineFunction &MF,
                                         MachineBasicBlock &MBB) const {
  // 関数のエピローグを生成する
  // ここではまだ何も実装しない
}
// @} MYRISCVXFrameLowering_emitEpilogue


// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas,
// if it needs dynamic stack realignment, if frame pointer elimination is
// disabled, or if the frame address is taken.
bool MYRISCVXFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
      MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken() ||
      TRI->needsStackRealignment(MF);
}
