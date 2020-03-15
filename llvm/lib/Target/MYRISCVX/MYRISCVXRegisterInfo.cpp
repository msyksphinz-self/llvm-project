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
// @{MYRISCVXRegisterInfo_getCalleeSavedRegs
const MCPhysReg *
MYRISCVXRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_LP32_SaveList;
}
// @}MYRISCVXRegisterInfo_getCalleeSavedRegs

// @{MYRISCVXRegisterInfo_getCallPreservedMask
const uint32_t *
MYRISCVXRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                           CallingConv::ID) const {
  return CSR_LP32_RegMask;
}
// @}MYRISCVXRegisterInfo_getCallPreservedMask

// @{MYRISCVXRegisterInfo_getReservedRegs
BitVector MYRISCVXRegisterInfo::
getReservedRegs(const MachineFunction &MF) const {
  static const uint16_t ReservedCPURegs[] = {
    MYRISCVX::ZERO, MYRISCVX::RA, MYRISCVX::FP, MYRISCVX::SP, MYRISCVX::GP, MYRISCVX::TP
  };
  BitVector Reserved(getNumRegs());

  for (unsigned I = 0; I < array_lengthof(ReservedCPURegs); ++I)
    Reserved.set(ReservedCPURegs[I]);
  Reserved.set(MYRISCVX::GP);

  return Reserved;
}
// @}MYRISCVXRegisterInfo_getReservedRegs

// @{MYRISCVXRegisterInfo_eliminateFrameIndex
void MYRISCVXRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj,
                    unsigned FIOperandNum, RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo MFI = MF.getFrameInfo();

  // @{eliminateFrameIndex_FindFrameIndex
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }
  // @}eliminateFrameIndex_FindFrameIndex

  LLVM_DEBUG(errs() << "\nFunction : " << MF.getFunction().getName() << "\n";
             errs() << "<--------->\n" << MI);

  // @{eliminateFrameIndex_getOffset
  int FrameIndex = MI.getOperand(i).getIndex();
  uint64_t stackSize = MF.getFrameInfo().getStackSize();
  int64_t spOffset = MF.getFrameInfo().getObjectOffset(FrameIndex);
  // @}eliminateFrameIndex_getOffset

  LLVM_DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
             << "spOffset   : " << spOffset << "\n"
             << "stackSize  : " << stackSize << "\n");

  // const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  // int MinCSFI = 0;
  // int MaxCSFI = -1;
  //
  // if (CSI.size()) {
  //   MinCSFI = CSI[0].getFrameIdx();
  //   MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  // }

  // The following stack frame objects are always referenced relative to $sp:
  //  1. Outgoing arguments.
  //  2. Pointer to dynamically allocated stack space.
  //  3. Locations for callee-saved registers.
  // Everything else is referenced relative to whatever register
  // getFrameRegister() returns.
  unsigned FrameReg = MYRISCVX::SP;

  // Calculate final offset.
  // - There is no need to change the offset if the frame object is one of the
  //   following: an outgoing argument, pointer to a dynamically allocated
  //   stack space or a $gp restore location,
  // - If the frame object is any of the following, its offset must be adjusted
  //   by adding the size of the stack:
  //   incoming argument, callee-saved register location or local variable.
  // @{eliminateFrameIndex_calcOffset
  int64_t Offset;
  Offset  = spOffset + (int64_t)stackSize;
  Offset += MI.getOperand(i+1).getImm();
  // @}eliminateFrameIndex_calcOffset

  LLVM_DEBUG(errs() << "Offset     : " << Offset << "\n" << "<--------->\n");

  // If MI is not a debug value, make sure Offset fits in the 12-bit immediate
  // field.
  if (!MI.isDebugValue() && !isInt<12>(Offset)) {
	assert("(!MI.isDebugValue() && !isInt<16>(Offset))");
  }

  // @{eliminateFrameIndex_changeMI
  MI.getOperand(i+0).ChangeToRegister(FrameReg, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
  // @}eliminateFrameIndex_changeMI
}
// @}MYRISCVXRegisterInfo_eliminateFrameIndex

// @{MYRISCVXRegisterInfo_requiresRegisterScavenging
bool
MYRISCVXRegisterInfo::requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}
// @}MYRISCVXRegisterInfo_requiresRegisterScavenging

bool
MYRISCVXRegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  return true;
}

// @{MYRISCVXRegisterInfo_getFrameRegister
unsigned MYRISCVXRegisterInfo::
getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  return TFI->hasFP(MF) ? (MYRISCVX::FP) :
      (MYRISCVX::SP);
}
// @}MYRISCVXRegisterInfo_getFrameRegister
