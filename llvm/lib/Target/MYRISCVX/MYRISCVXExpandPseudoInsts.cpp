//===-- MYRISCVXExpandPseudoInsts.cpp - Expand pseudo instructions
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"
#include "MYRISCVXInstrInfo.h"
#include "MYRISCVXTargetMachine.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define MYRISCVX_EXPAND_PSEUDO_NAME "MYRISCVX pseudo instruction expansion pass"

namespace {

// @{ MYRISCVXExpandPseudoInsts_cpp_MYRISCVXExpandPseudo
class MYRISCVXExpandPseudo : public MachineFunctionPass {
public:
  const MYRISCVXInstrInfo *TII;
  static char ID;

  MYRISCVXExpandPseudo() : MachineFunctionPass(ID) {
    initializeMYRISCVXExpandPseudoPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return MYRISCVX_EXPAND_PSEUDO_NAME;
  }

// @} MYRISCVXExpandPseudoInsts_cpp_MYRISCVXExpandPseudo
 private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);

  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadAddress(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         MachineBasicBlock::iterator &NextMBBI);

  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI,
                           unsigned FlagsHi, unsigned SecondOpcode);
};

char MYRISCVXExpandPseudo::ID = 0;

bool MYRISCVXExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const MYRISCVXInstrInfo *>(MF.getSubtarget().getInstrInfo());
  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);
  return Modified;
}

bool MYRISCVXExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

// @{ MYRISCVXExpandPseudo_cpp_expandMI
bool MYRISCVXExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
    case MYRISCVX::PseudoLLA:
      return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
    case MYRISCVX::PseudoLA:
      return expandLoadAddress(MBB, MBBI, NextMBBI);
  }

  return false;
}
// @} MYRISCVXExpandPseudo_cpp_expandMI


// @{ MYRISCVXExpandPseudo_cpp_expandLoadLocalAddress
bool MYRISCVXExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, MYRISCVXII::MO_PCREL_HI20,
                             MYRISCVX::ADDI);
}
// @} MYRISCVXExpandPseudo_cpp_expandLoadLocalAddress

bool MYRISCVXExpandPseudo::expandLoadAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();

  unsigned SecondOpcode;
  unsigned FlagsHi;
  if (MF->getTarget().isPositionIndependent()) {
    const auto &STI = MF->getSubtarget<MYRISCVXSubtarget>();
    SecondOpcode = STI.is64Bit() ? MYRISCVX::LD : MYRISCVX::LW;
    FlagsHi = MYRISCVXII::MO_GOT_HI20;
  } else {
    SecondOpcode = MYRISCVX::ADDI;
    FlagsHi = MYRISCVXII::MO_PCREL_HI20;
  }
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, FlagsHi, SecondOpcode);
}


// @{ MYRISCVXExpandPseudo_cpp_expandAuipcInstPair
bool MYRISCVXExpandPseudo::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  // @{ MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...
  Register DestReg = MI.getOperand(0).getReg();
  const MachineOperand &Symbol = MI.getOperand(1);

  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  // @} MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...

  // Tell AsmPrinter that we unconditionally want the symbol of this label to be
  // emitted.
  NewMBB->setLabelMustBeEmitted();

  MF->insert(++MBB.getIterator(), NewMBB);

  BuildMI(NewMBB, DL, TII->get(MYRISCVX::AUIPC), DestReg)
      .addDisp(Symbol, 0, FlagsHi);
  BuildMI(NewMBB, DL, TII->get(SecondOpcode), DestReg)
      .addReg(DestReg)
      .addMBB(NewMBB, MYRISCVXII::MO_PCREL_LO12_I);

  // @{ MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...
  // Move all the rest of the instructions to NewMBB.
  NewMBB->splice(NewMBB->end(), &MBB, std::next(MBBI), MBB.end());
  // Update machine-CFG edges.
  NewMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  // Make the original basic block fall-through to the new.
  MBB.addSuccessor(NewMBB);

  // @} MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *NewMBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();
  return true;
}
// @} MYRISCVXExpandPseudo_cpp_expandAuipcInstPair


}

INITIALIZE_PASS(MYRISCVXExpandPseudo, "myriscvx-expand-pseudo",
                MYRISCVX_EXPAND_PSEUDO_NAME, false, false)
namespace llvm {

// @{ MYRISCVXExpandPseudo_cpp_createMYRISCVXExpandPseudoPass
FunctionPass *createMYRISCVXExpandPseudoPass() {
  return new MYRISCVXExpandPseudo();
}
// @} MYRISCVXExpandPseudo_cpp_createMYRISCVXExpandPseudoPass

} // end of namespace llvm
