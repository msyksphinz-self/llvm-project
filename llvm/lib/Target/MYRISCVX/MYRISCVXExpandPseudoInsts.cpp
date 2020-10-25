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
// MYRISCVXExpandPseudoはMachineFunctionPassを継承して定義する
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
// 疑似命令LA/LLAに遭遇するとこれを置き換える
bool MYRISCVXExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  switch (MBBI->getOpcode()) {
    case MYRISCVX::PseudoLLA:
      // LLAの場合はexpandLoadLocalAddress()を呼び出す
      return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
    case MYRISCVX::PseudoLA:
      // LAの場合はexpandLoadAddress()を呼び出す
      return expandLoadAddress(MBB, MBBI, NextMBBI);
  }

  return false;
}
// @} MYRISCVXExpandPseudo_cpp_expandMI


// @{ MYRISCVXExpandPseudo_cpp_expandLoadLocalAddress
bool MYRISCVXExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  // expandAuipcInstPair()ではAUIPC + ADDI命令を生成する
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, MYRISCVXII::MO_PCREL_HI20,
                             MYRISCVX::ADDI);
}
// @} MYRISCVXExpandPseudo_cpp_expandLoadLocalAddress


// @{ MYRISCVXExpandPseudo_cpp_expandLoadAddress
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

  // PICの場合にはexpandAuipcInstPair()でAUIPC + LD/LW(GOT)命令を生成する
  // staticの場合にはexpandAuipcInstPair()でAUIPC + ADDI命令を生成する
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, FlagsHi, SecondOpcode);
}
// @} MYRISCVXExpandPseudo_cpp_expandLoadAddress


// @{ MYRISCVXExpandPseudo_cpp_expandAuipcInstPair
// AUIPCともう一つの命令のペアを生成する
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

  NewMBB->setLabelMustBeEmitted();

  MF->insert(++MBB.getIterator(), NewMBB);

  // 1命令目はAUIPCを生成する
  BuildMI(NewMBB, DL, TII->get(MYRISCVX::AUIPC), DestReg)
      .addDisp(Symbol, 0, FlagsHi);
  // 2命令目は引数に基づいて命令を生成する
  BuildMI(NewMBB, DL, TII->get(SecondOpcode), DestReg)
      .addReg(DestReg)
      .addMBB(NewMBB, MYRISCVXII::MO_PCREL_LO12_I);

  // @{ MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...
  NewMBB->splice(NewMBB->end(), &MBB, std::next(MBBI), MBB.end());
  NewMBB->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(NewMBB);

  // @} MYRISCVXExpandPseudo_cpp_expandAuipcInstPair ...

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
