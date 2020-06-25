//===-- MYRISCVXDeleteRedundantJmp.cpp - MYRISCVX Delete Redundant Jmp ----===//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fills delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

#define DEBUG_TYPE "myriscvx-delete-redundant-jmp"

STATISTIC(NumMYRISCVXDeleteRedundantJmp, "Number of useless jmp deleted");

static cl::opt<bool> EnableMYRISCVXDeleteRedundantJmp(
    "enable-MYRISCVX-del-useless-jmp",
    cl::init(true),
    cl::desc("Delete useless jmp instructions: jmp 0."),
    cl::Hidden);

namespace {

class MYRISCVXDeleteRedundantJmp : public MachineFunctionPass {
  static char ID;

 public:
  MYRISCVXDeleteRedundantJmp() : MachineFunctionPass(ID) { }

  virtual StringRef getPassName() const {
    return "MYRISCVX Del Useless jmp";
  }

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB, MachineBasicBlock &MBBN);

  bool runOnMachineFunction(MachineFunction &F) {
    bool Changed = false;
    if (EnableMYRISCVXDeleteRedundantJmp) {
      MachineFunction::iterator FJ = F.begin();
      if (FJ != F.end())
        FJ++;
      if (FJ == F.end())
        return Changed;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FJ != FE; ++FI, ++FJ)
        // In STL style, F.end() is the dummy BasicBlock() like '\0' in
        // C string.
        // FJ is the next BasicBlock of FI; When FI range from F.begin() to
        // the PreviousBasicBlock of F.end() call runOnMachineBasicBlock().
        Changed |= runOnMachineBasicBlock(*FI, *FJ);
    }
    return Changed;
  }

};

char MYRISCVXDeleteRedundantJmp::ID = 0;
} // end of anonymous namespace


bool MYRISCVXDeleteRedundantJmp::runOnMachineBasicBlock(MachineBasicBlock &MBB, MachineBasicBlock &MBBN) {
  bool Changed = false;
  MachineBasicBlock::iterator I = MBB.end();
  if (I != MBB.begin())
    I--; // set I to the last instruction
  else
    return Changed;
  if (I->getOpcode() == MYRISCVX::PseudoBR && I->getOperand(0).getMBB() == &MBBN) {
    ++NumMYRISCVXDeleteRedundantJmp;
    MBB.erase(I); // delete the "J" instruction to go adjacent block
    Changed = true; // Notify LLVM kernel Changed
  }
  return Changed;
}


/// createMYRISCVXDeleteRedundantJmpPass - Returns a pass that MYRISCVXDeleteRedundantJmp in MYRISCVX MachineFunctions
FunctionPass *llvm::createMYRISCVXDeleteRedundantJmpPass() {
  return new MYRISCVXDeleteRedundantJmp();
}
