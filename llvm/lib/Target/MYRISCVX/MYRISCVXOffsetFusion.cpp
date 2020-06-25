//===-- MYRISCVXOffsetFusion.cpp - MYRISCVX DelJmp -------------------------------===//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXTargetMachine.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"

using namespace llvm;

#define DEBUG_TYPE "del-jmp"

STATISTIC(NumMYRISCVXDeleteRedundantJmp, "Number of useless jmp deleted");

static cl::opt<bool> EnableMYRISCVXOffsetFusionPass(
    "enable-MYRISCVX-offset-fusion-pass",
    cl::init(true),
    cl::desc("Fuse ADDI and next Memory Access Offest."),
    cl::Hidden);

namespace {

class MYRISCVXOffsetFusionPass : public MachineFunctionPass {
  static char ID;

 public:
  MYRISCVXOffsetFusionPass() : MachineFunctionPass(ID) { }

  virtual StringRef getPassName() const {
    return "MYRISCVX Merge Offset Calculation";
  }

  bool runOnMachineBasicBlock(MachineBasicBlock &MBB, MachineBasicBlock &MBBN);

  bool runOnMachineFunction(MachineFunction &Fn) {

    for (MachineBasicBlock &MBB : Fn) {
      LLVM_DEBUG(dbgs() << "MBB: " << MBB.getName() << "\n");

      MachineInstr *PrevInst = nullptr;
      for (MachineInstr &HeadInst: MBB) {
        if (PrevInst != nullptr) {
          switch(HeadInst.getOpcode()) {
            case MYRISCVX::LB:
            case MYRISCVX::LH:
            case MYRISCVX::LW:
            case MYRISCVX::LBU:
            case MYRISCVX::LHU:
            case MYRISCVX::LWU:
            case MYRISCVX::LD:
            case MYRISCVX::SB:
            case MYRISCVX::SH:
            case MYRISCVX::SW:
            case MYRISCVX::SD:
              if (HeadInst.getOperand(2).getType() == MachineOperand::MO_Immediate &&
                  HeadInst.getOperand(2).getImm() == 0) {
                if (PrevInst->getOpcode() == MYRISCVX::ADDI &&
                    PrevInst->getOperand(0).getType() == MachineOperand::MO_Register &&
                    HeadInst.getOperand(1).getType() == MachineOperand::MO_Register &&
                    PrevInst->getOperand(0).getReg() == HeadInst.getOperand(1).getReg()) {
                  HeadInst.getOperand(2).setImm(PrevInst->getOperand(1).getImm());
                  MBB.erase(PrevInst);
                  LLVM_DEBUG(dbgs() << "  Replace Offset Calculation : " << PrevInst->getOperand(1).getImm() << '\n');
                }
              }
          }
        }
        PrevInst = &HeadInst;
      }
    }
    return true;
  }

};
}

char MYRISCVXOffsetFusionPass::ID = 0;

/// Returns an instance of the Merge Base Offset Optimization pass.
FunctionPass *llvm::createMYRISCVXOffsetFusionPass() {
  return new MYRISCVXOffsetFusionPass();
}
