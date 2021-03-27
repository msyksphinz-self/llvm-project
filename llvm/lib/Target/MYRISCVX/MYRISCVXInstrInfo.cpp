//===-- MYRISCVXInstrInfo.cpp - MYRISCVX Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXInstrInfo.h"

#include "MYRISCVXTargetMachine.h"
#include "MYRISCVXMachineFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "MYRISCVXGenInstrInfo.inc"

// Pin the vtable to this file.
void MYRISCVXInstrInfo::anchor() {}

MYRISCVXInstrInfo::MYRISCVXInstrInfo() {}

/// Return the number of bytes of code the specified instruction may be.
unsigned MYRISCVXInstrInfo::GetInstSizeInBytes(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
    default:
      return MI.getDesc().getSize();
  }
}


//@{ MYRISCVXInstrInfo_expandPostRA
/// 疑似命令を本物の命令に変換するための関数
bool MYRISCVXInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();

  switch (MI.getDesc().getOpcode()) {
    default:
      return false;
    case MYRISCVX::RetRA:     // MYRISCVX::RetRAノードの場合はexpandRetRA()に飛ぶ
      expandRetRA(MBB, MI);
      break;
  }

  MBB.erase(MI);
  return true;
}


void MYRISCVXInstrInfo::expandRetRA(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I) const {
  // expandRetRAでは, MYRISCVX::RetRAノードをret命令(=JALR x0,ra,0)に変換する
  BuildMI(MBB, I, I->getDebugLoc(), get(MYRISCVX::JALR))
      .addReg(MYRISCVX::ZERO).addReg(MYRISCVX::RA).addImm(0);
}
//@} MYRISCVXInstrInfo_expandPostRA
