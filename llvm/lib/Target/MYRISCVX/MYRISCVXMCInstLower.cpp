//===-- MYRISCVXMCInstLower.cpp - Convert MYRISCVX MachineInstr to MCInst --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower MYRISCVX MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXMCInstLower.h"

#include "MYRISCVXAsmPrinter.h"
#include "MYRISCVXInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

MYRISCVXMCInstLower::MYRISCVXMCInstLower(MYRISCVXAsmPrinter &asmprinter)
    : AsmPrinter(asmprinter) {}

void MYRISCVXMCInstLower::Initialize(MCContext* C) {
  Ctx = C;
}

MCOperand MYRISCVXMCInstLower::LowerOperand(const MachineOperand& MO,
                                            unsigned offset) const {
  MachineOperandType MOTy = MO.getType();

  switch (MOTy) {
    default: llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit()) break;
      return MCOperand::createReg(MO.getReg());
    case MachineOperand::MO_Immediate:
      return MCOperand::createImm(MO.getImm() + offset);
    case MachineOperand::MO_RegisterMask:
      break;
  }

  return MCOperand();
}

// @{ MYRISCVXMCInstLower_Lower
// MachineInstrからMCInstへの変換を行う関数
void MYRISCVXMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  // Opcodeはそのまま変換
  OutMI.setOpcode(MI->getOpcode());
  // オペランドは1つずつLowerOperand()を呼び出して変換する
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = LowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
// @} MYRISCVXMCInstLower_Lower
