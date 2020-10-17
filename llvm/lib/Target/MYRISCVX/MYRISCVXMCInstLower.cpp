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


// @{ MYRISCVXMCInstLower_LowerSymbolOperand_Switch
MCOperand MYRISCVXMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                                  MachineOperandType MOTy,
                                                  unsigned Offset) const {
  MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
  MYRISCVXMCExpr::MYRISCVXExprKind TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_None;
  const MCSymbol *Symbol;

  switch(MO.getTargetFlags()) {
    default:                   llvm_unreachable("Invalid target flag!");
    case MYRISCVXII::MO_NONE:
      break;
    case MYRISCVXII::MO_HI20:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_HI20;
      break;
    case MYRISCVXII::MO_LO12_I:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_LO12_I;
      break;
    case MYRISCVXII::MO_LO12_S:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_LO12_S;
      break;

    case MYRISCVXII::MO_CALL:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_CALL;
      break;
    case MYRISCVXII::MO_CALL_PLT:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_CALL_PLT;
      break;

    case MYRISCVXII::MO_GOT_HI20:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_GOT_HI20;
      break;

    case MYRISCVXII::MO_PCREL_HI20:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_PCREL_HI20;
      break;
    case MYRISCVXII::MO_PCREL_LO12_I:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_PCREL_LO12_I;
      break;
    case MYRISCVXII::MO_PCREL_LO12_S:
      TargetKind = MYRISCVXMCExpr::VK_MYRISCVX_PCREL_LO12_S;
      break;
  }
  // @} MYRISCVXMCInstLower_LowerSymbolOperand_Switch

  switch (MOTy) {
    case MachineOperand::MO_GlobalAddress:
      Symbol = AsmPrinter.getSymbol(MO.getGlobal());
      Offset += MO.getOffset();
      break;

    default:
      llvm_unreachable("<unknown operand type>");
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, Kind, *Ctx);

  if (Offset) {
    // Assume offset is never negative.
    assert(Offset > 0);
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, *Ctx),
                                   *Ctx);
  }

  if (TargetKind != MYRISCVXMCExpr::VK_MYRISCVX_None)
    Expr = MYRISCVXMCExpr::create(TargetKind, Expr, *Ctx);

  return MCOperand::createExpr(Expr);
}


// @{ MYRISCVXMCInstLower_LowerOperand
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
    case MachineOperand::MO_GlobalAddress:
      return LowerSymbolOperand(MO, MOTy, offset);
    case MachineOperand::MO_RegisterMask:
      break;
  }

  return MCOperand();
}
// @} MYRISCVXMCInstLower_LowerOperand


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
