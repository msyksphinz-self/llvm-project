//===-- MYRISCVXMCExpr.cpp - MYRISCVX specific MC expression classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"

#include "MYRISCVXMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVXmcexpr"

const MYRISCVXMCExpr *MYRISCVXMCExpr::create(MYRISCVXMCExpr::MYRISCVXExprKind Kind,
                                             const MCExpr *Expr, MCContext &Ctx) {
  return new (Ctx) MYRISCVXMCExpr(Kind, Expr);
}

const MYRISCVXMCExpr *MYRISCVXMCExpr::create(const MCSymbol *Symbol, MYRISCVXMCExpr::MYRISCVXExprKind Kind,
                                             MCContext &Ctx) {
  const MCSymbolRefExpr *MCSym =
      MCSymbolRefExpr::create(Symbol, MCSymbolRefExpr::VK_None, Ctx);
  return new (Ctx) MYRISCVXMCExpr(Kind, MCSym);
}


//@{MYRISCVXMCExpr_printImpl
void MYRISCVXMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  int64_t AbsVal;

  switch (Kind) {
    case CEK_None:
      llvm_unreachable("CEK_None and CEK_Special are invalid");
      break;
    case CEK_HI:
      OS << "%hi";
      break;
    case CEK_LO:
      OS << "%lo";
      break;
    case CEK_CALL:
      OS << "%call";
      break;
    case CEK_GOT:
      OS << "%got";
      break;
    case CEK_GOT_HI20:
      OS << "%got_pcrel_hi";
      break;
    case CEK_PCREL_LO12_I:
      OS << "%pcrel_lo12";
      break;
    case CEK_PCREL_LO12_S:
      OS << "%pcrel_lo12";
      break;
  }

  OS << '(';
  if (Expr->evaluateAsAbsolute(AbsVal))
    OS << AbsVal;
  else
    Expr->print(OS, MAI, true);
  OS << ')';
}
//@}MYRISCVXMCExpr_printImpl

bool
MYRISCVXMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                          const MCAsmLayout *Layout,
                                          const MCFixup *Fixup) const {
  return getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup);
}

void MYRISCVXMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void MYRISCVXMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
    case CEK_None:
      llvm_unreachable("CEK_None and CEK_Special are invalid");
      break;
    case CEK_HI:
    case CEK_LO:
    case CEK_CALL:
    case CEK_GOT:
    case CEK_GOT_HI20:
    case CEK_PCREL_LO12_I:
    case CEK_PCREL_LO12_S:
      break;
  }
}
