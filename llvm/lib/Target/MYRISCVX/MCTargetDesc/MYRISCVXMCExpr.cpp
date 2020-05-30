//===-- MYRISCVXMCExpr.cpp - MYRISCVX specific MC expression classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"
#include "MCTargetDesc/MYRISCVXFixupKinds.h"
#include "MCTargetDesc/MYRISCVXMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolELF.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVXmcexpr"

const MYRISCVXMCExpr *MYRISCVXMCExpr::create(MYRISCVXExprKind Kind, const MCExpr *Expr,
                                             MCContext &Ctx) {
  return new (Ctx) MYRISCVXMCExpr(Kind, Expr);
}

//@{MYRISCVXMCExpr_printImpl
void MYRISCVXMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  int64_t AbsVal;

  switch (Kind) {
    case CEK_None:
      llvm_unreachable("CEK_None and CEK_Special are invalid");
      break;
    case CEK_HI20:
      OS << "%hi";
      break;
    case CEK_LO12_I:
      OS << "%lo";
      break;
    case CEK_LO12_S:
      OS << "%lo";
      break;
    case CEK_CALL:
    case CEK_CALL_PLT:
      if (Expr->evaluateAsAbsolute(AbsVal))
        OS << AbsVal;
      else
        Expr->print(OS, MAI, true);
      return;
    case CEK_GOT_HI20:
      OS << "%got_pcrel_hi";
      break;
    case CEK_PCREL_HI20:
      OS << "%pcrel_hi";
      break;
    case CEK_PCREL_LO12_I:
      OS << "%pcrel_lo";
      break;
    case CEK_PCREL_LO12_S:
      OS << "%pcrel_lo";
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


const MCFixup *MYRISCVXMCExpr::getPCRelHiFixup(const MCFragment **DFOut) const {
  MCValue AUIPCLoc;
  if (!getSubExpr()->evaluateAsRelocatable(AUIPCLoc, nullptr, nullptr))
    return nullptr;

  const MCSymbolRefExpr *AUIPCSRE = AUIPCLoc.getSymA();
  if (!AUIPCSRE)
    return nullptr;

  const MCSymbol *AUIPCSymbol = &AUIPCSRE->getSymbol();
  const auto *DF = dyn_cast_or_null<MCDataFragment>(AUIPCSymbol->getFragment());

  if (!DF)
    return nullptr;

  uint64_t Offset = AUIPCSymbol->getOffset();
  if (DF->getContents().size() == Offset) {
    DF = dyn_cast_or_null<MCDataFragment>(DF->getNextNode());
    if (!DF)
      return nullptr;
    Offset = 0;
  }

  for (const MCFixup &F : DF->getFixups()) {
    if (F.getOffset() != Offset)
      continue;

    switch ((unsigned)F.getKind()) {
      default:
        continue;
      case MYRISCVX::fixup_MYRISCVX_GOT_HI20:
      case MYRISCVX::fixup_MYRISCVX_PCREL_HI20:
        if (DFOut)
          *DFOut = DF;
        return &F;
    }
  }

  return nullptr;
}

bool
MYRISCVXMCExpr::evaluateAsRelocatableImpl(MCValue &Res,
                                          const MCAsmLayout *Layout,
                                          const MCFixup *Fixup) const {
  return getSubExpr()->evaluateAsRelocatable(Res, Layout, Fixup);
}

// @{ MYRISCVXMCExpr_getVariantKindForName
MYRISCVXMCExpr::MYRISCVXExprKind
MYRISCVXMCExpr::getVariantKindForName(StringRef name) {
  return StringSwitch<MYRISCVXMCExpr::MYRISCVXExprKind>(name)
      .Case("hi", CEK_HI20)
      .Case("lo", CEK_LO12_I)
      .Case("lo", CEK_LO12_S)
      .Case("call", CEK_CALL)
      .Case("call", CEK_CALL_PLT)
      .Case("got_pcrel_hi", CEK_GOT_HI20)
      .Case("pcrel_hi", CEK_PCREL_HI20)
      .Case("pcrel_lo12", CEK_PCREL_LO12_I)
      .Case("pcrel_lo12", CEK_PCREL_LO12_S)
      .Default(CEK_None);
}
// @} MYRISCVXMCExpr_getVariantKindForName


void MYRISCVXMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void MYRISCVXMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
    case CEK_None:
      llvm_unreachable("CEK_None and CEK_Special are invalid");
      break;
    case CEK_HI20:
    case CEK_LO12_I:
    case CEK_LO12_S:
    case CEK_CALL:
    case CEK_CALL_PLT:
    case CEK_GOT_HI20:
    case CEK_PCREL_HI20:
    case CEK_PCREL_LO12_I:
    case CEK_PCREL_LO12_S:
      break;
  }
}
