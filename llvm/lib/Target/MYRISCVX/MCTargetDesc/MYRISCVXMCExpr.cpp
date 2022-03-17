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


//@{ MYRISCVXMCExpr_printImpl
void MYRISCVXMCExpr::printImpl(raw_ostream &OS, const MCAsmInfo *MAI) const {
  int64_t AbsVal;

  switch (Kind) {
    case VK_MYRISCVX_None:
      llvm_unreachable("VK_MYRISCVX_None and VK_MYRISCVX_Special are invalid");
      break;
    case VK_MYRISCVX_HI20:
      OS << "%hi";
      break;
    case VK_MYRISCVX_LO12_I:
      OS << "%lo";
      break;
    case VK_MYRISCVX_LO12_S:
      OS << "%lo";
      break;
    case VK_MYRISCVX_CALL:
    case VK_MYRISCVX_CALL_PLT:
      if (Expr->evaluateAsAbsolute(AbsVal))
        OS << AbsVal;
      else
        Expr->print(OS, MAI, true);
      return;
    case VK_MYRISCVX_GOT_HI20:
      OS << "%got_pcrel_hi";
      break;
    case VK_MYRISCVX_PCREL_HI20:
      OS << "%pcrel_hi";
      break;
    case VK_MYRISCVX_PCREL_LO12_I:
      OS << "%pcrel_lo";
      break;
    case VK_MYRISCVX_PCREL_LO12_S:
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
//@} MYRISCVXMCExpr_printImpl


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


bool MYRISCVXMCExpr::evaluateAsConstant(int64_t &Res) const
{
  MCValue Value;

  if (Kind == VK_MYRISCVX_PCREL_HI20   || Kind == VK_MYRISCVX_GOT_HI20   ||
      Kind == VK_MYRISCVX_PCREL_LO12_I || Kind == VK_MYRISCVX_PCREL_LO12_S ||
      Kind == VK_MYRISCVX_CALL || Kind == VK_MYRISCVX_CALL_PLT)
    return false;

  if (!getSubExpr()->evaluateAsRelocatable(Value, nullptr, nullptr))
    return false;

  if (!Value.isAbsolute())
    return false;

  Res = evaluateAsInt64(Value.getConstant());
  return true;
}


int64_t MYRISCVXMCExpr::evaluateAsInt64(int64_t Value) const
{
  switch (Kind) {
  default:
    llvm_unreachable("Invalid kind");
  case VK_MYRISCVX_LO12_I:
  case VK_MYRISCVX_LO12_S:
    return SignExtend64<12>(Value);
  case VK_MYRISCVX_HI20:
    // Add 1 if bit 11 is 1, to compensate for low 12 bits being negative.
    return ((Value + 0x800) >> 12) & 0xfffff;
  }
}


void MYRISCVXMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  Streamer.visitUsedExpr(*getSubExpr());
}

void MYRISCVXMCExpr::fixELFSymbolsInTLSFixups(MCAssembler &Asm) const {
  switch (getKind()) {
    case VK_MYRISCVX_None:
      llvm_unreachable("VK_MYRISCVX_None and VK_MYRISCVX_Special are invalid");
      break;
    case VK_MYRISCVX_HI20:
    case VK_MYRISCVX_LO12_I:
    case VK_MYRISCVX_LO12_S:
    case VK_MYRISCVX_CALL:
    case VK_MYRISCVX_CALL_PLT:
    case VK_MYRISCVX_GOT_HI20:
    case VK_MYRISCVX_PCREL_HI20:
    case VK_MYRISCVX_PCREL_LO12_I:
    case VK_MYRISCVX_PCREL_LO12_S:
      break;
  }
}


// @{ MYRISCVXMCExpr_getVariantKindForName
MYRISCVXMCExpr::MYRISCVXExprKind MYRISCVXMCExpr::getVariantKindForName(StringRef name) {
  return StringSwitch<MYRISCVXMCExpr::MYRISCVXExprKind>(name)
      .Case("lo", VK_MYRISCVX_LO12_I)
      .Case("hi", VK_MYRISCVX_HI20)
      .Case("pcrel_lo", VK_MYRISCVX_PCREL_LO12_I)
      .Case("pcrel_hi", VK_MYRISCVX_PCREL_HI20)
      .Case("got_pcrel_hi", VK_MYRISCVX_GOT_HI20)
      .Default(VK_MYRISCVX_None);
}
// @} MYRISCVXMCExpr_getVariantKindForName

StringRef MYRISCVXMCExpr::getVariantKindName(MYRISCVXExprKind Kind) {
  switch (Kind) {
    default:
      llvm_unreachable("Invalid ELF symbol kind");
    case VK_MYRISCVX_LO12_S:
      return "lo";
    case VK_MYRISCVX_LO12_I:
      return "lo";
    case VK_MYRISCVX_HI20:
      return "hi";
    case VK_MYRISCVX_PCREL_LO12_I:
      return "pcrel_lo";
    case VK_MYRISCVX_PCREL_LO12_S:
      return "pcrel_lo";
    case VK_MYRISCVX_PCREL_HI20:
      return "pcrel_hi";
    case VK_MYRISCVX_GOT_HI20:
      return "got_pcrel_hi";
  }
}
