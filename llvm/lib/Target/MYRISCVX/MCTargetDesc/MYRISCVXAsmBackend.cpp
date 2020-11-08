//===-- MYRISCVXAsmBackend.cpp - MYRISCVX Asm Backend  ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MYRISCVXAsmBackend class.
//
//===----------------------------------------------------------------------===//
//

#include "MCTargetDesc/MYRISCVXAsmBackend.h"
#include "MCTargetDesc/MYRISCVXFixupKinds.h"
#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"
#include "MCTargetDesc/MYRISCVXMCExpr.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// @{ MYRISCVXAsmBackend_adjustFixupValue
// Fixupの種類に応じて命令エンコーディングに合うように変換する
static unsigned adjustFixupValue(const MCFixup &Fixup, uint64_t Value,
                                 MCContext &Ctx) {

  unsigned Kind = Fixup.getKind();

  // Fixupの種類に応じてValueを変換する
  switch (Kind) {
    default:
      return 0;
    case FK_GPRel_4:
    case FK_Data_4:
      return Value;
    case MYRISCVX::fixup_MYRISCVX_LO12_I:
    case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_I:
      // Valueの下位12ビットを抽出する
      // (TargetOffsetにより左20ビットシフトされる想定)
      return Value & 0xfff;
    case MYRISCVX::fixup_MYRISCVX_LO12_S:
    case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_S:
      // ValueをS-typeの命令フォーマットに合うように変換する
      return (((Value >> 5) & 0x7f) << 25) | ((Value & 0x1f) << 7);
    case MYRISCVX::fixup_MYRISCVX_HI20:
    case MYRISCVX::fixup_MYRISCVX_PCREL_HI20:
      // ValueをU-typeの命令フォーマットに合うように変換する
      // (TargetOffsetにより左12ビットシフトされる想定)
      return ((Value + 0x800) >> 12) & 0xfffff;
      // @{ MYRISCVXAsmBackend_adjustFixupValue ...
    case MYRISCVX::fixup_MYRISCVX_JAL: {
      if (!isInt<21>(Value))
        Ctx.reportError(Fixup.getLoc(), "fixup value out of range");
      if (Value & 0x1)
        Ctx.reportError(Fixup.getLoc(), "fixup value must be 2-byte aligned");
      // Need to produce imm[19|10:1|11|19:12] from the 21-bit Value.
      unsigned Sbit = (Value >> 20) & 0x1;
      unsigned Hi8 = (Value >> 12) & 0xff;
      unsigned Mid1 = (Value >> 11) & 0x1;
      unsigned Lo10 = (Value >> 1) & 0x3ff;
      // Inst{31} = Sbit;
      // Inst{30-21} = Lo10;
      // Inst{20} = Mid1;
      // Inst{19-12} = Hi8;
      Value = (Sbit << 19) | (Lo10 << 9) | (Mid1 << 8) | Hi8;
      return Value;
    }
    case MYRISCVX::fixup_MYRISCVX_BRANCH: {
      if (!isInt<13>(Value))
        Ctx.reportError(Fixup.getLoc(), "fixup value out of range");
      if (Value & 0x1)
        Ctx.reportError(Fixup.getLoc(), "fixup value must be 2-byte aligned");
      // Need to extract imm[12], imm[10:5], imm[4:1], imm[11] from the 13-bit
      // Value.
      unsigned Sbit = (Value >> 12) & 0x1;
      unsigned Hi1 = (Value >> 11) & 0x1;
      unsigned Mid6 = (Value >> 5) & 0x3f;
      unsigned Lo4 = (Value >> 1) & 0xf;
      // Inst{31} = Sbit;
      // Inst{30-25} = Mid6;
      // Inst{11-8} = Lo4;
      // Inst{7} = Hi1;
      Value = (Sbit << 31) | (Mid6 << 25) | (Lo4 << 8) | (Hi1 << 7);
      return Value;
    }

    case MYRISCVX::fixup_MYRISCVX_CALL: {
      // Jalr will add UpperImm with the sign-extended 12-bit LowerImm,
      // we need to add 0x800ULL before extract upper bits to reflect the
      // effect of the sign extension.
      uint64_t UpperImm = (Value + 0x800ULL) & 0xfffff000ULL;
      uint64_t LowerImm = Value & 0xfffULL;
      return UpperImm | ((LowerImm << 20) << 32);
    }
      // @} MYRISCVXAsmBackend_adjustFixupValue ...
  }

}
// @} MYRISCVXAsmBackend_adjustFixupValue


// @{ MYRISCVXAsmBackend_cpp_createObjectTargetWriter
// MYRISCVXAsmBackendクラスでオーバーライドしているcreateObjectTargetWriter()メソッド
// createMYRISCVXELFObjectWriter()を呼び出す
std::unique_ptr<MCObjectTargetWriter>
MYRISCVXAsmBackend::createObjectTargetWriter() const {
  return createMYRISCVXELFObjectWriter(TheTriple, Is64Bit);
}
// @} MYRISCVXAsmBackend_cpp_createObjectTargetWriter


bool MYRISCVXAsmBackend::evaluateTargetFixup(
    const MCAssembler &Asm, const MCAsmLayout &Layout, const MCFixup &Fixup,
    const MCFragment *DF, const MCValue &Target, uint64_t &Value,
    bool &WasForced) {
  const MCFixup *AUIPCFixup;
  const MCFragment *AUIPCDF;
  MCValue AUIPCTarget;
  switch (Fixup.getTargetKind()) {
  default:
    llvm_unreachable("Unexpected fixup kind!");
  case MYRISCVX::fixup_MYRISCVX_PCREL_HI20:
    AUIPCFixup = &Fixup;
    AUIPCDF = DF;
    AUIPCTarget = Target;
    break;
  case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_I:
  case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_S: {
    AUIPCFixup = cast<MYRISCVXMCExpr>(Fixup.getValue())->getPCRelHiFixup(&AUIPCDF);
    if (!AUIPCFixup) {
      Asm.getContext().reportError(Fixup.getLoc(),
                                   "could not find corresponding %pcrel_hi");
      return true;
    }

    // MCAssembler::evaluateFixup will emit an error for this case when it sees
    // the %pcrel_hi, so don't duplicate it when also seeing the %pcrel_lo.
    const MCExpr *AUIPCExpr = AUIPCFixup->getValue();
    if (!AUIPCExpr->evaluateAsRelocatable(AUIPCTarget, &Layout, AUIPCFixup))
      return true;
    break;
  }
  }

  if (!AUIPCTarget.getSymA() || AUIPCTarget.getSymB())
    return false;

  const MCSymbolRefExpr *A = AUIPCTarget.getSymA();
  const MCSymbol &SA = A->getSymbol();
  if (A->getKind() != MCSymbolRefExpr::VK_None || SA.isUndefined())
    return false;

  auto *Writer = Asm.getWriterPtr();
  if (!Writer)
    return false;

  bool IsResolved = Writer->isSymbolRefDifferenceFullyResolvedImpl(
      Asm, SA, *AUIPCDF, false, true);
  if (!IsResolved)
    return false;

  Value = Layout.getSymbolOffset(SA) + AUIPCTarget.getConstant();
  Value -= Layout.getFragmentOffset(AUIPCDF) + AUIPCFixup->getOffset();

  if (shouldForceRelocation(Asm, *AUIPCFixup, AUIPCTarget)) {
    WasForced = true;
    return false;
  }

  return true;
}


// @{ MYRISCVXAsmBackend_applyFixup
// @{ MYRISCVXAsmBackend_applyFixup_adjustFixupValue
// applyFixup: 引数Valueを引数Fixupに適用するための関数
void MYRISCVXAsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                                    const MCValue &Target,
                                    MutableArrayRef<char> Data, uint64_t Value,
                                    bool IsResolved,
                                    const MCSubtargetInfo *STI) const {
  MCContext &Ctx = Asm.getContext();
  MCFixupKindInfo Info = getFixupKindInfo(Fixup.getKind());
  if (!Value)
    return; // FixupKindInfoに変換できなかった場合
  // FixupとValueを使用して新しいValueを作成する
  Value = adjustFixupValue(Fixup, Value, Ctx);

  // TargetOffestぶんだけシフトする
  Value <<= Info.TargetOffset;

  unsigned Offset = Fixup.getOffset();
  unsigned NumBytes = alignTo(Info.TargetSize + Info.TargetOffset, 8) / 8;

  assert(Offset + NumBytes <= Data.size() && "Invalid fixup offset!");

  // Valueを8バイト単位の配列に変換する
  for (unsigned i = 0; i != NumBytes; ++i) {
    Data[Offset + i] |= uint8_t((Value >> (i * 8)) & 0xff);
  }
}
// @} MYRISCVXAsmBackend_applyFixup_adjustFixupValue
// @} MYRISCVXAsmBackend_applyFixup


// This table *must* be in same the order of fixup_* kinds in
// MYRISCVXFixupKinds.h.
//
// @{ MYRISCVXAsmBackend_getFixupKindInfo
const MCFixupKindInfo &MYRISCVXAsmBackend::
getFixupKindInfo(MCFixupKind Kind) const {
  // FixupKindsの情報をテーブルで返す
  // Offsetの値はapplyFixup()でオフセット量として使用される
  const static MCFixupKindInfo Infos[MYRISCVX::NumTargetFixupKinds] = {
    // name                        offset  bits  flags
    { "fixup_MYRISCVX_HI20",           12,  20,      0   },
    { "fixup_MYRISCVX_LO12_I",         20,  12,      0   },
    { "fixup_MYRISCVX_LO12_S",          0,  32,      0   },
    { "fixup_MYRISCVX_PCREL_HI20",     12,  20,
      MCFixupKindInfo::FKF_IsPCRel | MCFixupKindInfo::FKF_IsTarget },
    { "fixup_MYRISCVX_PCREL_LO12_I",   20,  12,
      MCFixupKindInfo::FKF_IsPCRel | MCFixupKindInfo::FKF_IsTarget },
    { "fixup_MYRISCVX_PCREL_LO12_S",    0,  32,
      MCFixupKindInfo::FKF_IsPCRel | MCFixupKindInfo::FKF_IsTarget},
    { "fixup_MYRISCVX_CALL",            0,  32,      MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MYRISCVX_RELAX",           0,  0,       0   },
    { "fixup_MYRISCVX_GOT_HI20",       12,  20,      MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MYRISCVX_JAL",            12,  20,      MCFixupKindInfo::FKF_IsPCRel },
    { "fixup_MYRISCVX_BRANCH",          0,  32,      MCFixupKindInfo::FKF_IsPCRel },
  };

  if (Kind < FirstTargetFixupKind)
    return MCAsmBackend::getFixupKindInfo(Kind);

  assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
         "Invalid kind!");
  return Infos[Kind - FirstTargetFixupKind];
}
// @} MYRISCVXAsmBackend_getFixupKindInfo


/// WriteNopData - Write an (optimal) nop sequence of Count bytes
/// to the given output. If the target cannot generate such a sequence,
/// it should return an error.
///
/// \return - True on success.
bool MYRISCVXAsmBackend::writeNopData(raw_ostream &OS, uint64_t Count) const {
  return true;
}


// @{ MYRISCVXAsmBackend_createMYRISCVXAsmBackend
// LLVMInitializeMYRISCVXTargetMC()でMYRISCVXAsmBackendを登録するためのラッパー関数
MCAsmBackend *llvm::createMYRISCVXAsmBackend(const Target &T,
                                             const MCSubtargetInfo &STI,
                                             const MCRegisterInfo &MRI,
                                             const MCTargetOptions &Options) {
  const Triple &TT = STI.getTargetTriple();
  return new MYRISCVXAsmBackend(T, MRI, STI.getTargetTriple(), STI.getCPU(), TT.isArch64Bit());
}
// @} MYRISCVXAsmBackend_createMYRISCVXAsmBackend
