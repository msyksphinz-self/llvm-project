//===-- MYRISCVXELFObjectWriter.cpp - MYRISCVX ELF Writer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MYRISCVXBaseInfo.h"
#include "MCTargetDesc/MYRISCVXFixupKinds.h"
#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class MYRISCVXELFObjectWriter : public MCELFObjectTargetWriter {
 public:
  MYRISCVXELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~MYRISCVXELFObjectWriter() override;

  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
  bool needsRelocateWithSymbol(const MCSymbol &Sym,
                               unsigned Type) const override;
};
}

MYRISCVXELFObjectWriter::MYRISCVXELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_MYRISCVX,
                              /*HasRelocationAddend*/ false) {}

MYRISCVXELFObjectWriter::~MYRISCVXELFObjectWriter() {}

// @{ MYRISCVXELFObjectWriter_getRelocType
// MCFixupからリロケーション情報へ変換を行う
unsigned MYRISCVXELFObjectWriter::getRelocType(MCContext &Ctx,
                                               const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const {
  // リロケーション情報を取得する
  unsigned Type = (unsigned)ELF::R_MYRISCVX_NONE;
  unsigned Kind = (unsigned)Fixup.getKind();

  switch (Kind) {
    default:
      llvm_unreachable("invalid fixup kind!");
    case MYRISCVX::fixup_MYRISCVX_HI20:
      return ELF::R_MYRISCVX_HI20;
    case MYRISCVX::fixup_MYRISCVX_LO12_I:
      return ELF::R_MYRISCVX_LO12_I;
      // @{ MYRISCVXELFObjectWriter_getRelocType ...
    case MYRISCVX::fixup_MYRISCVX_LO12_S:
      return ELF::R_MYRISCVX_LO12_S;
    case MYRISCVX::fixup_MYRISCVX_PCREL_HI20:
      return ELF::R_MYRISCVX_PCREL_HI20;
    case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_I:
      return ELF::R_MYRISCVX_PCREL_LO12_I;
    case MYRISCVX::fixup_MYRISCVX_PCREL_LO12_S:
      return ELF::R_MYRISCVX_PCREL_LO12_S;
    case MYRISCVX::fixup_MYRISCVX_CALL:
      return ELF::R_MYRISCVX_CALL;
    case MYRISCVX::fixup_MYRISCVX_BRANCH:
      return ELF::R_MYRISCVX_BRANCH;
    case MYRISCVX::fixup_MYRISCVX_JAL:
      return ELF::R_MYRISCVX_JAL;
      // @} MYRISCVXELFObjectWriter_getRelocType ...
    case MYRISCVX::fixup_MYRISCVX_RELAX:
      return ELF::R_MYRISCVX_RELAX;
    case MYRISCVX::fixup_MYRISCVX_GOT_HI20:
      return ELF::R_MYRISCVX_GOT_HI20;
  }

  return Type;
}
// @} MYRISCVXELFObjectWriter_getRelocType


bool
MYRISCVXELFObjectWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                 unsigned Type) const {
  // TODO: this is very conservative, update once RISC-V psABI requirements
  //       are clarified.
  return true;
}


// @{ MYRISCVXELFObjectWriter_cpp_createMYRISCVXELFObjectWriter
// MYRISCVXELFObjectWriterクラスをインスタンス化する
std::unique_ptr<MCObjectTargetWriter>
llvm::createMYRISCVXELFObjectWriter (const Triple &TT, bool Is64Bit) {
  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(TT.getOS());
  return std::make_unique<MYRISCVXELFObjectWriter>(OSABI, Is64Bit);
}
// @} MYRISCVXELFObjectWriter_cpp_createMYRISCVXELFObjectWriter
