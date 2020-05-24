//===-- MYRISCVXFixupKinds.h - MYRISCVX Specific Fixup Entries ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXFIXUPKINDS_H
#define LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
  namespace MYRISCVX {
  // @{ MYRISCVXFixupkinds_Fixups
    // Although most of the current fixup types reflect a unique relocation
    // one can have multiple fixup types for a given relocation and thus need
    // to be uniquely named.
    //
    // This table *must* be in the save order of
    // MCFixupKindInfo Infos[MYRISCVX::NumTargetFixupKinds]
    // in MYRISCVXAsmBackend.cpp.
    enum Fixups {
      fixup_MYRISCVX_HI20 = FirstTargetFixupKind,
      fixup_MYRISCVX_LO12_I,
      fixup_MYRISCVX_LO12_S,
      fixup_MYRISCVX_PCREL_HI20,
      fixup_MYRISCVX_PCREL_LO12_I,
      fixup_MYRISCVX_PCREL_LO12_S,
      fixup_MYRISCVX_CALL,
      fixup_MYRISCVX_RELAX,
      fixup_MYRISCVX_GOT_HI20,
      fixup_MYRISCVX_JAL,
      fixup_MYRISCVX_BRANCH,

      // fixup_MYRISCVX_invalid - used as a sentinel and a marker, must be last fixup
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  // @} MYRISCVXFixupkinds_Fixups
  } // namespace MYRISCVX
} // namespace llvm

#endif // LLVM_MYRISCVX_MYRISCVXFIXUPKINDS_H
