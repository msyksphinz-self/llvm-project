//===-- MYRISCVXBaseInfo.h - Top level definitions for MYRISCVX MC --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the MYRISCVX target useful for the compiler back-end and the MC libraries.
//
//===--------------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXBASEINFO_H
#define LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXBASEINFO_H

#include "MYRISCVXMCTargetDesc.h"

#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// MYRISCVXII - This namespace holds all of the target specific flags that
/// instruction info tracks.
namespace MYRISCVXII {
// @{ MachineBaseInfo_MYRISCVXII_TOF
/// リロケーション情報を示すためのフラグは
/// ターゲットオペランドに付加されるフラグとして定義される
enum TOF {
  //===------------------------------------------------------------------===//
  // MYRISCVX専用のオペランドフラグを定義する
  MO_NONE,

  // 関数呼び出しのために使用するリロケーションフラグ
  MO_CALL,
  MO_CALL_PLT,

  // PC相対アドレス用のリロケーションフラグ
  MO_PCREL_HI20,
  MO_PCREL_LO12_I,
  MO_PCREL_LO12_S,

  // 絶対アドレス用のリロケーションフラグ
  MO_HI20,
  MO_LO12_I,
  MO_LO12_S,

  /// GOT用のリロケーションフラグ
  MO_GOT_HI20
}; // enum TOF {
// @} MachineBaseInfo_MYRISCVXII_TOF

enum {
  //===------------------------------------------------------------------===//
  // Instruction encodings.  These are the standard/most common forms for
  // MYRISCVX instructions.
  //

  // Pseudo - This represents an instruction that is a pseudo instruction
  // or one that has not been implemented yet.  It is illegal to code generate
  // it, but tolerated for intermediate implementation stages.
  Pseudo = 0,
  FrmR   = 1,
  FrmI   = 2,
  FrmS   = 3,
  FrmU   = 4,
  FrmB   = 5,
  FrmJ   = 6,

  FormMask = 15
};
}

}

#endif
