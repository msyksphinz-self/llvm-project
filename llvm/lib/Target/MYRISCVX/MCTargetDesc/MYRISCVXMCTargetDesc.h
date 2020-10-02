//===-- MYRISCVXMCTargetDesc.h - MYRISCVX Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MYRISCVX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXMCTARGETDESC_H
#define LLVM_LIB_TARGET_MYRISCVX_MCTARGETDESC_MYRISCVXMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class Target;
class Triple;

Target &getTheMYRISCVX32Target();
Target &getTheMYRISCVX64Target();

// @{ MYRISCVXMC_TargetDesc_h_AddInclude
// MYRISCVXGenRegisterInfo.inc / MYRISCVXGenInstrInfo.inc / 
// MYRISCVXGenSubtargetInfo.inc からヘッダファイルに必要な情報を抽出する
#define GET_REGINFO_ENUM
#include "MYRISCVXGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "MYRISCVXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MYRISCVXGenSubtargetInfo.inc"
// @} MYRISCVXMC_TargetDesc_h_AddInclude


} // End llvm namespace

#endif
