//===-- llvm/Target/MYRISCVXTargetObjectFile.h - MYRISCVX Object Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MYRISCVXTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_MYRISCVX_MYRISCVXTARGETOBJECTFILE_H

#include "MYRISCVXTargetMachine.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class MYRISCVXTargetMachine;

// @{ MYRISCVXTargetObjectFile_h_MYRISCVXTargetObjectFile
// ELF形式をファイルを生成する際に使用するクラス
class MYRISCVXTargetObjectFile : public TargetLoweringObjectFileELF {
  const MYRISCVXTargetMachine *TM;
 public:

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};
// @} MYRISCVXTargetObjectFile_h_MYRISCVXTargetObjectFile

} // end namespace llvm

#endif
