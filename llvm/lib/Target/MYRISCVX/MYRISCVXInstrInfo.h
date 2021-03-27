//===-- MYRISCVXInstrInfo.h - MYRISCVX Instruction Information --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MYRISCVXINSTRINFO_H
#define LLVM_LIB_TARGET_MYRISCVX_MYRISCVXINSTRINFO_H

#include "MYRISCVX.h"
#include "MYRISCVXRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MYRISCVXGenInstrInfo.inc"

namespace llvm {

// @{ MYRISCVXInstrInfo_MYRISCVXGenInstrInfo
// tdファイルで定義した命令をインスタンス化しLLVMに渡すためのクラス
class MYRISCVXInstrInfo : public MYRISCVXGenInstrInfo {
  // @} MYRISCVXInstrInfo_MYRISCVXGenInstrInfo
  virtual void anchor();
 public:
  explicit MYRISCVXInstrInfo();

  static const MYRISCVXInstrInfo *create(MYRISCVXSubtarget &STI);

  /// Return the number of bytes of code the specified instruction may be.
  unsigned GetInstSizeInBytes(const MachineInstr &MI) const;

  void expandRetRA(MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const;
  bool expandPostRAPseudo(MachineInstr &MI) const override;
 protected:
};
}

#endif
