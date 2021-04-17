//===-- Cpu0TargetMachine.h - Define TargetMachine for Cpu0 -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MYRISCVX specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MYRISCVXTARGETMACHINE_H
#define LLVM_LIB_TARGET_MYRISCVX_MYRISCVXTARGETMACHINE_H

#include "MCTargetDesc/MYRISCVXABIInfo.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

// @{MYRISCVXTargetMachine_h_MYRISCVXTargetMachine
namespace llvm {
class formatted_raw_ostream;
class MYRISCVXRegisterInfo;

// MYRISCVXTargetMachineはLLVMTargetMachineを継承して定義される
class MYRISCVXTargetMachine : public LLVMTargetMachine {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  MYRISCVXABIInfo ABI;
  MYRISCVXSubtarget DefaultSubtarget;
// @}MYRISCVXTargetMachine_h_MYRISCVXTargetMachine

  mutable StringMap<std::unique_ptr<MYRISCVXSubtarget>> SubtargetMap;
 public:
  MYRISCVXTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                        StringRef FS, const TargetOptions &Options,
                        Optional<Reloc::Model> RM,
                        Optional<CodeModel::Model> CM,
                        CodeGenOpt::Level OL, bool JIT);
  ~MYRISCVXTargetMachine() override;

  const MYRISCVXSubtarget *getSubtargetImpl(const Function &F) const override {
    return &DefaultSubtarget;
  }

  // Pass Pipeline Configuration
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
  const MYRISCVXABIInfo &getABI() const { return ABI; }
};

// @{MYRISCVXTargetMachine_h_MYRISCVXTargetMachine_32_64
/// MYRISCVX32TargetMachine - MYRISCVX32 ターゲットマシン.
///
class MYRISCVX32TargetMachine : public MYRISCVXTargetMachine {
  virtual void anchor();
 public:
  MYRISCVX32TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                          StringRef FS, const TargetOptions &Options,
                          Optional<Reloc::Model> RM,
                          Optional<CodeModel::Model> CM,
                          CodeGenOpt::Level OL, bool JIT);
};

/// MYRISCVX64TargetMachine - MYRISCVX64 ターゲットマシン.
///
class MYRISCVX64TargetMachine : public MYRISCVXTargetMachine {
  virtual void anchor();
 public:
  MYRISCVX64TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                          StringRef FS, const TargetOptions &Options,
                          Optional<Reloc::Model> RM,
                          Optional<CodeModel::Model> CM,
                          CodeGenOpt::Level OL, bool JIT);
};
// @}MYRISCVXTargetMachine_h_MYRISCVXTargetMachine_32_64
} // End llvm namespace

#endif // LLVM_LIB_TARGET_MYRISCVX_MYRISCVXTARGETMACHINE_H
