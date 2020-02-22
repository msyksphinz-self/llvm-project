//===-- MYRISCVXTargetMachine.cpp - Define TargetMachine for MYRISCVX ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about MYRISCVX target spec.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"
#include "MYRISCVXTargetMachine.h"
#include "MYRISCVXISelDAGToDAG.h"
#include "MYRISCVXTargetObjectFile.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

// @{MYRISCVXTargetMachine_cpp_LLVMInitializeMYRISCVXTarget
extern "C" void LLVMInitializeMYRISCVXTarget() {
  // Register the target.
  //- Little endian Target Machine
  RegisterTargetMachine<MYRISCVX32TargetMachine> X(getTheMYRISCVX32Target());
  RegisterTargetMachine<MYRISCVX64TargetMachine> Y(getTheMYRISCVX64Target());
}
// @}MYRISCVXTargetMachine_cpp_LLVMInitializeMYRISCVXTarget


// @{MYRISCVXTargetMachine_cpp_computeDataLayout
static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options) {
  std::string Ret = "";
  Ret += "e";
  Ret += "-m:m";

  // Pointers on some ABIs.
  if (TT.isArch64Bit()) {
    Ret += "-p:64:64";
  } else {
    Ret += "-p:32:32";
  }

  // 8 and 16 bit integers only need to have natural alignment, but try to
  // align them to 32 bits. 64 bit integers have natural alignment.
  Ret += "-i8:8:32-i16:16:32-i64:64";

  if (TT.isArch64Bit()) {
    Ret += "-n64-S128";
  } else {
    Ret += "-n32-S64";
  }

  return Ret;
}
// @}MYRISCVXTargetMachine_cpp_computeDataLayout

static Reloc::Model getEffectiveRelocModel(bool JIT,
                                           Optional<Reloc::Model> RM) {
  if (!RM.hasValue() || JIT)
    return Reloc::Static;
  return *RM;
}

// @{MYRISCVXTargetMachine_cpp_MYRISCVXTargetMachine
MYRISCVXTargetMachine::MYRISCVXTargetMachine(const Target &T, const Triple &TT,
                                             StringRef CPU, StringRef FS,
                                             const TargetOptions &Options,
                                             Optional<Reloc::Model> RM,
                                             Optional<CodeModel::Model> CM,
                                             CodeGenOpt::Level OL, bool JIT)
    //- Default is big endian
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options), TT,
                        CPU, FS, Options, getEffectiveRelocModel(JIT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(make_unique<MYRISCVXTargetObjectFile>()),
      ABI(MYRISCVXABIInfo::computeTargetABI(Options.MCOptions.getABIName())),
      DefaultSubtarget(TT, CPU, FS, *this) {
  initAsmInfo();
}
// @}MYRISCVXTargetMachine_cpp_MYRISCVXTargetMachine

MYRISCVXTargetMachine::~MYRISCVXTargetMachine() {}

void MYRISCVX32TargetMachine::anchor() { }

// @{MYRISCVXTargetMachine_cpp_MYRISCVX32_64TargetMachine
MYRISCVX32TargetMachine::MYRISCVX32TargetMachine(const Target &T, const Triple &TT,
                                                 StringRef CPU, StringRef FS,
                                                 const TargetOptions &Options,
                                                 Optional<Reloc::Model> RM,
                                                 Optional<CodeModel::Model> CM,
                                                 CodeGenOpt::Level OL, bool JIT)
    : MYRISCVXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT) {}

void MYRISCVX64TargetMachine::anchor() { }

MYRISCVX64TargetMachine::MYRISCVX64TargetMachine(const Target &T, const Triple &TT,
                                                 StringRef CPU, StringRef FS,
                                                 const TargetOptions &Options,
                                                 Optional<Reloc::Model> RM,
                                                 Optional<CodeModel::Model> CM,
                                                 CodeGenOpt::Level OL, bool JIT)
    : MYRISCVXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT) {}
// @}MYRISCVXTargetMachine_cpp_MYRISCVX32_64TargetMachine

namespace {
//@MYRISCVXPassConfig {
/// MYRISCVX Code Generator Pass Configuration Options.
class MYRISCVXPassConfig : public TargetPassConfig {
 public:
  MYRISCVXPassConfig(MYRISCVXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  MYRISCVXTargetMachine &getMYRISCVXTargetMachine() const {
    return getTM<MYRISCVXTargetMachine>();
  }

  // @{MYRISCVXTargetMachine_cpp_addInstSelector
  bool addInstSelector() override;
  // @}MYRISCVXTargetMachine_cpp_addInstSelector

};
} // namespace

TargetPassConfig *MYRISCVXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MYRISCVXPassConfig(*this, PM);
}

// @{MYRISCVXTargetMachine_cpp_addInstSelector_impl
// Install an instruction selector pass using
// the ISelDag to gen MYRISCVX code.
bool MYRISCVXPassConfig::addInstSelector() {
  addPass(createMYRISCVXISelDag(getMYRISCVXTargetMachine(), getOptLevel()));
  return false;
}
// @}MYRISCVXTargetMachine_cpp_addInstSelector_impl
