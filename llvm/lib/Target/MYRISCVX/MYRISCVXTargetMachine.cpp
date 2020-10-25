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
  // MYRISCVXターゲットの登録
  RegisterTargetMachine<MYRISCVX32TargetMachine> X(getTheMYRISCVX32Target());
  RegisterTargetMachine<MYRISCVX64TargetMachine> Y(getTheMYRISCVX64Target());
}
// @}MYRISCVXTargetMachine_cpp_LLVMInitializeMYRISCVXTarget


// @{MYRISCVXTargetMachine_cpp_computeDataLayout
// MYRISCVXターゲットマシンのデータレイアウトを決める
static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options) {
  std::string Ret = "";
  Ret += "e";
  Ret += "-m:m";

  // ポインタのサイズを指定する. RV32の場合は32ビット, RV64の場合は64ビット
  if (TT.isArch64Bit()) {
    Ret += "-p:64:64";
  } else {
    Ret += "-p:32:32";
  }

  // 8ビットと16ビット変数は32ビットにアラインする
  // 64ビット変数は64ビットにアラインする
  Ret += "-i8:8:32-i16:16:32-i64:64";
  // 整数レジスタ・スタックのアライメント
  if (TT.isArch64Bit()) {
    Ret += "-n64-S128";
  } else {
    Ret += "-n32-S64";
  }

  return Ret;
}
// @}MYRISCVXTargetMachine_cpp_computeDataLayout


// @{ MYRISCVXTargetMachine_cpp_getEffectiveRelocModel
// MYRISCVXの有効なリロケーションモデルを返す
// *RMを返すので、殆どの場合にはオプションで指定したモデルがそのまま使用される
static Reloc::Model getEffectiveRelocModel(bool JIT,
                                           Optional<Reloc::Model> RM) {
  if (!RM.hasValue() || JIT)
    return Reloc::Static;
  return *RM;
}
// @} MYRISCVXTargetMachine_cpp_getEffectiveRelocModel


// @{ MYRISCVXTargetMachine_cpp_MYRISCVXTargetMachine
// @{ MYRISCVXTargetMachine_cpp_getEffectiveCodeModel
// MYRISCVXTargetMachineのコンストラクタ：
//   MYRISCVX32TargetMachine()とMYRISCVX64TargetMachine()から呼び出される
MYRISCVXTargetMachine::MYRISCVXTargetMachine(const Target &T, const Triple &TT,
                                             StringRef CPU, StringRef FS,
                                             const TargetOptions &Options,
                                             Optional<Reloc::Model> RM,
                                             Optional<CodeModel::Model> CM,
                                             CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options), TT,
                        CPU, FS, Options, getEffectiveRelocModel(JIT, RM),
                        /* コードモデルの設定. デフォルトではSmall(=static), Medium(=PIC)も指定できる */
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      // @} MYRISCVXTargetMachine_cpp_getEffectiveCodeModel
      TLOF(std::make_unique<MYRISCVXTargetObjectFile>()),
      ABI(MYRISCVXABIInfo::computeTargetABI(Options.MCOptions.getABIName())),
      DefaultSubtarget(TT, CPU, /* TuneCPU */CPU, FS, *this) {
  initAsmInfo();
}
// @}MYRISCVXTargetMachine_cpp_MYRISCVXTargetMachine

MYRISCVXTargetMachine::~MYRISCVXTargetMachine() {}

void MYRISCVX32TargetMachine::anchor() { }

// @{MYRISCVXTargetMachine_cpp_MYRISCVX32_64TargetMachine
// MYRISCVX32ターゲットのコンストラクタ. MYRISCVXTargetMachineインスタンスを生成する
MYRISCVX32TargetMachine::MYRISCVX32TargetMachine(const Target &T, const Triple &TT,
                                                 StringRef CPU, StringRef FS,
                                                 const TargetOptions &Options,
                                                 Optional<Reloc::Model> RM,
                                                 Optional<CodeModel::Model> CM,
                                                 CodeGenOpt::Level OL, bool JIT)
    : MYRISCVXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT) {}

void MYRISCVX64TargetMachine::anchor() { }

// MYRISCVX64ターゲットのコンストラクタ. MYRISCVXTargetMachineインスタンスを生成する
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
/// MYRISCVX Code Generator Pass コンフィグレーションオプション.
class MYRISCVXPassConfig : public TargetPassConfig {
 public:
  MYRISCVXPassConfig(MYRISCVXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  MYRISCVXTargetMachine &getMYRISCVXTargetMachine() const {
    return getTM<MYRISCVXTargetMachine>();
  }

  // @{ MYRISCVXTargetMachine_cpp_addInstSelector
  bool addInstSelector() override;
  // @} MYRISCVXTargetMachine_cpp_addInstSelector

  // @{ MYRISCVXTargetMachine_cpp_addPreEmitPass2
  void addPreEmitPass2() override;

};
  // @{ MYRISCVXTargetMachine_cpp_addPreEmitPass2 ...
} // namespace

TargetPassConfig *MYRISCVXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MYRISCVXPassConfig(*this, PM);
}

// Install an instruction selector pass using
// the ISelDag to gen MYRISCVX code.
// @{ MYRISCVXTargetMachine_cpp_addInstSelector_impl
bool MYRISCVXPassConfig::addInstSelector() {
  addPass(createMYRISCVXISelDag(getMYRISCVXTargetMachine(), getOptLevel()));
  return false;
}
// @} MYRISCVXTargetMachine_cpp_addInstSelector_impl


// @} MYRISCVXTargetMachine_cpp_addPreEmitPass2 ...
void MYRISCVXPassConfig::addPreEmitPass2() {
  addPass(createMYRISCVXExpandPseudoPass());
}
// @} MYRISCVXTargetMachine_cpp_addPreEmitPass2
