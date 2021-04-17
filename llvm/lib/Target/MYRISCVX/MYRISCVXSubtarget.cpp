//===-- MYRISCVXSubtarget.cpp - MYRISCVX Subtarget Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MYRISCVX specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXSubtarget.h"

#include "MYRISCVXMachineFunction.h"
#include "MYRISCVX.h"
#include "MYRISCVXRegisterInfo.h"

#include "MYRISCVXTargetMachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MYRISCVXGenSubtargetInfo.inc"

extern bool FixGlobalBaseReg;

void MYRISCVXSubtarget::anchor() { }

MYRISCVXSubtarget::MYRISCVXSubtarget(const Triple &TT, StringRef &CPU, StringRef &TuneCPU,
                                     StringRef &FS,
                                     const MYRISCVXTargetMachine &_TM) :
    MYRISCVXGenSubtargetInfo(TT, CPU, TuneCPU, FS),
    TM(_TM), TargetTriple(TT), TSInfo(),
    InstrInfo(),
    FrameLowering(initializeSubtargetDependencies(CPU, TuneCPU, FS, TM)),
    TLInfo(TM, *this), RegInfo(*this, getHwMode()) {
}


bool MYRISCVXSubtarget::isPositionIndependent() const {
  return TM.isPositionIndependent();
}

// @{MYRISCVXSubtarget_initializeSubtargetDependencies
// Subtargetの依存関係を解析する
MYRISCVXSubtarget &
MYRISCVXSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef TuneCPU, StringRef FS,
                                                   const TargetMachine &TM) {
  if (TargetTriple.getArch() == Triple::myriscvx32) {
    // --march=myriscvx32が指定されると、CPUモデルは"cpu-rv32"を設定する
    if (CPU.empty() || CPU == "generic") {
      CPU = "cpu-rv32";
    }
  } else if (TargetTriple.getArch() == Triple::myriscvx64) {
    // --march=myriscvx64が指定されると、CPUモデルは"cpu-rv64"を設定する
    // つまり, 後にFeatureRV64が有効化される
    if (CPU.empty() || CPU == "generic") {
      CPU = "cpu-rv64";
    }
  } else {
    errs() << "!!!Error, TargetTriple.getArch() = " << TargetTriple.getArch()
           <<  "CPU = " << CPU << "\n";
    exit(0);
  }

  // llcのオプションから属性を解析する.
  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  // 特定のCPU構成における命令スケジューリングの初期化
  InstrItins = getInstrItineraryForCPU(CPU);
  // ParseSubtargetFeatures()の結果HasRV64がTrueか否かで, XLenVTのサイズを決定する
  if (HasRV64 == true) {
    XLenVT = MVT::i64;
  } else {
    XLenVT = MVT::i32;
  }
  return *this;
}
// @}MYRISCVXSubtarget_initializeSubtargetDependencies

bool MYRISCVXSubtarget::abiUsesSoftFloat() const {
  //  return TM->Options.UseSoftFloat;
  return true;
}

const MYRISCVXABIInfo &MYRISCVXSubtarget::getABI() const { return TM.getABI(); }
