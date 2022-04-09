//===-- MYRISCVXTargetInfo.cpp - MYRISCVX Target Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// @{ MYRISCVX/TargetInfo/MYRISCVXTargetInfo.cpp
#include "MYRISCVX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace llvm {
// @{ getTheMYRISCVX32_64Target
// MYRISCVX32ターゲットインスタンスを返す
// staticなので常に同じインスタンスが返される
Target &getTheMYRISCVX32Target() {
  static Target TheMYRISCVX32Target;
  return TheMYRISCVX32Target;
}

// MYRISCVX64ターゲットインスタンスを返す
// staticなので常に同じインスタンスが返される
Target &getTheMYRISCVX64Target() {
  static Target TheMYRISCVX64Target;
  return TheMYRISCVX64Target;
}
// @} getTheMYRISCVX32_64Target
}

// @{ LLVMInitializeMYRISCVXTargetInfo
extern "C" void LLVMInitializeMYRISCVXTargetInfo() {
  RegisterTarget<Triple::myriscvx32,
        /*HasJIT=*/true>
      X(getTheMYRISCVX32Target(), "myriscvx32", "MYRISCVX (32-bit)", "MYRISCVX");

  RegisterTarget<Triple::myriscvx64,
        /*HasJIT=*/true>
      Y(getTheMYRISCVX64Target(), "myriscvx64", "MYRISCVX (64-bit)", "MYRISCVX");
}
// @} LLVMInitializeMYRISCVXTargetInfo
// @} MYRISCVX/TargetInfo/MYRISCVXTargetInfo.cpp
