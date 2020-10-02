//===-- MYRISCVXMCTargetDesc.cpp - MYRISCVX Target Descriptions -------------------===//
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

#include "MYRISCVXMCTargetDesc.h"

#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

// @{ MYRISCVXMC_TargetDesc_cpp_AddInclude
#define GET_INSTRINFO_MC_DESC
#include "MYRISCVXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MYRISCVXGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MYRISCVXGenRegisterInfo.inc"
// @} MYRISCVXMC_TargetDesc_cpp_AddInclude

// @{ MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstrInfo
static MCInstrInfo *createMYRISCVXMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMYRISCVXMCInstrInfo(X); // defined in MYRISCVXGenInstrInfo.inc
  return X;
}
// @} MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstrInfo

// @{ MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCRegisterInfo
static MCRegisterInfo *createMYRISCVXMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMYRISCVXMCRegisterInfo(X, MYRISCVX::RA); // defined in MYRISCVXGenRegisterInfo.inc
  return X;
}
// @} MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCRegisterInfo

// @{ MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCSubtargetInfo
static MCSubtargetInfo *createMYRISCVXMCSubtargetInfo(const Triple &TT,
                                                      StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = TT.isArch64Bit() ? "cpu-rv64" : "cpu-rv32";
  return createMYRISCVXMCSubtargetInfoImpl(TT, CPU, FS);
}
// @} MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCSubtargetInfo

namespace {

// @{ MYRISCVXMC_TargetDesc_cpp_MYRISCVXMCInstrAnalysis
class MYRISCVXMCInstrAnalysis : public MCInstrAnalysis {
 public:
  MYRISCVXMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}
};
}

static MCInstrAnalysis *createMYRISCVXMCInstrAnalysis(const MCInstrInfo *Info) {
  return new MYRISCVXMCInstrAnalysis(Info);
}
// @} MYRISCVXMC_TargetDesc_cpp_MYRISCVXMCInstrAnalysis

extern "C" void LLVMInitializeMYRISCVXTargetMC() {

}
