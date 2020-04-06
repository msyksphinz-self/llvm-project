//===-- MYRISCVXMCTargetDesc.cpp - MYRISCVX Target Descriptions ------------===//
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
#include "MYRISCVXTargetStreamer.h"
#include "InstPrinter/MYRISCVXInstPrinter.h"
#include "MYRISCVXMCAsmInfo.h"

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
#include "llvm/Support/TargetRegistry.h"

#include "MCTargetDesc/MYRISCVXELFStreamer.h"
#include "MCTargetDesc/MYRISCVXMCAsmInfo.h"
#include "InstPrinter/MYRISCVXInstPrinter.h"

using namespace llvm;

// @{MYRISCVXMC_TargetDesc_cpp_AddInclude
#define GET_INSTRINFO_MC_DESC
#include "MYRISCVXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MYRISCVXGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MYRISCVXGenRegisterInfo.inc"
// @}MYRISCVXMC_TargetDesc_cpp_AddInclude

// @{MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstrInfo
static MCInstrInfo *createMYRISCVXMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMYRISCVXMCInstrInfo(X); // defined in MYRISCVXGenInstrInfo.inc
  return X;
}
// @}MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstrInfo

// @{MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCRegisterInfo
static MCRegisterInfo *createMYRISCVXMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMYRISCVXMCRegisterInfo(X, MYRISCVX::RA); // defined in MYRISCVXGenRegisterInfo.inc
  return X;
}
// @}MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCRegisterInfo

// @{MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCSubtargetInfo
static MCSubtargetInfo *createMYRISCVXMCSubtargetInfo(const Triple &TT,
                                                      StringRef CPU, StringRef FS) {
  return createMYRISCVXMCSubtargetInfoImpl(TT, CPU, FS);
}
// @}MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCSubtargetInfo

// @{MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCAsmInfo
static MCAsmInfo *createMYRISCVXMCAsmInfo(const MCRegisterInfo &MRI,
                                          const Triple &TT) {
  MCAsmInfo *MAI = new MYRISCVXMCAsmInfo(TT);

  unsigned SP = MRI.getDwarfRegNum(MYRISCVX::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}
// @}MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCAsmInfo

// @{MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstPrinter
static MCInstPrinter *createMYRISCVXMCInstPrinter(const Triple &T,
                                                  unsigned SyntaxVariant,
                                                  const MCAsmInfo &MAI,
                                                  const MCInstrInfo &MII,
                                                  const MCRegisterInfo &MRI) {
 return new MYRISCVXInstPrinter(MAI, MII, MRI);
}
// @}MYRISCVXMC_TargetDesc_cpp_createMYRISCVXMCInstPrinter

namespace {

// @{MYRISCVXMC_TargetDesc_cpp_MYRISCVXMCInstrAnalysis
class MYRISCVXMCInstrAnalysis : public MCInstrAnalysis {
 public:
  MYRISCVXMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}
};
}

static MCInstrAnalysis *createMYRISCVXMCInstrAnalysis(const MCInstrInfo *Info) {
  return new MYRISCVXMCInstrAnalysis(Info);
}
// @}MYRISCVXMC_TargetDesc_cpp_MYRISCVXMCInstrAnalysis


// @{MYISCVXMCTarget_cpp_createMYRISCVXAsmTargetStreamer
static MCTargetStreamer *createMYRISCVXAsmTargetStreamer(MCStreamer &S,
                                                         formatted_raw_ostream &OS,
                                                         MCInstPrinter *InstPrint,
                                                         bool isVerboseAsm) {
  return new MYRISCVXTargetAsmStreamer(S, OS);
}
// @}MYISCVXMCTarget_cpp_createMYRISCVXAsmTargetStreamer


// @{MYISCVXMCTarget_cpp_createMYRISCVXObjectTargetStreamer
static MCTargetStreamer
*createMYRISCVXObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new MYRISCVXTargetELFStreamer(S, STI);
}
// @}MYISCVXMCTarget_cpp_createMYRISCVXObjectTargetStreamer


// @{MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC
extern "C" void LLVMInitializeMYRISCVXTargetMC() {
  for (Target *T : {&getTheMYRISCVX32Target(), &getTheMYRISCVX64Target()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createMYRISCVXMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createMYRISCVXMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createMYRISCVXMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T,
	                                        createMYRISCVXMCSubtargetInfo);
    // Register the MC instruction analyzer.
    TargetRegistry::RegisterMCInstrAnalysis(*T, createMYRISCVXMCInstrAnalysis);
    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T,
	                                      createMYRISCVXMCInstPrinter);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createMYRISCVXAsmTargetStreamer);

    // @{MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC_Object
    // Register the object target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(*T, createMYRISCVXObjectTargetStreamer);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createMYRISCVXMCCodeEmitter);

    // @{MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC_RegisterMCAsmBackend
    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(*T, createMYRISCVXAsmBackend);
    // @}MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC_RegisterMCAsmBackend
    // @}MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC_Object
  }
}
// @}MYRISCVXMCTargetDesc_cpp_LLVMInitializeMYRISCVXTargetMC
