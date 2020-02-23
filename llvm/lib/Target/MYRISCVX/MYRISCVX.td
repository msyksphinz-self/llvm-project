//===-- MYRISCVX.td - Describe the MYRISCVX Target Machine -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the MYRISCVX target.
//===----------------------------------------------------------------------===//

// @{Add_MYRISCVX_td_target

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// MYRISCVX subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

// @{MYRISCVX_td_SubTarget
def FeatureRV64 : SubtargetFeature<"64bit", "HasRV64", "true", "RV64 support">;

def : ProcessorModel<"cpu-rv32", NoSchedModel, []>;
def : ProcessorModel<"cpu-rv64", NoSchedModel, [FeatureRV64]>;
// @}MYRISCVX_td_SubTarget

//===----------------------------------------------------------------------===//
// Target-dependent interfaces
//===----------------------------------------------------------------------===//

// @{MYRISCVX_td_RV64_RV32
def RV64 : HwMode<"+64bit">;
def RV32 : HwMode<"-64bit">;
// @}MYRISCVX_td_RV64_RV32

// @{MYRISCVX_td_IsRV64
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<"FeatureRV64">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<"!FeatureRV64">;
// @}MYRISCVX_td_IsRV64

include "MYRISCVXRegisterInfo.td"
include "MYRISCVXSchedule.td"
include "MYRISCVXInstrInfo.td"
include "MYRISCVXCallingConv.td"

def MYRISCVXInstrInfo : InstrInfo;

// Will generate MYRISCVXGenAsmWrite.inc included by MYRISCVXInstPrinter.cpp, contents
//  as follows,
// void MYRISCVXInstPrinter::printInstruction(const MCInst *MI, raw_ostream &O) {...}
// const char *MYRISCVXInstPrinter::getRegisterName(unsigned RegNo) {...}
def MYRISCVX : Target {
// def MYRISCVXInstrInfo : InstrInfo as before.
  let InstructionSet = MYRISCVXInstrInfo;
}
// @}Add_MYRISCVX_td_target
