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

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Target-dependent interfaces
//===----------------------------------------------------------------------===//

include "MYRISCVXRegisterInfo.td"
include "MYRISCVXInstrInfo.td"

def MYRISCVXInstrInfo : InstrInfo;

// @{ MYRISCVX_td_MYRISCVX
def MYRISCVX : Target {
  // MYRISCVXInstrInfoは上記で定義したもの. InstrInfoを継承している
  let InstructionSet = MYRISCVXInstrInfo;
}
// @} MYRISCVX_td_MYRISCVX
