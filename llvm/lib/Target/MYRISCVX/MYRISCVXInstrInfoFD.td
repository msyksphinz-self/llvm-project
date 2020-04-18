//===- MYRISCVXInstrInfo.td - Target Description for MYRISCVX Target -*- tablegen -*-=//
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

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MYRISCVX Instructions
//===----------------------------------------------------------------------===//

// @{ MYRISCVXInstrInfoFD_td_FP_Arithmetic
def FADD_S  : ArithLogicR<0b1010011, 0b000, 0b0000000, "fadd.s", fadd, FPR_S>;
def FSUB_S  : ArithLogicR<0b1010011, 0b000, 0b0000100, "fsub.s", fsub, FPR_S>;
def FMUL_S  : ArithLogicR<0b1010011, 0b000, 0b0001000, "fmul.s", fmul, FPR_S>;
def FDIV_S  : ArithLogicR<0b1010011, 0b000, 0b0001100, "fdiv.s", fdiv, FPR_S>;

def FADD_D  : ArithLogicR<0b1010011, 0b000, 0b0000001, "fadd.d", fadd, FPR_D>;
def FSUB_D  : ArithLogicR<0b1010011, 0b000, 0b0000101, "fsub.d", fsub, FPR_D>;
def FMUL_D  : ArithLogicR<0b1010011, 0b000, 0b0001001, "fmul.d", fmul, FPR_D>;
def FDIV_D  : ArithLogicR<0b1010011, 0b000, 0b0001101, "fdiv.d", fdiv, FPR_D>;
// @} MYRISCVXInstrInfoFD_td_FP_Arithmetic
