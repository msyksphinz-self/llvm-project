//===- MYRISCVXInstrInfoFD.td - Target Description for MYRISCVX Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX Floating Point implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

// @{ MYRISCVXInstrInfo_FloatLoadStore
def FLW : LoadMemory <0b0000111, 0b010, "flw", FPR_S>;
def FSW : StoreMemory<0b0100111, 0b010, "fsw", FPR_S>;

def FLD : LoadMemory <0b0000111, 0b011, "fld", FPR_D>;
def FSD : StoreMemory<0b0100111, 0b011, "fsd", FPR_D>;
// @} MYRISCVXInstrInfo_FloatLoadStore


// @{ MYRISCVXInstrInfo_td_MemoryPattern_Define_forFP
defm : LoadPattern <load_a, FLW>;
defm : StorePattern<store,  FSW, FPR_S>;

defm : LoadPattern <load_a, FLD>;
defm : StorePattern<store , FSD, FPR_D>;
// @} MYRISCVXInstrInfo_td_MemoryPattern_Define_forFP


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
