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
