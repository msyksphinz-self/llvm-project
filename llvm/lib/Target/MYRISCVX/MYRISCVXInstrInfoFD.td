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


// @{ MYRISCVXInstrInfoFD_td_FPCompDestR
// Arithmetic and logical instructions with 2 register operands.
class FPCompDestR<bits<7> opcode, bits<3> funct3, bits<7>funct7,
                  string instr_asm, SDPatternOperator OpNode,
                  RegisterClass RC> :
  MYRISCVX_R<opcode, funct3, funct7, (outs GPR:$rd), (ins RC:$rs1, RC:$rs2),
  !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
  [(set GPR:$rd, (OpNode RC:$rs1, RC:$rs2))], IIAlu> {
    let isReMaterializable = 1;
}
// @} MYRISCVXInstrInfoFD_td_FPCompDestR


// @{ MYRISCVXInstrInfoFD_td_FPTwoOp
// Floating-Point instructions with 1 register operands.
class FPTwoOp<bits<7> opcode, bits<3> rm, bits<7> funct7,
                string instr_asm,
                RegisterClass RC> :
  MYRISCVX_R<opcode, rm, funct7, (outs RC:$rd), (ins RC:$rs1, RC:$rs2),
             !strconcat(instr_asm, "\t$rd, $rs1, $rs2"),
             [], IIAlu> {
    let isReMaterializable = 1;
  }
// @} MYRISCVXInstrInfoFD_td_FPTwoOp


// @{ MYRISCVXInstrInfoFD_td_FPSingleOp
// Floating-Point instructions with 1 register operands.
class FPSingleOp<bits<7> opcode, bits<3> rm, bits<7> funct7, bits<5> rs2_op,
                string instr_asm,
                RegisterClass DstRC, RegisterClass SrcRC> :
  MYRISCVX_R<opcode, rm, funct7, (outs DstRC:$rd), (ins SrcRC:$rs1),
             !strconcat(instr_asm, "\t$rd, $rs1"),
             [], IIAlu> {
    let isReMaterializable = 1;
    let rs2 = rs2_op;
  }
// @} MYRISCVXInstrInfoFD_td_FPSingleOp


//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// @{ MYRISCVXInstrInfoFD_td_FP_FMA_Format
// Floating-Point FMA instructions with 3 register operands.
class FPMultAdd<bits<7> opcode, bits<2> fmt, bits<3> rm,
                string instr_asm,
                RegisterClass RC> :
  MYRISCVX_FMA<opcode, fmt, rm, (outs RC:$rd), (ins RC:$rs1, RC:$rs2, RC:$rs3),
               !strconcat(instr_asm, "\t$rd, $rs1, $rs2, $rs3"),
               [], IIAlu> {
    let isReMaterializable = 1;
}
// @} MYRISCVXInstrInfoFD_td_FP_FMA_Format


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


// @{ MYRISCVXInstrInfoFD_td_FP_FMA
//
// Single Floating Point Operations
//
def FMADD_S  : FPMultAdd<0b1010011, 0b00, 0b000, "fmadd.s",  FPR_S>;
def FMSUB_S  : FPMultAdd<0b1000111, 0b00, 0b000, "fmsub.s",  FPR_S>;
def FNMSUB_S : FPMultAdd<0b1001011, 0b00, 0b000, "fnmsub.s", FPR_S>;
def FNMADD_S : FPMultAdd<0b1001111, 0b00, 0b000, "fnmadd.s", FPR_S>;
//
// Double Floating Point Operations
//
def FMADD_D  : FPMultAdd<0b1010011, 0b01, 0b000, "fmadd.d",  FPR_D>;
def FMSUB_D  : FPMultAdd<0b1000111, 0b01, 0b000, "fmsub.d",  FPR_D>;
def FNMSUB_D : FPMultAdd<0b1001011, 0b01, 0b000, "fnmsub.d", FPR_D>;
def FNMADD_D : FPMultAdd<0b1001111, 0b01, 0b000, "fnmadd.d", FPR_D>;
// @} MYRISCVXInstrInfoFD_td_FP_FMA


// @{ MYRISCVXInstrInfoFD_td_FP_S_FMA_Pattern
// Single Floating-Point FMA pattern registration
def : Pat<(fma FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3),
          (FMADD_S  $rs1, $rs2, $rs3)>;
def : Pat<(fma FPR_S:$rs1, FPR_S:$rs2, (fneg FPR_S:$rs3)),
          (FMSUB_S  FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
def : Pat<(fma (fneg FPR_S:$rs1), FPR_S:$rs2, FPR_S:$rs3),
          (FNMSUB_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
def : Pat<(fma (fneg FPR_S:$rs1), FPR_S:$rs2, (fneg FPR_S:$rs3)),
          (FNMADD_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;

def : Pat<(fadd (fmul       FPR_S:$rs1,  FPR_S:$rs2), FPR_S:$rs3),
          (FMADD_S  $rs1, $rs2, $rs3)>;
def : Pat<(fsub (fmul       FPR_S:$rs1,  FPR_S:$rs2), FPR_S:$rs3),
          (FMSUB_S  FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
def : Pat<(fadd (fmul (fneg FPR_S:$rs1), FPR_S:$rs2), FPR_S:$rs3),
          (FNMSUB_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
def : Pat<(fsub (fmul (fneg FPR_S:$rs1), FPR_S:$rs2), FPR_S:$rs3),
          (FNMADD_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;

def : Pat<(fadd (fneg (fmul FPR_S:$rs1, FPR_S:$rs2)), FPR_S:$rs3),
          (FNMSUB_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
def : Pat<(fsub (fneg (fmul FPR_S:$rs1, FPR_S:$rs2)), FPR_S:$rs3),
          (FNMADD_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;

def : Pat<(fsub FPR_S:$rs3, (fmul FPR_S:$rs1, FPR_S:$rs2)),
          (FNMSUB_S FPR_S:$rs1, FPR_S:$rs2, FPR_S:$rs3)>;
// @} MYRISCVXInstrInfoFD_td_FP_S_FMA_Pattern

// @} MYRISCVXInstrInfoFD_td_FP_D_FMA_Pattern
// Double Floating-Point FMA pattern registration
def : Pat<(fma FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3),
          (FMADD_D  $rs1, $rs2, $rs3)>;
def : Pat<(fma FPR_D:$rs1, FPR_D:$rs2, (fneg FPR_D:$rs3)),
          (FMSUB_D  FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
def : Pat<(fma (fneg FPR_D:$rs1), FPR_D:$rs2, FPR_D:$rs3),
          (FNMSUB_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
def : Pat<(fma (fneg FPR_D:$rs1), FPR_D:$rs2, (fneg FPR_D:$rs3)),
          (FNMADD_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;

def : Pat<(fadd (fmul       FPR_D:$rs1,  FPR_D:$rs2), FPR_D:$rs3),
          (FMADD_D  $rs1, $rs2, $rs3)>;
def : Pat<(fsub (fmul       FPR_D:$rs1,  FPR_D:$rs2), FPR_D:$rs3),
          (FMSUB_D  FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
def : Pat<(fadd (fmul (fneg FPR_D:$rs1), FPR_D:$rs2), FPR_D:$rs3),
          (FNMSUB_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
def : Pat<(fsub (fmul (fneg FPR_D:$rs1), FPR_D:$rs2), FPR_D:$rs3),
          (FNMADD_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;

def : Pat<(fadd (fneg (fmul FPR_D:$rs1, FPR_D:$rs2)), FPR_D:$rs3),
          (FNMSUB_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
def : Pat<(fsub (fneg (fmul FPR_D:$rs1, FPR_D:$rs2)), FPR_D:$rs3),
          (FNMADD_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;

def : Pat<(fsub FPR_D:$rs3, (fmul FPR_D:$rs1, FPR_D:$rs2)),
          (FNMSUB_D FPR_D:$rs1, FPR_D:$rs2, FPR_D:$rs3)>;
// @} MYRISCVXInstrInfoFD_td_FP_D_FMA_Pattern


// @{ MYRISCVXInstrInfoFD_td_FP_Compare_S
def FMAX_S  : ArithLogicR<0b1010011, 0b001, 0b0010100, "fmax.s", fmaxnum, FPR_S>;
def FMIN_S  : ArithLogicR<0b1010011, 0b000, 0b0010100, "fmin.s", fminnum, FPR_S>;
def FEQ_S   : FPCompDestR<0b1010011, 0b010, 0b1010000, "feq.s" , seteq,   FPR_S>;
def FLT_S   : FPCompDestR<0b1010011, 0b001, 0b1010000, "flt.s" , setlt,   FPR_S>;
def FLE_S   : FPCompDestR<0b1010011, 0b000, 0b1010000, "fle.s" , setle,   FPR_S>;
// @} MYRISCVXInstrInfoFD_td_FP_Compare_S

// @{ MYRISCVXInstrInfoFD_td_FP_Compare_S_Pattern
def : Pat<(setueq FPR_S:$rs1, FPR_S:$rs2), (FEQ_S $rs1, $rs2)>;
def : Pat<(setult FPR_S:$rs1, FPR_S:$rs2), (FLT_S $rs1, $rs2)>;
def : Pat<(setule FPR_S:$rs1, FPR_S:$rs2), (FLE_S $rs1, $rs2)>;
def : Pat<(setugt FPR_S:$rs1, FPR_S:$rs2), (FLE_S $rs2, $rs1)>;
def : Pat<(setuge FPR_S:$rs1, FPR_S:$rs2), (FLT_S $rs2, $rs1)>;

def : Pat<(setgt  FPR_S:$rs1, FPR_S:$rs2), (FLE_S $rs2, $rs1)>;
def : Pat<(setge  FPR_S:$rs1, FPR_S:$rs2), (FLT_S $rs2, $rs1)>;
// @} MYRISCVXInstrInfoFD_td_FP_Compare_S_Pattern


// @{ MYRISCVXInstrInfoFD_td_FP_Alias_S
def : InstAlias<"fge.s $rd, $rs1, $rs2",
                (FLT_S GPR:$rd, FPR_S:$rs2, FPR_S:$rs1), 0>;
def : InstAlias<"fgt.s $rd, $rs1, $rs2",
                (FLE_S GPR:$rd, FPR_S:$rs2, FPR_S:$rs1), 0>;
// @} MYRISCVXInstrInfoFD_td_FP_Alias_S


// @{ MYRISCVXInstrInfoFD_td_FP_Compare_D
def FMAX_D  : ArithLogicR<0b1010011, 0b001, 0b0010101, "fmax.s", fmaxnum, FPR_D>;
def FMIN_D  : ArithLogicR<0b1010011, 0b000, 0b0010101, "fmin.s", fminnum, FPR_D>;
def FEQ_D   : FPCompDestR<0b1010011, 0b010, 0b1010001, "feq.d" , seteq,   FPR_D>;
def FLT_D   : FPCompDestR<0b1010011, 0b001, 0b1010001, "flt.d" , setlt,   FPR_D>;
def FLE_D   : FPCompDestR<0b1010011, 0b000, 0b1010001, "fle.d" , setle,   FPR_D>;
// @} MYRISCVXInstrInfoFD_td_FP_Compare_D

// @{ MYRISCVXInstrInfoFD_td_FP_Compare_D_Pattern
def : Pat<(setueq FPR_D:$rs1, FPR_D:$rs2), (FEQ_D $rs1, $rs2)>;
def : Pat<(setult FPR_D:$rs1, FPR_D:$rs2), (FLT_D $rs1, $rs2)>;
def : Pat<(setule FPR_D:$rs1, FPR_D:$rs2), (FLE_D $rs1, $rs2)>;
def : Pat<(setugt FPR_D:$rs1, FPR_D:$rs2), (FLE_D $rs2, $rs1)>;
def : Pat<(setuge FPR_D:$rs1, FPR_D:$rs2), (FLT_D $rs2, $rs1)>;

def : Pat<(setgt  FPR_D:$rs1, FPR_D:$rs2), (FLE_D $rs2, $rs1)>;
def : Pat<(setge  FPR_D:$rs1, FPR_D:$rs2), (FLT_D $rs2, $rs1)>;
// @} MYRISCVXInstrInfoFD_td_FP_Compare_D_Pattern

// @{ MYRISCVXInstrInfoFD_td_FP_Alias_D
def : InstAlias<"fge.d $rd, $rs1, $rs2",
                (FLT_D GPR:$rd, FPR_D:$rs2, FPR_D:$rs1), 0>;
def : InstAlias<"fgt.d $rd, $rs1, $rs2",
                (FLE_D GPR:$rd, FPR_D:$rs2, FPR_D:$rs1), 0>;
// @} MYRISCVXInstrInfoFD_td_FP_Alias_D


// @{ MYRISCVXInstrInfoFD_td_FP_Single_TwoOps
def FSQRT_S   : FPSingleOp<0b1010011, 0b000, 0b0101100, 0b00000, "fsqrt.s",  FPR_S, FPR_S>;
def FSIGNJ_S  : FPTwoOp<0b1010011, 0b000, 0b0010000, "fsgnj.s" , FPR_S>;
def FSIGNJN_S : FPTwoOp<0b1010011, 0b001, 0b0010000, "fsgnjn.s", FPR_S>;
def FSIGNJX_S : FPTwoOp<0b1010011, 0b010, 0b0010000, "fsgnjx.s", FPR_S>;
def FCVT_W_S  : FPSingleOp<0b1010011, 0b000, 0b1100000, 0b00000, "fcvt.w.s",  GPR, FPR_S>;
def FCVT_WU_S : FPSingleOp<0b1010011, 0b000, 0b1100000, 0b00001, "fcvt.wu.s", GPR, FPR_S>;
def FMV_X_W   : FPSingleOp<0b1010011, 0b000, 0b1110000, 0b00000, "fmv.x.w",   GPR, FPR_S>;
def FCLASS_S  : FPSingleOp<0b1010011, 0b001, 0b1110000, 0b00000, "fclass.s",  FPR_S, FPR_S>;
def FCVT_S_W  : FPSingleOp<0b1010011, 0b000, 0b1101000, 0b00000, "fcvt.s.w",  FPR_S, GPR>;
def FCVT_S_WU : FPSingleOp<0b1010011, 0b000, 0b1101000, 0b00001, "fcvt.s.wu", FPR_S, GPR>;
def FMV_W_X   : FPSingleOp<0b1010011, 0b000, 0b1111000, 0b00000, "fmv.w.x",   FPR_S, GPR>;
// @} MYRISCVXInstrInfoFD_td_FP_Single_TwoOps


// @{ MYRISCVXInstrInfoFD_td_FP_Single_TwoOps_Patterns
def : Pat<(fsqrt FPR_S:$rs1), (FSQRT_S   $rs1)>;
def : Pat<(fneg  FPR_S:$rs1), (FSIGNJN_S $rs1, $rs1)>;
def : Pat<(fabs  FPR_S:$rs1), (FSIGNJX_S $rs1, $rs1)>;
def : InstAlias<"fmv.s  $rd, $rs1", (FSIGNJ_S  FPR_S:$rd, FPR_S:$rs1, FPR_S:$rs1), 0>;
def : InstAlias<"fneg.s $rd, $rs1", (FSIGNJN_S FPR_S:$rd, FPR_S:$rs1, FPR_S:$rs1), 0>;
def : InstAlias<"fabs.s $rd, $rs1", (FSIGNJX_S FPR_S:$rd, FPR_S:$rs1, FPR_S:$rs1), 0>;
def : Pat<(fp_to_sint FPR_S:$rs1), (FCVT_W_S  $rs1)>;
def : Pat<(fp_to_uint FPR_S:$rs1), (FCVT_WU_S $rs1)>;
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_S_W  $rs1)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_S_WU $rs1)>;
// @} MYRISCVXInstrInfoFD_td_FP_Single_TwoOps_Patterns


// @{ MYRISCVXInstrInfoFD_td_FP_Double_Single_TwoOps
def FSQRT_D   : FPSingleOp<0b1010011, 0b000, 0b0101101, 0b00000, "fsqrt.d",  FPR_D, FPR_D>;
def FSIGNJ_D  : FPTwoOp<0b1010011, 0b000, 0b0010001, "fsgnj.d" , FPR_D>;
def FSIGNJN_D : FPTwoOp<0b1010011, 0b001, 0b0010001, "fsgnjn.d", FPR_D>;
def FSIGNJX_D : FPTwoOp<0b1010011, 0b010, 0b0010001, "fsgnjx.d", FPR_D>;
def FCVT_W_D  : FPSingleOp<0b1010011, 0b000, 0b1100001, 0b00000, "fcvt.w.d",  GPR, FPR_D>;
def FCVT_WU_D : FPSingleOp<0b1010011, 0b000, 0b1100001, 0b00001, "fcvt.wu.d", GPR, FPR_D>;
// def FMV_X_W   : FPSingleOp<0b1010011, 0b000, 0b1110001, 0b00000, "fmv.x.w",   GPR, FPR_D>;
def FCLASS_D  : FPSingleOp<0b1010011, 0b001, 0b1110001, 0b00000, "fclass.d",  FPR_D, FPR_D>;
def FCVT_D_W  : FPSingleOp<0b1010011, 0b000, 0b1101001, 0b00000, "fcvt.d.w",  FPR_D, GPR>;
def FCVT_D_WU : FPSingleOp<0b1010011, 0b000, 0b1101001, 0b00001, "fcvt.d.wu", FPR_D, GPR>;
// def FMV_W_X   : FPSingleOp<0b1010011, 0b000, 0b1111001, 0b00000, "fmv.w.x",   FPR_D, GPR>;
// @} MYRISCVXInstrInfoFD_td_FP_Double_Single_TwoOps


// @{ MYRISCVXInstrInfoFD_td_FP_Double_Single_TwoOps_Patterns
def : Pat<(fsqrt FPR_D:$rs1), (FSQRT_D $rs1)>;
def : Pat<(fneg  FPR_D:$rs1), (FSIGNJN_D $rs1, $rs1)>;
def : Pat<(fabs  FPR_D:$rs1), (FSIGNJX_D $rs1, $rs1)>;
def : InstAlias<"fmv.d  $rd, $rs1", (FSIGNJ_D  FPR_D:$rd, FPR_D:$rs1, FPR_D:$rs1), 0>;
def : InstAlias<"fneg.d $rd, $rs1", (FSIGNJN_D FPR_D:$rd, FPR_D:$rs1, FPR_D:$rs1), 0>;
def : InstAlias<"fabs.d $rd, $rs1", (FSIGNJX_D FPR_D:$rd, FPR_D:$rs1, FPR_D:$rs1), 0>;
def : Pat<(fp_to_sint FPR_D:$rs1), (FCVT_W_D  $rs1)>;
def : Pat<(fp_to_uint FPR_D:$rs1), (FCVT_WU_D $rs1)>;
def : Pat<(sint_to_fp GPR:$rs1), (FCVT_D_W  $rs1)>;
def : Pat<(uint_to_fp GPR:$rs1), (FCVT_D_WU $rs1)>;
// @} MYRISCVXInstrInfoFD_td_FP_Double_Single_TwoOps_Patterns
