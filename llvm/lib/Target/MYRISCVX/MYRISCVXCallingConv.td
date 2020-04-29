//===-- MYRISCVXCallingConv.td - Calling Conventions for MYRISCVX --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------------------------===//
// This describes the calling conventions for MYRISCVX architecture.
//===------------------------------------------------------------------------------===//

/// CCIfSubtarget - Match if the current subtarget has a feature F.
class CCIfSubtarget<string F, CCAction A, string Invert = "">
    : CCIf<!strconcat(Invert,
                      "static_cast<const MYRISCVXSubtarget&>"
			"(State.getMachineFunction().getSubtarget()).",
                      F),
           A>;

// @{MYRISCVXCallingConv_CSR_LP32
def CSR_LP32 : CalleeSavedRegs<(add SP, FP, S1, (sequence "S%u", 2, 11))>;
// @}MYRISCVXCallingConv_CSR_LP32

//===----------------------------------------------------------------------===//
// MYRISCVX LP32/STACK32 Return Convention
//===----------------------------------------------------------------------===//

// @{MYRISCVXCallingConv_RetCC_LP32
def RetCC_LP32 : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[A0, A1]>>
]>;
// @}MYRISCVXCallingConv_RetCC_LP32

//===----------------------------------------------------------------------===//
// MYRISCVX LP64/STACK64 Return Convention
//===----------------------------------------------------------------------===//

// @{MYRISCVXCallingConv_RetCC_LP64
def RetCC_LP64 : CallingConv<[
  CCIfType<[i32, i64], CCAssignToReg<[A0, A1]>>
]>;
// @}MYRISCVXCallingConv_RetCC_LP64

// @{MYRISCVXCallingConv_RetCC_MYRISCVX
def RetCC_MYRISCVX : CallingConv<[
  CCIfSubtarget<"is64Bit()", CCDelegateTo<RetCC_LP64>>,
  CCDelegateTo<RetCC_LP32>
]>;
// @}MYRISCVXCallingConv_RetCC_MYRISCVX
