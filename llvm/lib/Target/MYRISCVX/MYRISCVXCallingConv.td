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

// @{ MYRISCVXCallingConv_CSR_LP32
// Calee Savedなレジスタのリストを作成する
// RISC-Vの仕様書に基づきCallee savedなレジスタを指定する
def CSR_LP32 : CalleeSavedRegs<(add RA, SP, FP, S1, (sequence "S%u", 2, 11))>;
// @} MYRISCVXCallingConv_CSR_LP32



//===----------------------------------------------------------------------===//
// MYRISCVX LP32/STACK32 Calling Convention
//===----------------------------------------------------------------------===//

// @{ MYRISCVXCallingConv_CC_LP32
// RV32の引数渡しConventionを定義する
def CC_LP32 : CallingConv<[
  // i1/i8/i16の型はi32に変換する
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i32型の引数はA0 - A7のレジスタに割り当てる
  CCIfType<[i32], CCAssignToReg<[A0, A1, A2, A3, A4, A5, A6, A7]>>,

  // 引数をスタックに割り当てる場合は4バイトサイズ, 4バイトアラインで配置する
  CCIfType<[i32], CCAssignToStack<4, 4>>
]>;
// @} MYRISCVXCallingConv_CC_LP32

// @{ MYRISCVXCallingConv_CC_STACK32
def CC_STACK32 : CallingConv<[
  // Promote i8/i16 arguments to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // Integer values get stored in stack slots that are 4 bytes in
  // size and 4-byte aligned.
  CCIfType<[i32], CCAssignToStack<4, 4>>
]>;
// @} MYRISCVXCallingConv_CC_STACK32

//===----------------------------------------------------------------------===//
// MYRISCVX LP64/STACK64 Calling Convention
//===----------------------------------------------------------------------===//

// @{ MYRISCVXCallingConv_CC_LP64
// RV64の引数渡しConventionを定義する
def CC_LP64 : CallingConv<[
  // i1/i8/i16の型はi32に変換する
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i32/i64型の引数はA0 - A7のレジスタに割り当てる
  CCIfType<[i32, i64], CCAssignToReg<[A0, A1, A2, A3, A4, A5, A6, A7]>>,

  // 引数をスタックに割り当てる場合は型に応じて確保するサイズとバイトアラインを指定する
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[i64], CCAssignToStack<8, 8>>
]>;
// @} MYRISCVXCallingConv_CC_LP64

// @{ MYRISCVXCallingConv_CC_STACK64
def CC_STACK64 : CallingConv<[
  // Promote i8/i16/i32 arguments to i64.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // Integer values get stored in stack slots that are 8 bytes in
  // size and 8-byte aligned.
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[i64], CCAssignToStack<8, 8>>
]>;
// @} MYRISCVXCallingConv_CC_STACK64


// @{ MYRISCVXCallingConv_CC_MYRISCVX
// MYRISCVXの引数渡しのConvention
def CC_MYRISCVX : CallingConv<[
  // RV64の場合はCC_LP64を使用する
  CCIfSubtarget<"isABI_LP64   ()", CCDelegateTo<CC_LP64>>,
  // RV32の場合はCC_LP32を使用する
  CCDelegateTo<CC_LP32>
]>;
// @} MYRISCVXCallingConv_CC_MYRISCVX


//===----------------------------------------------------------------------===//
// MYRISCVX LP32 Return Convention
//===----------------------------------------------------------------------===//

// @{ MYRISCVXCallingConv_RetCC_LP32
// RetCC_LP32はRV32用の戻り値に関する規約
// 戻り値をA0/A1に格納する
def RetCC_LP32 : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[A0, A1]>>
]>;
// @} MYRISCVXCallingConv_RetCC_LP32

//===----------------------------------------------------------------------===//
// MYRISCVX LP64 Return Convention
//===----------------------------------------------------------------------===//

// @{ MYRISCVXCallingConv_RetCC_LP64
// RetCC_LP64はRV64用の戻り値に関する規約
// 戻り値をA0/A1に格納する
def RetCC_LP64 : CallingConv<[
  CCIfType<[i32, i64], CCAssignToReg<[A0, A1]>>
]>;
// @} MYRISCVXCallingConv_RetCC_LP64

// @{ MYRISCVXCallingConv_RetCC_MYRISCVX
// MYRISCVXの戻り値に関する規約を定義する
// RV64ならばRetCC_LP64, RV32ならRetCC_LP32を使用する
def RetCC_MYRISCVX : CallingConv<[
  CCIfSubtarget<"is64Bit()", CCDelegateTo<RetCC_LP64>>,
  CCDelegateTo<RetCC_LP32>
]>;
// @} MYRISCVXCallingConv_RetCC_MYRISCVX
