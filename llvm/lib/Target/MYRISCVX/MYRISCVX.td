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

// @{ Add_MYRISCVX_td_target

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Target-dependent interfaces
//===----------------------------------------------------------------------===//

// @{ MYRISCVX_td_RV64_RV32
def RV64 : HwMode<"+64bit">;    // MYRISCVX64用のスイッチを定義
def RV32 : HwMode<"-64bit">;    // MYRISCVX32用のスイッチを定義
// @} MYRISCVX_td_RV64_RV32

include "MYRISCVXRegisterInfo.td"   // レジスタ定義を含んだtdファイル
include "MYRISCVXSchedule.td"
include "MYRISCVXInstrInfo.td"      // 命令定義を含んだtdファイル

def MYRISCVXInstrInfo : InstrInfo;

// @{ MYRISCVX_td_MYRISCVX
// MYRISCVXターゲットアーキテクチャを作成する
def MYRISCVX : Target {
  // MYRISCVXInstrInfoは上記で定義したもの. InstrInfoを継承している
  let InstructionSet = MYRISCVXInstrInfo;
}
// @} MYRISCVX_td_MYRISCVX
// @} Add_MYRISCVX_td_target
