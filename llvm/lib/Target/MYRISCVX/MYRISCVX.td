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
// MYRISCVX subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

// @{ MYRISCVX_td_SubTarget
// MYRISCVXの属性としてFeatureRV64を定義する
def FeatureRV64 : SubtargetFeature<"64bit", "HasRV64", "true", "RV64 support">;

// CPUモデルcpu-rv32/cpu-rv64
// cpu-rv64の時はFeatureRV64が有効になる
def : ProcessorModel<"cpu-rv32", NoSchedModel, []>;
def : ProcessorModel<"cpu-rv64", NoSchedModel, [FeatureRV64]>;
// @} MYRISCVX_td_SubTarget

//===----------------------------------------------------------------------===//
// Target-dependent interfaces
//===----------------------------------------------------------------------===//

// @{ MYRISCVX_td_RV64_RV32
def RV64 : HwMode<"+64bit">;    // MYRISCVX64用のスイッチを定義
def RV32 : HwMode<"-64bit">;    // MYRISCVX32用のスイッチを定義
// @} MYRISCVX_td_RV64_RV32

// @{ MYRISCVX_td_IsRV64
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of FeatureRV64),
                       "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of (not FeatureRV64)),
                       "RV32I Base Instruction Set">;
// @} MYRISCVX_td_IsRV64

include "MYRISCVXRegisterInfo.td"   // レジスタ定義を含んだtdファイル
include "MYRISCVXSchedule.td"
include "MYRISCVXInstrInfo.td"      // 命令定義を含んだtdファイル
include "MYRISCVXCallingConv.td"

def MYRISCVXInstrInfo : InstrInfo;

// @{ Add_MYRISCVX_td_MYRISCVXAsmParser
// AsmParserで使用するMatcherを生成する
// @{ Add_MYRISCVX_td_AllowDuplicateRegisterNames
def MYRISCVXAsmParser : AsmParser {
  // AsmParserのParse時にAltNameで指定したレジスタ名も受け付ける
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}
// @} Add_MYRISCVX_td_AllowDuplicateRegisterNames
// @} Add_MYRISCVX_td_MYRISCVXAsmParser

// @{ MYRISCVX_td_MYRISCVX
// MYRISCVXターゲットアーキテクチャを作成する
// @{ Add_MYRISCVX_td_Target_AssemblyParsers
def MYRISCVX : Target {
  // MYRISCVXInstrInfoは上記で定義したもの. InstrInfoを継承している
  let InstructionSet = MYRISCVXInstrInfo;
  // アセンブリのParserとして`MYRISCVXAsmParser`を使用する
  let AssemblyParsers = [MYRISCVXAsmParser];
}
// @} MYRISCVX_td_MYRISCVX
// @} Add_MYRISCVX_td_Target_AssemblyParsers
// @} Add_MYRISCVX_td_target
