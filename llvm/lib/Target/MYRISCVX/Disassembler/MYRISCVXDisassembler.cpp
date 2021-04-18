//===- MYRISCVXDisassembler.cpp - Disassembler for MYRISCVX -------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the MYRISCVX Disassembler.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"

#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"
#include "MYRISCVXRegisterInfo.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

// @{ MYRISCVXDisassembler_MYRISCVXDisassembler
/// MYRISCVXDisassembler - a disasembler class for MYRISCVX32.
class MYRISCVXDisassembler : public MCDisassembler {
public:
  /// Constructor     - Initializes the disassembler.
  ///
  MYRISCVXDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}

  /// getInstruction(): Bytesバイト列から命令を切り取ってディスアセンブルする
  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
// @} MYRISCVXDisassembler_MYRISCVXDisassembler

} // end anonymous namespace

static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const void *Decoder);
template <unsigned N>
static DecodeStatus decodeImmOperand(MCInst &Inst, uint64_t Imm,
                                     int64_t Address, const void *Decoder);

template <unsigned N>
static DecodeStatus decodeBranchTarget(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder);

static MCDisassembler *createMYRISCVX32Disassembler(const Target &T,
                                                    const MCSubtargetInfo &STI,
                                                    MCContext &Ctx) {
  // Little Endian
  return new MYRISCVXDisassembler(STI, Ctx);
}

static MCDisassembler *createMYRISCVX64Disassembler(const Target &T,
                                                    const MCSubtargetInfo &STI,
                                                    MCContext &Ctx) {
  return new MYRISCVXDisassembler(STI, Ctx);
}

// @{ MYRISCVXDisassembler_LLVMInitializeMYRISCVXDisassembler
extern "C" void LLVMInitializeMYRISCVXDisassembler() {
  // MYRISCVXのディスアセンブラを登録する
  // LLVMInitializeXXXDisassembler()という関数を用意することで自動的に
  // llvm-objdumpにターゲットが組み込まれる
  TargetRegistry::RegisterMCDisassembler(getTheMYRISCVX32Target(),
                                         createMYRISCVX32Disassembler);
  TargetRegistry::RegisterMCDisassembler(getTheMYRISCVX64Target(),
                                         createMYRISCVX64Disassembler);
}
// @} MYRISCVXDisassembler_LLVMInitializeMYRISCVXDisassembler

// @{ MYRISCVXDisassembler_cpp_getInstruction
#include "MYRISCVXGenDisassemblerTables.inc"
// getInstruction()は引数Bytesから命令を切り取ってデコードする
DecodeStatus MYRISCVXDisassembler::getInstruction(MCInst &Instr, uint64_t &Size,
                                                  ArrayRef<uint8_t> Bytes,
                                                  uint64_t Address,
                                                  raw_ostream &CStream) const {
  // getInstruction(): Bytesバイト列から命令を切り取ってディスアセンブルする
  DecodeStatus Result;
  uint32_t Insn = support::endian::read32le(Bytes.data());

  // DecoderTableMYRISCVX32テーブルを使って命令をデコードし, MCInst構造体に格納する
  Result = decodeInstruction(DecoderTableMYRISCVX32, Instr, Insn, Address, this,
                             STI);
  if (Result != MCDisassembler::Fail) {
    Size = 4;
    return Result;
  }
  // デコードに失敗した場合MCDisassembler::Failを返す
  return MCDisassembler::Fail;
}
// @} MYRISCVXDisassembler_cpp_getInstruction

// @{ MYRISCVXDisassembler_cpp_GPRDecodeTable
const unsigned int GPRDecodeTable[] = {
    MYRISCVX::ZERO, MYRISCVX::RA,  MYRISCVX::SP,  MYRISCVX::GP, MYRISCVX::TP,
    MYRISCVX::T0,   MYRISCVX::T1,  MYRISCVX::T2,  MYRISCVX::FP, MYRISCVX::S1,
    MYRISCVX::A0,   MYRISCVX::A1,  MYRISCVX::A2,  MYRISCVX::A3, MYRISCVX::A4,
    MYRISCVX::A5,   MYRISCVX::A6,  MYRISCVX::A7,  MYRISCVX::S2, MYRISCVX::S3,
    MYRISCVX::S4,   MYRISCVX::S5,  MYRISCVX::S6,  MYRISCVX::S7, MYRISCVX::S8,
    MYRISCVX::S9,   MYRISCVX::S10, MYRISCVX::S11, MYRISCVX::T3, MYRISCVX::T4,
    MYRISCVX::T5,   MYRISCVX::T6};
// @} MYRISCVXDisassembler_cpp_GPRDecodeTable

// @{ MYRISCVXDisassembler_cpp_DecodeGPRRegisterClass
static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > sizeof(GPRDecodeTable))
    return MCDisassembler::Fail;

  // GPRDecodeTableテーブルを参照してデコードを行う
  unsigned int Reg = GPRDecodeTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}
// @} MYRISCVXDisassembler_cpp_DecodeGPRRegisterClass

// @{ MYRISCVXDisassembler_cpp_decodeImmOperand
// 即値オペランドのデコード
template <unsigned N>
static DecodeStatus decodeImmOperand(MCInst &Inst, uint64_t Imm,
                                     int64_t Address, const void *Decoder) {
  assert(isUInt<N>(Imm) && "Invalid immediate");

  Inst.addOperand(MCOperand::createImm(SignExtend64<N>(Imm)));
  return MCDisassembler::Success;
}
// @} MYRISCVXDisassembler_cpp_decodeImmOperand

// @{ MYRISCVXDisassembler_cpp_decodeBranchTarget
template <unsigned N>
static DecodeStatus decodeBranchTarget(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  // 分岐ターゲットは機械語では下位1ビットが省略されているので,
  // 1ビット左シフトして人間でも読めるようにする
  Inst.addOperand(MCOperand::createImm(SignExtend64<N>(Imm << 1)));
  return MCDisassembler::Success;
}
// @} MYRISCVXDisassembler_cpp_decodeBranchTarget
