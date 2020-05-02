//===- MYRISCVXMatInt.cpp - Immediate materialisation -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXMatInt.h"
#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

namespace llvm {

namespace MYRISCVXMatInt {

// @{ MYRISCVXMatInt_cpp_generateInstSeq_head
void generateInstSeq(int64_t Val, bool IsRV64, InstSeq &Res) {
  // @} MYRISCVXMatInt_cpp_generateInstSeq_head

  // @{ MYRISCVXMatInt_cpp_generateInstSeq_isInt32
  if (isInt<32>(Val)) {
    int64_t Hi20 = ((Val + 0x800) >> 12) & 0xFFFFF;
    int64_t Lo12 = SignExtend64<12>(Val);

    if (Hi20)
      Res.push_back(Inst(MYRISCVX::LUI, Hi20));

    if (Lo12 || Hi20 == 0) {
      unsigned AddiOpc = (IsRV64 && Hi20) ? MYRISCVX::ADDIW : MYRISCVX::ADDI;
      Res.push_back(Inst(AddiOpc, Lo12));
    }
    return;
  }
  // @} MYRISCVXMatInt_cpp_generateInstSeq_isInt32

  assert(IsRV64 && "Can't emit >32-bit imm for non-RV64 target");

  // @{ MYRISCVXMatInt_cpp_generateInstSeq_higher32bit
  // @{ MYRISCVXMatInt_cpp_generateInstSeq_generate_LoHi
  int64_t Lo12 = SignExtend64<12>(Val);
  int64_t Hi52 = ((uint64_t)Val + 0x800ull) >> 12;
  int ShiftAmount = 12 + findFirstSet((uint64_t)Hi52);
  Hi52 = SignExtend64(Hi52 >> (ShiftAmount - 12), 64 - ShiftAmount);
  // @} MYRISCVXMatInt_cpp_generateInstSeq_generate_LoHi

  // @{ MYRISCVXMatInt_cpp_generateInstSeq_generate_high12
  generateInstSeq(Hi52, IsRV64, Res);
  // @} MYRISCVXMatInt_cpp_generateInstSeq_generate_high12

  Res.push_back(Inst(MYRISCVX::SLLI, ShiftAmount));
  if (Lo12)
    Res.push_back(Inst(MYRISCVX::ADDI, Lo12));
  // @} MYRISCVXMatInt_cpp_generateInstSeq_higher32bit
}
}
}
