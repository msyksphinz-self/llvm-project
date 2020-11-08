//===-- MYRISCVXELFStreamer.cpp - MYRISCVX ELF Target Streamer Methods ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides MYRISCVX specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXELFStreamer.h"
#include "MYRISCVXMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// @{ MYRISCVXELFStreamer_cpp_MYRISCVXTargetELFStreamer
// ELFオブジェクトファイルを生成するためのクラス
MYRISCVXTargetELFStreamer::MYRISCVXTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : MYRISCVXTargetStreamer(S) {}
// @} MYRISCVXELFStreamer_cpp_MYRISCVXTargetELFStreamer

// @{ MYRISCVXELFStreamer_cpp_getStreamer
MCELFStreamer &MYRISCVXTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}
// @} MYRISCVXELFStreamer_cpp_getStreamer

// void MYRISCVXTargetELFStreamer::emitDirectiveOptionPush() {}
// void MYRISCVXTargetELFStreamer::emitDirectiveOptionPop() {}
// void MYRISCVXTargetELFStreamer::emitDirectiveOptionRVC() {}
// void MYRISCVXTargetELFStreamer::emitDirectiveOptionNoRVC() {}
// void MYRISCVXTargetELFStreamer::emitDirectiveOptionRelax() {}
// void MYRISCVXTargetELFStreamer::emitDirectiveOptionNoRelax() {}
