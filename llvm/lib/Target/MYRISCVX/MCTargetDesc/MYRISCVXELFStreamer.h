//===-- MYRISCVXELFStreamer.h - MYRISCVX ELF Target Streamer ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MYRISCVXELFSTREAMER_H
#define LLVM_LIB_TARGET_MYRISCVX_MYRISCVXELFSTREAMER_H

#include "MYRISCVXTargetStreamer.h"
#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

// @{ MYRISCVXELFStreamer_h_MYRISCVXTargetELFStreamer
// オブジェクトファイルを出力するためのStreamerクラス
// MYRISCVXTargetStreamerから派生している
class MYRISCVXTargetELFStreamer : public MYRISCVXTargetStreamer {
public:
  MCELFStreamer &getStreamer();
  MYRISCVXTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);
};
// @} MYRISCVXELFStreamer_h_MYRISCVXTargetELFStreamer
}
#endif
