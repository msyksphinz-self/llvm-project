//===-- MYRISCVXMCAsmInfo.cpp - MYRISCVX Asm Properties -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MYRISCVXMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXMCAsmInfo.h"

#include "llvm/ADT/Triple.h"

using namespace llvm;

void MYRISCVXMCAsmInfo::anchor() { }

// @{ MYRISCVXMCAsmInfo_cpp_MYRISCVXMCAsmInfo
MYRISCVXMCAsmInfo::MYRISCVXMCAsmInfo(const Triple &TheTriple) {
  AlignmentIsInBytes          = true;           // 出力アセンブリファイルにアライメントに関する情報を出すか
  Data16bitsDirective         = "\t.2byte\t";   // 16ビットアライン時のディレクティブ
  Data32bitsDirective         = "\t.4byte\t";   // 32ビットアライン時のディレクティブ
  Data64bitsDirective         = "\t.8byte\t";   // 64ビットアライン時のディレクティブ
  PrivateGlobalPrefix         = "$";            // 定数プールなどのグローバル領域のプレフィックス
  // PrivateLabelPrefix: display $BB for the labels of basic block
  PrivateLabelPrefix          = "$";            // ラベルに使用されるプレフィックス
  CommentString               = "#";            // コメント
  ZeroDirective               = "\t.space\t";
  GPRel32Directive            = "\t.gpword\t";
  GPRel64Directive            = "\t.gpdword\t";
  WeakRefDirective            = "\t.weak\t";
  UseAssignmentForEHBegin = true;

  SupportsDebugInformation = true;
  DwarfRegNumForCFI = true;
}
// @} MYRISCVXMCAsmInfo_cpp_MYRISCVXMCAsmInfo
