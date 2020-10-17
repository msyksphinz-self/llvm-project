//===-- MYRISCVXAsmPrinter.cpp - MYRISCVX LLVM Assembly Printer -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format MYRISCVX assembly language.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXAsmPrinter.h"

#include "MCTargetDesc/MYRISCVXInstPrinter.h"
#include "MYRISCVX.h"
#include "MYRISCVXInstrInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-asm-printer"

bool MYRISCVXAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  MYRISCVXFI = MF.getInfo<MYRISCVXFunctionInfo>();
  AsmPrinter::runOnMachineFunction(MF);
  return true;
}

// @{ MYRISCVXAsmPrinter_cpp_EmitInstruction_PseudoExpansionLowering
// @{ MYRISCVXAsmPrinter_cpp_EmitInstruction
// @{ MYRISCVXAsmPrinter_cpp_EmitInstruction_MCInstLower
// MachineInstr形式の命令をMCInst形式に変換し、アセンブリ命令を出力する
void MYRISCVXAsmPrinter::emitInstruction(const MachineInstr *MI) {
  // @{ MYRISCVXAsmPrinter_cpp_EmitInstruction_MCInstLower ...
  // 命令出力時に, まずは疑似命令出力のチェックを行う
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;
  // @} MYRISCVXAsmPrinter_cpp_EmitInstruction_PseudoExpansionLowering

  if (MI->isDebugValue()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);

    PrintDebugValueComment(MI, OS);
    return;
  }

  // @} MYRISCVXAsmPrinter_cpp_EmitInstruction_MCInstLower ...
  MCInst TmpInst0;
  MCInstLowering.Lower(MI, TmpInst0);
  OutStreamer->emitInstruction(TmpInst0, getSubtargetInfo());
}
// @} MYRISCVXAsmPrinter_cpp_EmitInstruction_MCInstLower
// @} MYRISCVXAsmPrinter_cpp_EmitInstruction


bool MYRISCVXAsmPrinter::lowerOperand(const MachineOperand &MO, MCOperand &MCOp) {
  MCOp = MCInstLowering.LowerOperand(MO);
  return MCOp.isValid();
}

// @{ MYRISCVXAsmPrinter_cpp_EmitInstruction_PseudoLowering_inc
// 疑似命令への置き換えパタンは自動的に生成されているので, 
// 自動さ生成されたソースコードをincludeしておく
#include "MYRISCVXGenMCPseudoLowering.inc"
// @} MYRISCVXAsmPrinter_cpp_EmitInstruction_PseudoLowering_inc


/// Emit Set directives.
const char *MYRISCVXAsmPrinter::getCurrentABIString() const {
  switch (static_cast<MYRISCVXTargetMachine &>(TM).getABI().GetEnumValue()) {
    case MYRISCVXABIInfo::ABI::LP:    return "abilp";
    case MYRISCVXABIInfo::ABI::STACK: return "abistack";
    default: llvm_unreachable("Unknown MYRISCVX ABI");
  }
}

void MYRISCVXAsmPrinter::emitFunctionEntryLabel() {
  OutStreamer->emitLabel(CurrentFnSym);
}

/// EmitFunctionBodyStart - Targets can override this to emit stuff before
/// the first basic block in the function.
void MYRISCVXAsmPrinter::emitFunctionBodyStart() {
  MCInstLowering.Initialize(&MF->getContext());

  emitFrameDirective();

  if (OutStreamer->hasRawTextSupport()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    printSavedRegsBitmask(OS);
    OutStreamer->emitRawText(OS.str());
  }
}

/// EmitFunctionBodyEnd - Targets can override this to emit stuff after
/// the last basic block in the function.
void MYRISCVXAsmPrinter::emitFunctionBodyEnd() {}

void MYRISCVXAsmPrinter::emitStartOfAsmFile(Module &M) {
  // Tell the assembler which ABI we are using
  if (OutStreamer->hasRawTextSupport())
    OutStreamer->emitRawText("\t.section .mdebug." +
                             Twine(getCurrentABIString()));

  // return to previous section
  if (OutStreamer->hasRawTextSupport())
    OutStreamer->emitRawText(StringRef("\t.previous"));
}


void MYRISCVXAsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                                raw_ostream &OS) {
  // TODO: implement
  OS << "PrintDebugValueComment()";
}


// Force static initialization.
// @{ MYRISCVXAsmPrinter_cpp_LLVMInitializeMYRISCVXAsmPrinter
extern "C" void LLVMInitializeMYRISCVXAsmPrinter() {
  RegisterAsmPrinter<MYRISCVXAsmPrinter> X(getTheMYRISCVX32Target());
  RegisterAsmPrinter<MYRISCVXAsmPrinter> Y(getTheMYRISCVX64Target());
}
// @} MYRISCVXAsmPrinter_cpp_LLVMInitializeMYRISCVXAsmPrinter
