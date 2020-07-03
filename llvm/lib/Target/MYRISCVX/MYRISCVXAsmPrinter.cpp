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


bool MYRISCVXAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << MYRISCVXInstPrinter::getRegisterName(MYRISCVX::ZERO);
        return false;
      }
      break;
    case 'i': // Literal 'i' if operand is not a register.
      if (!MO.isReg())
        OS << 'i';
      return false;
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << MYRISCVXInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  case MachineOperand::MO_BlockAddress: {
    MCSymbol *Sym = GetBlockAddressSymbol(MO.getBlockAddress());
    Sym->print(OS, MAI);
    return false;
  }
  default:
    break;
  }

  return true;
}

bool MYRISCVXAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  if (!ExtraCode) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // For now, we only support register memory operands in registers and
    // assume there is no addend
    if (!MO.isReg())
      return true;

    OS << "0(" << MYRISCVXInstPrinter::getRegisterName(MO.getReg()) << ")";
    return false;
  }

  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);
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
