//===-- MYRISCVXInstPrinter.cpp - Convert MYRISCVX MCInst to assembly syntax ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------------------------===//
//
// This class prints an MYRISCVX MCInst to a .s file.
//
//===------------------------------------------------------------------------------===//

#include "MYRISCVXInstPrinter.h"
#include "MCTargetDesc/MYRISCVXMCExpr.h"

#include "MYRISCVXInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "MYRISCVXGenAsmWriter.inc"

void MYRISCVXInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  //- getRegisterName(RegNo) defined in MYRISCVXGenAsmWriter.inc which indicate in
  //   MYRISCVX.td.
  OS << StringRef(getRegisterName(RegNo)).lower();
}

// @{ MYRISCVXInstPrinter_cpp_printInst
void MYRISCVXInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                    StringRef Annot, const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  // Try to print any aliases first.
  if (!printAliasInstr(MI, Address, O))
    //- printInstruction(MI, O) defined in MYRISCVXGenAsmWriter.inc which came from
    //   MYRISCVX.td indicate.
    printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}
// @} MYRISCVXInstPrinter_cpp_printInst


// @{ MYRISCVXInstPrinter_cpp_printOperand
void MYRISCVXInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  // MCInst形式の命令のオペランドを出力する
  // レジスタと即値の場合でそれぞれの専用関数を呼び出す
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
    return;
  }

  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }

  // それ以外の時は専用の出力関数を呼び出す
  assert(Op.isExpr() && "unknown operand kind in printOperand");
  Op.getExpr()->print(O, &MAI, true);
}
// @} MYRISCVXInstPrinter_cpp_printOperand


void MYRISCVXInstPrinter::printUnsignedImm(const MCInst *MI, int opNum,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

// @{ MYRISCVXInstPrinter_cpp_printMemOperand
void MYRISCVXInstPrinter::printMemOperand(const MCInst *MI, int opNum, raw_ostream &O) {
  // メモリオペランドに対する出力関数. imm(reg)の形式で出力する
  printOperand(MI, opNum+1, O);
  O << "(";
  printOperand(MI, opNum, O);
  O << ")";
}
// @} MYRISCVXInstPrinter_cpp_printMemOperand
