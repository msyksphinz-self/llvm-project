//===-- MYRISCVXMCCodeEmitter.cpp - Convert MYRISCVX Code to Machine Code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MYRISCVXMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//

#include "MYRISCVXMCCodeEmitter.h"
#include "MCTargetDesc/MYRISCVXBaseInfo.h"
#include "MCTargetDesc/MYRISCVXFixupKinds.h"
#include "MCTargetDesc/MYRISCVXMCExpr.h"
#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"

#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mccodeemitter"

#define GET_INSTRMAP_INFO
#include "MYRISCVXGenInstrInfo.inc"
#undef GET_INSTRMAP_INFO

namespace llvm {


MCCodeEmitter *createMYRISCVXMCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           MCContext &Ctx) {
  return new MYRISCVXMCCodeEmitter(MCII, Ctx, true);
}
} // End of namespace llvm

void MYRISCVXMCCodeEmitter::EmitByte(unsigned char C, raw_ostream &OS) const {
  OS << (char)C;
}

// @{ MYRISCVXMCCodeEmitter_cpp_EmitInstruction
void MYRISCVXMCCodeEmitter::EmitInstruction(uint64_t Val, unsigned Size, raw_ostream &OS) const {
  // Output the instruction encoding in little endian byte order.
  for (unsigned i = 0; i < Size; ++i) {
    unsigned Shift = IsLittleEndian ? i * 8 : (Size - 1 - i) * 8;
    EmitByte((Val >> Shift) & 0xff, OS);
  }
}
// @} MYRISCVXMCCodeEmitter_cpp_EmitInstruction


// @{ MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_Head
void MYRISCVXMCCodeEmitter::expandFunctionCall(const MCInst &MI, raw_ostream &OS,
                                               SmallVectorImpl<MCFixup> &Fixups,
                                               const MCSubtargetInfo &STI) const {
// @} MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_Head

  // @{ MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_Func_Ra
  MCInst TmpInst;
  MCOperand Func;
  Register Ra;
  if (MI.getOpcode() == MYRISCVX::PseudoTAILCALLReg) {
    Func = MI.getOperand(0);
    Ra = MYRISCVX::T1;
  } else if (MI.getOpcode() == MYRISCVX::PseudoCALLReg) {
    Func = MI.getOperand(0);
    Ra = MYRISCVX::RA;
  }
  // @} MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_Func_Ra

  uint32_t Binary;

  assert(Func.isExpr() && "Expected expression");

  const MCExpr *CallExpr = Func.getExpr();

  // @{ MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_AUIPC
  TmpInst = MCInstBuilder(MYRISCVX::AUIPC)
                .addReg(Ra)
                .addOperand(MCOperand::createExpr(CallExpr));
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(OS, Binary, support::little);
  // @} MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_AUIPC

  // @{ MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_JALR
  if (MI.getOpcode() == MYRISCVX::PseudoTAILCALLReg) {
    // Emit JALR X0, X6, 0
    TmpInst = MCInstBuilder(MYRISCVX::JALR).addReg(MYRISCVX::ZERO).addReg(Ra).addImm(0);
  } else {
    // Emit JALR Ra, Ra, 0
    TmpInst = MCInstBuilder(MYRISCVX::JALR).addReg(Ra).addReg(Ra).addImm(0);
  }
  Binary = getBinaryCodeForInstr(TmpInst, Fixups, STI);
  support::endian::write(OS, Binary, support::little);
  // @} MYRISCVXMCCodeEmitter_cpp_expandFunctionCall_JALR
}
// @} MYRISCVXMCCodeEmitter_cpp_expandFunctionCall


// @{ MYRISCVXMCCodeEmitter_cpp_encodeInstruction_expandFunctionCall
// @{ MYRISCVXMCCodeEmitter_cpp_encodeInstruction
/// encodeInstruction - Emit the instruction.
/// Size the instruction (currently only 4 bytes)
void MYRISCVXMCCodeEmitter::
encodeInstruction(const MCInst &MI, raw_ostream &OS,
                  SmallVectorImpl<MCFixup> &Fixups,
                  const MCSubtargetInfo &STI) const
{
  if (MI.getOpcode() == MYRISCVX::PseudoCALLReg ||
      MI.getOpcode() == MYRISCVX::PseudoTAILCALLReg) {
    expandFunctionCall(MI, OS, Fixups, STI);
    return;
  }
  // @} MYRISCVXMCCodeEmitter_cpp_encodeInstruction_expandFunctionCall

  uint32_t Binary = getBinaryCodeForInstr(MI, Fixups, STI);

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  uint64_t TSFlags = Desc.TSFlags;

  // Pseudo instructions don't get encoded and shouldn't be here
  // in the first place!
  if ((TSFlags & MYRISCVXII::FormMask) == MYRISCVXII::Pseudo)
    llvm_unreachable("Pseudo opcode found in encodeInstruction()");

  // For now all instructions are 4 bytes
  int Size = 4; // FIXME: Have Desc.getSize() return the correct value!

  EmitInstruction(Binary, Size, OS);
}
// @} MYRISCVXMCCodeEmitter_cpp_encodeInstruction


// @{ MYRISCVXMCCodeEmitter_getBranch12TargetOpValue
/// getBranch12TargetOpValue - Return binary encoding of the branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MYRISCVXMCCodeEmitter::
getBranch12TargetOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    unsigned Res = MO.getImm();
    assert((Res & 1) == 0 && "LSB is non-zero");
    return Res >> 1;
  }

  return getExprOpValue (MI, MO.getExpr(), Fixups, STI);
}
// @} MYRISCVXMCCodeEmitter_getBranch12TargetOpValue

// @{ MYRISCVXMCCodeEmitter_getBranch20TargetOpValue
/// getBranch20TargetOpValue - Return binary encoding of the branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned MYRISCVXMCCodeEmitter::
getBranch20TargetOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm()) {
    unsigned Res = MO.getImm();
    assert((Res & 1) == 0 && "LSB is non-zero");
    return Res >> 1;
  }

  return getExprOpValue (MI, MO.getExpr(), Fixups, STI);
}
// @} MYRISCVXMCCodeEmitter_getBranch20TargetOpValue


// @{ MYRISCVXMCCodeEmitter_getExprOpValue_Head
// @{ MYRISCVXMCCodeEmitter_getExprOpValue_Fixup
// @{ MYRISCVXMCCodeEmitter_getExprOpValue
unsigned MYRISCVXMCCodeEmitter::
getExprOpValue(const MCInst &MI, const MCExpr *Expr,
               SmallVectorImpl<MCFixup> &Fixups,
               const MCSubtargetInfo &STI) const {
  // @} MYRISCVXMCCodeEmitter_getExprOpValue_Head
  MCExpr::ExprKind Kind = Expr->getKind();
  // @{ MYRISCVXMCCodeEmitter_getExprOpValue_Fixup ...
  if (Kind == MCExpr::Constant) {
    return cast<MCConstantExpr>(Expr)->getValue();
  }

  if (Kind == MCExpr::Binary) {
    unsigned Res = getExprOpValue(MI, cast<MCBinaryExpr>(Expr)->getLHS(), Fixups, STI);
    Res += getExprOpValue(MI, cast<MCBinaryExpr>(Expr)->getRHS(), Fixups, STI);
    return Res;
  }

  MCInstrDesc const &Desc = MCII.get(MI.getOpcode());
  unsigned MIFrm = Desc.TSFlags & MYRISCVXII::FormMask;
  MYRISCVX::Fixups FixupKind = MYRISCVX::Fixups(0);

  // @} MYRISCVXMCCodeEmitter_getExprOpValue_Fixup ...
  if (Kind == MCExpr::Target) {
    const MYRISCVXMCExpr *MYRISCVXExpr = cast<MYRISCVXMCExpr>(Expr);

    switch (MYRISCVXExpr->getKind()) {
      default: llvm_unreachable("Unsupported fixup kind for target expression!");
      case MYRISCVXMCExpr::CEK_None:
        llvm_unreachable("Unhandled fixup kind!");
      case MYRISCVXMCExpr::CEK_LO12_I:
        FixupKind = MYRISCVX::fixup_MYRISCVX_LO12_I;
        break;
      case MYRISCVXMCExpr::CEK_LO12_S:
        FixupKind = MYRISCVX::fixup_MYRISCVX_LO12_S;
        break;
      case MYRISCVXMCExpr::CEK_HI20:
        FixupKind = MYRISCVX::fixup_MYRISCVX_HI20;
        break;
      case MYRISCVXMCExpr::CEK_CALL:
        FixupKind = MYRISCVX::fixup_MYRISCVX_CALL;
        break;
      case MYRISCVXMCExpr::CEK_GOT_HI20:
        FixupKind = MYRISCVX::fixup_MYRISCVX_GOT_HI20;
        break;
      case MYRISCVXMCExpr::CEK_PCREL_HI20:
        FixupKind = MYRISCVX::fixup_MYRISCVX_PCREL_HI20;
        break;
      case MYRISCVXMCExpr::CEK_PCREL_LO12_I:
        FixupKind = MYRISCVX::fixup_MYRISCVX_PCREL_LO12_I;
        break;
      case MYRISCVXMCExpr::CEK_PCREL_LO12_S:
        FixupKind = MYRISCVX::fixup_MYRISCVX_PCREL_LO12_S;
        break;
    } // switch
  } else if (Kind == MCExpr::SymbolRef &&
             cast<MCSymbolRefExpr>(Expr)->getKind() == MCSymbolRefExpr::VK_None) {
    if (Desc.getOpcode() == MYRISCVX::JAL) {
      FixupKind = MYRISCVX::fixup_MYRISCVX_JAL;
    } else if (MIFrm == MYRISCVXII::FrmB) {
      FixupKind = MYRISCVX::fixup_MYRISCVX_BRANCH;
    } else {
      llvm_unreachable("Invalid Kind. Error");
    }
  }
  // @} MYRISCVXMCCodeEmitter_getExprOpValue_Fixup

  Fixups.push_back(MCFixup::create(0, Expr, MCFixupKind(FixupKind), MI.getLoc()));

  return 0;
}
// @} MYRISCVXMCCodeEmitter_getExprOpValue


// @{ MYRISCVXMCCodeEmitter_getMachineOpValue
/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned MYRISCVXMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups,
                  const MCSubtargetInfo &STI) const {
  if (MO.isReg()) {
    unsigned Reg = MO.getReg();
    unsigned RegNo = Ctx.getRegisterInfo()->getEncodingValue(Reg);
    return RegNo;
  } else if (MO.isImm()) {
    return static_cast<unsigned>(MO.getImm());
  }
  // MO must be an Expr.
  assert(MO.isExpr());
  return getExprOpValue(MI, MO.getExpr(),Fixups, STI);
}
// @} MYRISCVXMCCodeEmitter_getMachineOpValue


#include "MYRISCVXGenMCCodeEmitter.inc"
