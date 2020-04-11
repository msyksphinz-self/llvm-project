//===-- MYRISCVXAsmParser.cpp - Parse MYRISCVX assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVX.h"

#include "MCTargetDesc/MYRISCVXMCExpr.h"
#include "MCTargetDesc/MYRISCVXMCTargetDesc.h"
#include "MYRISCVXRegisterInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-asm-parser"

namespace {
class MYRISCVXAssemblerOptions {
 public:
  MYRISCVXAssemblerOptions():
      reorder(true), macro(true) {
  }

  bool isReorder() {return reorder;}
  void setReorder() {reorder = true;}
  void setNoreorder() {reorder = false;}

  bool isMacro() {return macro;}
  void setMacro() {macro = true;}
  void setNomacro() {macro = false;}

 private:
  bool reorder;
  bool macro;
};
}

// @{ MYRISCVXAsmParser_MYRISCVXAsmParser
namespace {
class MYRISCVXAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;
  MYRISCVXAssemblerOptions Options;

#define GET_ASSEMBLER_HEADER
#include "MYRISCVXGenAsmMatcher.inc"

  SMLoc getLoc() const { return getParser().getTok().getLoc(); }

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                     SMLoc &EndLoc) override;
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  OperandMatchResultTy parseMemOperand(OperandVector &);

  bool ParseOperand(OperandVector &Operands, StringRef Mnemonic);

  bool needsExpansion(MCInst &Inst);

  void expandInstruction(MCInst &Inst, SMLoc IDLoc,
                         SmallVectorImpl<MCInst> &Instructions,
                         MCStreamer &Out);
  void expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                     SmallVectorImpl<MCInst> &Instructions);
  void expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                            SmallVectorImpl<MCInst> &Instructions,
                            MCStreamer &Out);
  bool reportParseError(StringRef ErrorMsg);

  const MCExpr *evaluateRelocExpr(const MCExpr *Expr, StringRef RelocStr);

  bool parseDirectiveSet();

  OperandMatchResultTy parseImmediate(OperandVector &Operands);
  OperandMatchResultTy parseOperandWithModifier(OperandVector &Operands);
  OperandMatchResultTy parseRegister(OperandVector &Operands);

  bool parseSetAtDirective();
  bool parseSetNoAtDirective();
  bool parseSetMacroDirective();
  bool parseSetNoMacroDirective();
  bool parseSetReorderDirective();
  bool parseSetNoReorderDirective();

 public:
  MYRISCVXAsmParser(const MCSubtargetInfo &STI, MCAsmParser &parser,
                    const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), Parser(parser) {
    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
  }

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

};
}
// @} MYRISCVXAsmParser_MYRISCVXAsmParser


namespace {

/// MYRISCVXOperand - Instances of this class represent a parsed MYRISCVX machine
/// instruction.
class MYRISCVXOperand : public MCParsedAsmOperand {

  enum KindTy {
    k_Immediate,
    k_Memory,
    k_Register,
    k_Token
  } Kind;

 public:
  MYRISCVXOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  struct Token {
    const char *Data;
    unsigned Length;
  };
  struct PhysRegOp {
    unsigned RegNum; /// Register Number
  };
  struct ImmOp {
    const MCExpr *Val;
  };
  struct MemOp {
    unsigned Base;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct PhysRegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

  SMLoc StartLoc, EndLoc;

 public:
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const{
    // Add as immediate when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst,Expr);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst,Expr);
  }

  bool isReg() const override { return Kind == k_Register; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isToken() const override { return Kind == k_Token; }
  bool isMem() const override { return Kind == k_Memory; }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const override {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Base;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_Memory) && "Invalid access!");
    return Mem.Off;
  }

  static std::unique_ptr<MYRISCVXOperand> CreateToken(StringRef Str, SMLoc S) {
    auto Op = make_unique<MYRISCVXOperand>(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  /// Internal constructor for register kinds
  static std::unique_ptr<MYRISCVXOperand> createReg(unsigned RegNum, SMLoc S,
                                                    SMLoc E) {
    auto Op = make_unique<MYRISCVXOperand>(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<MYRISCVXOperand> createImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = make_unique<MYRISCVXOperand>(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<MYRISCVXOperand> CreateMem(unsigned Base, const MCExpr *Off,
                                                    SMLoc S, SMLoc E) {
    auto Op = make_unique<MYRISCVXOperand>(k_Memory);
    Op->Mem.Base = Base;
    Op->Mem.Off = Off;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
      case k_Immediate:
        OS << "Imm<";
        OS << *Imm.Val;
        OS << ">";
        break;
      case k_Memory:
        OS << "Mem<";
        OS << Mem.Base;
        OS << ", ";
        OS << *Mem.Off;
        OS << ">";
        break;
      case k_Register:
        OS << "Register<" << Reg.RegNum << ">";
        break;
      case k_Token:
        OS << Tok.Data;
        break;
    }
  }
};
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MYRISCVXGenAsmMatcher.inc"

void printMYRISCVXOperands(OperandVector &Operands) {
  for (size_t i = 0; i < Operands.size(); i++) {
    MYRISCVXOperand* op = static_cast<MYRISCVXOperand*>(&*Operands[i]);
    assert(op != nullptr);
    LLVM_DEBUG(dbgs() << "<" << *op << ">");
  }
  LLVM_DEBUG(dbgs() << "\n");
}


// @{ MYRISCVXAsmParser_needsExpansion
bool MYRISCVXAsmParser::needsExpansion(MCInst &Inst) {
  switch(Inst.getOpcode()) {
    case MYRISCVX::LoadImm32Reg:
    case MYRISCVX::LoadAddr32Imm:
      return true;
    default:
      return false;
  }
}
// @} MYRISCVXAsmParser_needsExpansion


void MYRISCVXAsmParser::expandInstruction(MCInst &Inst, SMLoc IDLoc,
                                          SmallVectorImpl<MCInst> &Instructions,
                                          MCStreamer &Out){
  switch(Inst.getOpcode()) {
    case MYRISCVX::LoadImm32Reg:
      return expandLoadImm(Inst, IDLoc, Instructions);
    case MYRISCVX::LoadAddr32Imm:
      return expandLoadAddressImm(Inst,IDLoc,Instructions, Out);
  }
}


// @{ MYRISCVXAsmParser_expandLoadImm
void MYRISCVXAsmParser::expandLoadImm(MCInst &Inst, SMLoc IDLoc,
                                      SmallVectorImpl<MCInst> &Instructions){
  MCInst tmpInst;
  const MCOperand &ImmOp = Inst.getOperand(1);
  assert(ImmOp.isImm() && "expected immediate operand kind");
  const MCOperand &RegOp = Inst.getOperand(0);
  assert(RegOp.isReg() && "expected register operand kind");

  int ImmValue = ImmOp.getImm();
  tmpInst.setLoc(IDLoc);
  if ( -2048 <= ImmValue && ImmValue < 2048) {
    // for 0 <= j <= 65535.
    // li d,j => ori d,$zero,j
    tmpInst.setOpcode(MYRISCVX::ADDI);
    tmpInst.addOperand(MCOperand::createReg(RegOp.getReg()));
    tmpInst.addOperand(
        MCOperand::createReg(MYRISCVX::ZERO));
    tmpInst.addOperand(MCOperand::createImm(ImmValue));
    Instructions.push_back(tmpInst);
  } else {
    // for any other value of j that is representable as a 32-bit integer.
    // li d,j => lui d,hi16(j)
    //           ori d,d,lo16(j)
    tmpInst.setOpcode(MYRISCVX::LUI);
    tmpInst.addOperand(MCOperand::createReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::createImm((((ImmValue + 0x800) & 0xfffff000) >> 12) & 0xffffff));
    Instructions.push_back(tmpInst);
    tmpInst.clear();
    tmpInst.setOpcode(MYRISCVX::ADDI);
    tmpInst.addOperand(MCOperand::createReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::createReg(RegOp.getReg()));
    tmpInst.addOperand(MCOperand::createImm(ImmValue & 0x00fff));
    tmpInst.setLoc(IDLoc);
    Instructions.push_back(tmpInst);
  }
}
// @} MYRISCVXAsmParser_expandLoadImm


// @{MYRISCVXAsmParser_expandLoadAddressImm
void MYRISCVXAsmParser::expandLoadAddressImm(MCInst &Inst, SMLoc IDLoc,
                                             SmallVectorImpl<MCInst> &Instructions,
                                             MCStreamer &Out){
  MCContext &Ctx = getContext();

  MCSymbol *TmpLabel = Ctx.createTempSymbol(
      "pcrel_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
  Out.EmitLabel(TmpLabel);

  MCOperand DestReg = Inst.getOperand(0);
  const MYRISCVXMCExpr *Symbol = MYRISCVXMCExpr::create(
      MYRISCVXMCExpr::CEK_GOT_HI20, Inst.getOperand(1).getExpr(), Ctx);

  MCInst tmpInst0 = MCInstBuilder(MYRISCVX::LUI)
      .addOperand(DestReg)
      .addExpr(Symbol);
  Instructions.push_back(tmpInst0);

  const MCExpr *RefToLinkTmpLabel =
      MYRISCVXMCExpr::create(MYRISCVXMCExpr::CEK_PCREL_LO12_I,
                             MCSymbolRefExpr::create(TmpLabel, Ctx),
                             Ctx);

  MCInst tmpInst1 = MCInstBuilder(MYRISCVX::ADDI)
      .addOperand(DestReg)
      .addOperand(DestReg)
      .addExpr(RefToLinkTmpLabel);
  Instructions.push_back(tmpInst1);

}
// @}MYRISCVXAsmParser_expandLoadAddressImm

// @{ MYRISCVXAsmParser_MatchAndEmitInstruction
bool MYRISCVXAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                                OperandVector &Operands,
                                                MCStreamer &Out,
                                                uint64_t &ErrorInfo,
                                                bool MatchingInlineAsm) {
  // @{ MYRISCVXAsmParser_MatchAndEmitInstruction_MatchInstructionImpl
  MCInst Inst;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                              MatchingInlineAsm);
  switch (MatchResult) {
    default: break;
    case Match_Success: {
      if (needsExpansion(Inst)) {
        SmallVector<MCInst, 4> Instructions;
        expandInstruction(Inst, IDLoc, Instructions, Out);
        for(unsigned i =0; i < Instructions.size(); i++){
          Out.EmitInstruction(Instructions[i], getSTI());
        }
      } else {
        Inst.setLoc(IDLoc);
        Out.EmitInstruction(Inst, getSTI());
      }
      return false;
    }
      // @} MYRISCVXAsmParser_MatchAndEmitInstruction_MatchInstructionImpl
    case Match_MissingFeature:
      Error(IDLoc, "instruction requires a CPU feature not currently enabled");
      return true;
    case Match_InvalidOperand: {
      SMLoc ErrorLoc = IDLoc;
      if (ErrorInfo != ~0U) {
        if (ErrorInfo >= Operands.size())
          return Error(IDLoc, "too few operands for instruction");

        ErrorLoc = ((MYRISCVXOperand &)*Operands[ErrorInfo]).getStartLoc();
        if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
      }

      return Error(ErrorLoc, "invalid operand for instruction");
    }
    case Match_MnemonicFail:
      return Error(IDLoc, "invalid instruction");
  }
  return true;
}
// @} MYRISCVXAsmParser_MatchAndEmitInstruction


bool MYRISCVXAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                      SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  StringRef Name = getLexer().getTok().getIdentifier();

  if (!MatchRegisterName(Name)) {
    getParser().Lex(); // Eat identifier token.n
    return false;
  }

  return Error(StartLoc, "invalid register name");
}


// @{ MYRISCVXAsmParser_ParseOperand
bool MYRISCVXAsmParser::ParseOperand(OperandVector &Operands,
                                     StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  // OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);
  // if (ResTy == MatchOperand_Success)
  //   return false;

  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  // if (ResTy == MatchOperand_ParseFail)
  //   return true;

  LLVM_DEBUG(dbgs() << ".. Generic Parser\n");

  // Attempt to parse token as a register.
  if (parseRegister(Operands) == MatchOperand_Success)
    return false;

  // Attempt to parse token as an immediate
  if (parseImmediate(Operands) == MatchOperand_Success) {
    // Parse memory base register if present
    if (getLexer().is(AsmToken::LParen)) {
      return parseMemOperand(Operands) != MatchOperand_Success;
    }
    return false;
  }

  LLVM_DEBUG(dbgs() << "unknown operand\n");
  return true;
}
// @} MYRISCVXAsmParser_ParseOperand


// @{ MYRISCVXAsmParser_ParseRegister
OperandMatchResultTy MYRISCVXAsmParser::parseRegister(OperandVector &Operands)
{
  switch (getLexer().getKind()) {
    default:
      return MatchOperand_NoMatch;
    case AsmToken::Identifier:
      StringRef Name = getLexer().getTok().getIdentifier();
      unsigned RegNo = MatchRegisterName(Name);
      if (RegNo == 0) {
        RegNo = MatchRegisterAltName(Name);
        if (RegNo == 0) {
          return MatchOperand_NoMatch;
        }
      }
      SMLoc S = getLoc();
      SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);
      getLexer().Lex();
      Operands.push_back(MYRISCVXOperand::createReg(RegNo, S, E));
  }

  return MatchOperand_Success;
}
// @} MYRISCVXAsmParser_ParseRegister


// @{ MYRISCVXAsmParser_parseImmediate
OperandMatchResultTy MYRISCVXAsmParser::parseImmediate(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);
  const MCExpr *Res;

  switch (getLexer().getKind()) {
    default:
      return MatchOperand_NoMatch;
    case AsmToken::LParen:
    case AsmToken::Minus:
    case AsmToken::Plus:
    case AsmToken::Integer:
    case AsmToken::String:
    case AsmToken::Identifier:
    case AsmToken::Dollar:
      LLVM_DEBUG(dbgs() << "< parseImmediate Passed >");
      if (getParser().parseExpression(Res))
        return MatchOperand_ParseFail;
      break;
    case AsmToken::Percent:
      return parseOperandWithModifier(Operands);
  }

  Operands.push_back(MYRISCVXOperand::createImm(Res, S, E));
  return MatchOperand_Success;
}
// @} MYRISCVXAsmParser_parseImmediate


// @{ MYRISCVXAsmParser_parseOperandWithModifier
OperandMatchResultTy
MYRISCVXAsmParser::parseOperandWithModifier(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);

  if (getLexer().getKind() != AsmToken::Percent) {
    Error(getLoc(), "expected '%' for operand modifier");
    return MatchOperand_ParseFail;
  }

  getParser().Lex(); // Eat '%'

  if (getLexer().getKind() != AsmToken::Identifier) {
    Error(getLoc(), "expected valid identifier for operand modifier");
    return MatchOperand_ParseFail;
  }
  StringRef Identifier = getParser().getTok().getIdentifier();
  MYRISCVXMCExpr::MYRISCVXExprKind VK = MYRISCVXMCExpr::getVariantKindForName(Identifier);
  if (VK == MYRISCVXMCExpr::CEK_None) {
    Error(getLoc(), "unrecognized operand modifier");
    return MatchOperand_ParseFail;
  }

  getParser().Lex(); // Eat the identifier
  if (getLexer().getKind() != AsmToken::LParen) {
    Error(getLoc(), "expected '('");
    return MatchOperand_ParseFail;
  }
  getParser().Lex(); // Eat '('

  const MCExpr *SubExpr;
  if (getParser().parseParenExpression(SubExpr, E)) {
    return MatchOperand_ParseFail;
  }

  const MCExpr *ModExpr = MYRISCVXMCExpr::create(VK, SubExpr, getContext());
  Operands.push_back(MYRISCVXOperand::createImm(ModExpr, S, E));
  return MatchOperand_Success;
}
// @} MYRISCVXAsmParser_parseOperandWithModifier


// @{ MYRISCVXAsmParser_parseMemOperand
OperandMatchResultTy
MYRISCVXAsmParser::parseMemOperand(OperandVector &Operands) {

  const MCExpr *IdVal = 0;
  SMLoc S;
  // first operand is the offset
  S = Parser.getTok().getLoc();

  const AsmToken &Tok = Parser.getTok(); // get next token
  if (Tok.isNot(AsmToken::LParen)) {
    Error(Parser.getTok().getLoc(), "'(' expected");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // Eat '(' token.
  Operands.push_back(MYRISCVXOperand::CreateToken("(", getLoc()));

  if (parseRegister(Operands)) {
    Error(Parser.getTok().getLoc(), "unexpected token in operand");
  }

  const AsmToken &Tok2 = Parser.getTok(); // get next token
  if (Tok2.isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "')' expected");
    return MatchOperand_ParseFail;
  }

  Parser.Lex(); // Eat ')' token.
  Operands.push_back(MYRISCVXOperand::CreateToken(")", getLoc()));

  if (!IdVal)
    IdVal = MCConstantExpr::create(0, getContext());

  return MatchOperand_Success;
}
// @} MYRISCVXAsmParser_parseMemOperand

// @{ MYRISCVXAsmParser_ParseInstruction
bool MYRISCVXAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 OperandVector &Operands) {

  // Create the leading tokens for the mnemonic, split by '.' characters.
  size_t Start = 0, Next = Name.find('.');
  StringRef Mnemonic = Name.slice(Start, Next);

  Operands.push_back(MYRISCVXOperand::CreateToken(Mnemonic, NameLoc));

  // Read the remaining operands.
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (ParseOperand(Operands, Name)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }

    while (getLexer().is(AsmToken::Comma) ) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (ParseOperand(Operands, Name)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        return Error(Loc, "unexpected token in argument list");
      }
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    Parser.eatToEndOfStatement();
    return Error(Loc, "unexpected token in argument list");
  }

  Parser.Lex(); // Consume the EndOfStatement
  return false;
}
// @} MYRISCVXAsmParser_ParseInstruction


bool MYRISCVXAsmParser::reportParseError(StringRef ErrorMsg) {
  SMLoc Loc = getLexer().getLoc();
  Parser.eatToEndOfStatement();
  return Error(Loc, ErrorMsg);
}

bool MYRISCVXAsmParser::parseSetReorderDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setReorder();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MYRISCVXAsmParser::parseSetNoReorderDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setNoreorder();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MYRISCVXAsmParser::parseSetMacroDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("unexpected token in statement");
    return false;
  }
  Options.setMacro();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}

bool MYRISCVXAsmParser::parseSetNoMacroDirective() {
  Parser.Lex();
  // if this is not the end of the statement, report error
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  if (Options.isReorder()) {
    reportParseError("`noreorder' must be set before `nomacro'");
    return false;
  }
  Options.setNomacro();
  Parser.Lex(); // Consume the EndOfStatement
  return false;
}
bool MYRISCVXAsmParser::parseDirectiveSet() {

  // get next token
  const AsmToken &Tok = Parser.getTok();

  if (Tok.getString() == "reorder") {
    return parseSetReorderDirective();
  } else if (Tok.getString() == "noreorder") {
    return parseSetNoReorderDirective();
  } else if (Tok.getString() == "macro") {
    return parseSetMacroDirective();
  } else if (Tok.getString() == "nomacro") {
    return parseSetNoMacroDirective();
  }
  return true;
}

bool MYRISCVXAsmParser::ParseDirective(AsmToken DirectiveID) {

  if (DirectiveID.getString() == ".ent") {
    // ignore this directive for now
    Parser.Lex();
    return false;
  }

  if (DirectiveID.getString() == ".end") {
    // ignore this directive for now
    Parser.Lex();
    return false;
  }

  if (DirectiveID.getString() == ".frame") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (DirectiveID.getString() == ".set") {
    return parseDirectiveSet();
  }

  if (DirectiveID.getString() == ".fmask") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (DirectiveID.getString() == ".mask") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  if (DirectiveID.getString() == ".gpword") {
    // ignore this directive for now
    Parser.eatToEndOfStatement();
    return false;
  }

  return true;
}

extern "C" void LLVMInitializeMYRISCVXAsmParser() {
  RegisterMCAsmParser<MYRISCVXAsmParser> X(getTheMYRISCVX32Target());
}
