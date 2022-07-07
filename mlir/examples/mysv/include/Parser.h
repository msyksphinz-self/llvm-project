//===- Parser.h - MYSV Language Parser -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the MYSV language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef MYSV_PARSER_H
#define MYSV_PARSER_H

#include "AST.h"
#include "Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <utility>
#include <vector>

namespace mysv {

/// This is a simple recursive parser for the MYSV language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken();   // prime the lexer
    // Parse functions one at a time and accumulate in this vector.
    std::vector<AssignExprAST> functions;
    while (auto f = parseAssign()) {
      functions.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof)
        break;
    }

    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<NumberExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return result;
  }

  /// identifierexpr
  ///   ::= identifier
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string name(lexer.getId());

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    return std::make_unique<VarExprAST>(std::move(loc), name);
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {
      default:
        llvm::errs() << "unknown token '" << lexer.getCurToken()
                     << "' when expecting an expression\n";
        return nullptr;
      case tok_identifier:
        return parseIdentifierExpr();
      case tok_number:
        return parseNumberExpr();
    }
  }


  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int tokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrec < exprPrec)
        return lhs;

      // Okay, we know this is a binop.
      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                            std::move(lhs), std::move(rhs));
    }
  }


  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }


  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpr() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;
    return parseBinOpRHS(0, std::move(lhs));
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= assign identifier = expr
  std::unique_ptr<AssignExprAST> parseAssign() {
    if (lexer.getCurToken() != tok_assign) {
      return parseError<AssignExprAST>("assign", "to begin assign");
    }
    auto loc = lexer.getLastLocation();
    lexer.getNextToken();  // eat var
    if (lexer.getCurToken() != tok_identifier) {
      return parseError<AssignExprAST>("identifier", "after assign notification");
    }
    std::string id(lexer.getId());
    lexer.getNextToken();  // eat id

    lexer.consume(Token('='));

    auto expr = parseExpr();

    lexer.consume(Token(';'));
    return std::make_unique<AssignExprAST>(std::move(loc), std::move(id), std::move(expr));
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace mysv

#endif // MYSV_PARSER_H
