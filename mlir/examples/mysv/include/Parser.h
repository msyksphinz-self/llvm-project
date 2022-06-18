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

    auto expr = parseNumberExpr();

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
