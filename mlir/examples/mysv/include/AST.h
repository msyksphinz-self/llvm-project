//===- AST.h - Node definition for the MYSV¨ AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the MYSV¨ language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef MYSV_AST_H
#define MYSV_AST_H

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
// #include "llvm/Support/Casting.h"
#include <stdint.h>
#include <utility>
#include <vector>

namespace mysv {

/// Base class for all expression nodes.
class ExprAST {
 public:
  enum ExprASTKind {
    Expr_Assign,
    Expr_Num,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

 private:
  const ExprASTKind kind;
  Location location;
};


/// Expression class for numeric literals like "1".
class NumberExprAST : public ExprAST {
  uint64_t val;

 public:
  NumberExprAST(Location loc, uint64_t val)
      : ExprAST(Expr_Num, std::move(loc)), val(val) {}

  uint64_t getValue() { return val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};



/// Expression class for assignment.
class AssignExprAST : public ExprAST {
  std::string name;
  std::unique_ptr<NumberExprAST> initVal;

 public:
  AssignExprAST(Location loc, llvm::StringRef name, std::unique_ptr<NumberExprAST> initVal)
      : ExprAST(Expr_Assign, std::move(loc)),
        name(name),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  NumberExprAST *getInitVal() { return initVal.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Assign; }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<AssignExprAST> assigns;

 public:
  ModuleAST(std::vector<AssignExprAST> assigns)
      : assigns(std::move(assigns)) {}

  auto begin() { return assigns.begin(); }
  auto end() { return assigns.end(); }
};

void dump(ModuleAST &);


} // namespace mysv


#endif // MYSV_AST_H
