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

#include "mysv/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <stdint.h>
#include <utility>
#include <vector>

namespace mysv {

  /// A variable type with shape information.
  struct VarType {
    std::vector<int64_t> shape;
  };

  /// Base class for all expression nodes.
  class ExprAST {
 public:
 ExprAST(Location location)
     : location(std::move(location)) {}
    virtual ~ExprAST() = default;

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
  };



  /// Expression class for assignment.
  class AssignExprAST : public ExprAST {
    std::string name;
    std::unique_ptr<ExprAST> initVal;

 public:
 AssignExprAST(Location loc, llvm::StringRef name, std::unique_ptr<ExprAST> initVal)
     : ExprAST(std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

    llvm::StringRef getName() { return name; }
    ExprAST *getInitVal() { return initVal.get(); }
  };
