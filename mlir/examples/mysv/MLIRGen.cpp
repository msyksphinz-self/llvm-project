//===- MLIRGen.cpp - MLIR Generation from a MYSV AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the MYSV language.
//
//===----------------------------------------------------------------------===//

#include "MLIRGen.h"
#include "AST.h"
#include "Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::mysv;
using namespace mysv;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the MYSV AST.
///
/// This will emit operations that are specific to the MYSV language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
 public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Mysv module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (AssignExprAST &a : moduleAST)
      mlirGen(a);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the MYSV operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::Value mlirGen(AssignExprAST &assignAST) {
    // // Create a scope in the symbol table to hold variable declarations.
    // ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());

    auto *expr = assignAST.getExpr();
    if (!expr) {
      emitError(loc(assignAST.loc()),
                "missing exprializer in variable declaration");
      return nullptr;
    }

    auto assign = mlirGen(*expr);
    if (!assign)
      return nullptr;

    // Register the assign in the symbol table.
    if (failed(declare(assignAST.getName(), assign)))
      return nullptr;

    return assign;
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
      case mysv::ExprAST::Expr_Var:
        return mlirGen(cast<VarExprAST>(expr));
      case mysv::ExprAST::Expr_Num:
        return mlirGen(cast<NumberExprAST>(expr));
      case mysv::ExprAST::Expr_BinOp:
        return mlirGen(cast<BinaryExprAST>(expr));
      default:
        emitError(loc(expr.loc()))
            << "MLIR codegen encountered an unhandled expr kind '"
            << Twine(expr.getKind()) << "'";
        return nullptr;
    }
  }


  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `mysv.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  mlir::Value mlirGen(NumberExprAST &lit) {
    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getI64Type();

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    // auto dataAttribute = mlir::DenseElementsAttr::get(elementType);

    // Build the MLIR op `mysv.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), elementType, lit.getValue());
  }


  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VarExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }


  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());
    mlir::Type elementType = builder.getI64Type();

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, elementType, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, elementType, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

 private:
  /// A "module" matches a MYSV source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;


  /// Helper conversion for a MYSV AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }
};

} // namespace

namespace mysv {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

}  // namespace mysv
