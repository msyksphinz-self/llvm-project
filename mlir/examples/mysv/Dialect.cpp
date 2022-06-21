//===- Dialect.cpp - Mlir IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Mlir IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mysv;

#include "mlir/Dialect/MYSV/MYSVDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MlirDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void MYSVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MYSV/MYSV.cpp.inc"
      >();
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MYSV/MYSV.cpp.inc"
