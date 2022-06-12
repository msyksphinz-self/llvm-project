//===- MYSVBase.td - Base defs for arith dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYSV
#define MYSV

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def MYSV_Dialect : Dialect {
  let name = "mysv";
  let cppNamespace = "::mlir::mysv";
  let summary = "The My original System Verilog Dialect.";
  let description = [{
    The My original System Verilog Dialect.
  }];

  let hasConstantMaterializer = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // MYSV
