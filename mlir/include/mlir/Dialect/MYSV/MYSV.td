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

  // let hasConstantMaterializer = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

// Base class for MYSV dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class MYSV_Op<string mnemonic, list<Trait> traits = []> :
    Op<MYSV_Dialect, mnemonic, traits>;


def ConstantOp : MYSV_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `SI64Attr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins SI64Attr:$value);

  // The constant operation returns a single value of TensorType.
  // I64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs I64);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "IntegerAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "int":$value)>
  ];

  // Invoke a static verify method to verify this return operation.
  let hasVerifier = 1;
}

#endif // MYSV
