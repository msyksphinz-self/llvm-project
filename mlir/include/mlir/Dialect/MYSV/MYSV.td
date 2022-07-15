//===- MYSVBase.td - Base defs for arith dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MYSV
#define MYSV

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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


def ConstantOp : MYSV_Op<"constant", [NoSideEffect]> {
  // The constant operation takes an attribute as the only input.
  // `SI64Attr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins SI64Attr:$value);

  // The constant operation returns a single value of TensorType.
  // I64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs I64);

  // // Add custom build methods for the constant operation. These method populates
  // // the `state` that MLIR uses to create operations, i.e. these are used when
  // // using `builder.create<ConstantOp>(...)`.
  // let builders = [
  //   // Build a constant with a given constant tensor value.
  //   OpBuilder<(ins "IntegerAttr":$value), [{
  //     build($_builder, $_state, value.getType(), value);
  //   }]>,
  //
  //   // Build a constant with a given constant floating-point value.
  //   OpBuilder<(ins "int":$value)>
  // ];
  //
  // // Invoke a static verify method to verify this return operation.
  // let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : MYSV_Op<"mul", [NoSideEffect]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);

  // // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  //
  // // Allow building a MulOp with from the two input operands.
  // let builders = [
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  // ];
}



//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : MYSV_Op<"add", [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);

  // // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  //
  // // Allow building an AddOp with from the two input operands.
  // let builders = [
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  // ];
}


//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : MYSV_Op<"sub", [
    NoSideEffect, IsolatedFromAbove
  ]> {
  let summary = "element-wise subition operation";
  let description = [{
    The "sub" operation performs element-wise subition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  //
  // // Allow building an SubOp with from the two input operands.
  // let builders = [
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  // ];
}

#endif // MYSV
