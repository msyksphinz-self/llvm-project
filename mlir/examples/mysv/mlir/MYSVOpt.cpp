//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect.h"
#include <numeric>

#define DEBUG_TYPE "mysv-opt"

using namespace mlir;
using namespace mysv;


/// This is an example of a c++ rewrite pattern for the SubOp. It
/// optimizes the following scenario
struct MinusSameValue : public mlir::OpRewritePattern<SubOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  MinusSameValue(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(SubOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value op0 = op.getOperand(0);
    mlir::Value op1 = op.getOperand(1);

    LLVM_DEBUG(llvm::dbgs() << "MinusSameValue::matchAndReWrite\n");

    mlir::Type elementType = rewriter.getI64Type();

    mlir::Value zero = rewriter.create<ConstantOp>(op.getLoc(), elementType, /*value=*/0);

    rewriter.replaceOp(op, zero);
    return success();
    // return failure();
  }
};

/// Register our patterns as "canonicalization" patterns on the SubOp so
/// that they can be picked up by the Canonicalization framework.
void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MinusSameValue>(context);
}
