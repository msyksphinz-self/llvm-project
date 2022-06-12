add_mlir_dialect(MYSV mysv)
add_mlir_doc(MYSV MYSV Dialects/ -gen-dialect-doc -dialect=mysv)

include_directories(.)

set(LLVM_TARGET_DEFINITIONS MYSV.td)
mlir_tablegen(MYSVConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRMYSVConversionsIncGen)
