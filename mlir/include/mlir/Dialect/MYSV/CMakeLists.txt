add_mlir_dialect(MYSV mysv)
add_mlir_doc(MYSV MYSV Dialects/ -gen-dialect-doc -dialect=mysv)

include_directories(.)

set(LLVM_TARGET_DEFINITIONS MYSV.td)
mlir_tablegen(MYSVOps.h.inc -gen-op-decls)
mlir_tablegen(MYSVOps.cpp.inc -gen-op-defs)
# mlir_tablegen(MYSVDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(MYSVDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRMYSVConversionsIncGen)
