# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMMYRISCVXDesc
  MYRISCVXInstPrinter.cpp
  MYRISCVXMCTargetDesc.cpp
  MYRISCVXABIInfo.cpp
  MYRISCVXMCAsmInfo.cpp
  MYRISCVXMCExpr.cpp
  MYRISCVXTargetStreamer.cpp
)
