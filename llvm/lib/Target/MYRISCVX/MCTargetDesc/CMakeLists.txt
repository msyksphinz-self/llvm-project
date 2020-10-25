# MCTargetDesc/CMakeLists.txt
add_llvm_library(LLVMMYRISCVXDesc
  MYRISCVXMCTargetDesc.cpp
  MYRISCVXABIInfo.cpp
  MYRISCVXMCAsmInfo.cpp
  MYRISCVXTargetStreamer.cpp
  MYRISCVXTargetStreamer.cpp
  MYRISCVXInstPrinter.cpp
)
