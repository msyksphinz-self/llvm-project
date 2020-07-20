# @{MYRISCVX_CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# MYRISCVXGenRegisterInfo.incを生成する。
# このヘッダファイルはMYRISCVXRegisterInfo.tdから生成される。
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)

# MYRISCVXCommonTableGen must be defined
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen should match with LLVMBuild.txt MYRISCVXCodeGen
add_llvm_target(MYRISCVXCodeGen
  MYRISCVXTargetMachine.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
# @{MYRISCVX_CMakeLists.txt
