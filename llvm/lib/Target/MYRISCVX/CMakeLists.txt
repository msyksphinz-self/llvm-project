# @{MYRISCVX_CMakeLists.txt
add_llvm_component_group(MYRISCVX)
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# MYRISCVXGenRegisterInfo.incを生成する
# このヘッダファイルはMYRISCVXRegisterInfo.tdから生成される
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)

# MYRISCVXCommonTableGenを定義しなければならない
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen はLLVMBuild.txtのMYRISCVXCodeGenと一致しなければならない
add_llvm_target(MYRISCVXCodeGen
  MYRISCVXTargetMachine.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
# @{MYRISCVX_CMakeLists.txt
