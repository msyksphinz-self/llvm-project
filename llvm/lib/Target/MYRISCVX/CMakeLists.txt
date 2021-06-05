# @{MYRISCVX_CMakeLists.txt
add_llvm_component_group(MYRISCVX)
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# @{MYRISCVX_CMakeLists_TableGen
# MYRISCVXGenRegisterInfo.incを生成する
# tablegenを使用してMYRISCVXRegisterInfo.tdからMYRISCVXGenRegisterInfo.incが生成される
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MYRISCVXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MYRISCVXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MYRISCVXGenMCPseudoLowering.inc -gen-pseudo-lowering)
# @}MYRISCVX_CMakeLists_TableGen

# MYRISCVXCommonTableGenを定義しなければならない
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen はLLVMBuild.txtのMYRISCVXCodeGenと一致しなければならない
add_llvm_target(MYRISCVXCodeGen
  MYRISCVXTargetMachine.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
# @{MYRISCVX_CMakeLists.txt
