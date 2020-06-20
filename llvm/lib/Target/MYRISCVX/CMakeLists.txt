# @{MYRISCVX_CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# @{MYRISCVX_CMakeLists_TableGen
# MYRISCVXGenRegisterInfo.incを生成する。
# このヘッダファイルはMYRISCVXRegisterInfo.tdから生成される。
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MYRISCVXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MYRISCVXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MYRISCVXGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM MYRISCVXGenCallingConv.inc -gen-callingconv)
# @}MYRISCVX_CMakeLists_TableGen

# MYRISCVXCommonTableGen must be defined
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen should match with LLVMBuild.txt MYRISCVXCodeGen
add_llvm_target(MYRISCVXCodeGen
  MYRISCVXTargetMachine.cpp
  MYRISCVXSubtarget.cpp
  MYRISCVXRegisterInfo.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
# @{MYRISCVX_CMakeLists.txt
