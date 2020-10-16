# @{MYRISCVX_CMakeLists.txt
add_llvm_component_group(MYRISCVX)
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# @{MYRISCVX_CMakeLists_TableGen
# MYRISCVXGenRegisterInfo.incを生成する
# tablegenを使用してMYRISCVXRegisterInfo.tdからMYRISCVXGenRegisterInfo.incが生成される
# @{ MYRISCVX_CMakeLists_CallingConv_TableGen
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MYRISCVXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MYRISCVXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MYRISCVXGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM MYRISCVXGenDAGISel.inc -gen-dag-isel)
# MYRISCVCallingConv.tdからMYRISCVXGenCallingConv.incを生成するためのtablegenを追加
tablegen(LLVM MYRISCVXGenCallingConv.inc -gen-callingconv)
# @} MYRISCVX_CMakeLists_CallingConv_TableGen
# @}MYRISCVX_CMakeLists_TableGen

tablegen(LLVM MYRISCVXGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM MYRISCVXGenMCCodeEmitter.inc -gen-emitter)

tablegen(LLVM MYRISCVXGenAsmWriter.inc -gen-asm-writer)

# MYRISCVXCommonTableGenを定義しなければならない
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen はLLVMBuild.txtのMYRISCVXCodeGenと一致しなければならない
add_llvm_target(MYRISCVXCodeGen
  MYRISCVXTargetMachine.cpp
  MYRISCVXSubtarget.cpp
  MYRISCVXRegisterInfo.cpp
  MYRISCVXFrameLowering.cpp
  MYRISCVXInstrInfo.cpp
  MYRISCVXISelLowering.cpp
  MYRISCVXMachineFunction.cpp
  MYRISCVXTargetObjectFile.cpp
  MYRISCVXAsmPrinter.cpp
  MYRISCVXMCInstLower.cpp
  MYRISCVXISelDAGToDAG.cpp
  MYRISCVXMatInt.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
# @{MYRISCVX_CMakeLists.txt
