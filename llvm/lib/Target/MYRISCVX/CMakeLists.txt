# @{MYRISCVX_CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS MYRISCVX.td)

# @{MYRISCVX_CMakeLists_TableGen
# Generate MYRISCVXGenRegisterInfo.inc and MYRISCVXGenInstrInfo.inc which included by
#  your hand code C++ files.
# MYRISCVXGenRegisterInfo.inc came from MYRISCVXRegisterInfo.td, MYRISCVXGenInstrInfo.inc
#  came from MYRISCVXInstrInfo.td.
tablegen(LLVM MYRISCVXGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MYRISCVXGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MYRISCVXGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MYRISCVXGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM MYRISCVXGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM MYRISCVXGenCallingConv.inc -gen-callingconv)
# @}MYRISCVX_CMakeLists_TableGen

tablegen(LLVM MYRISCVXGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM MYRISCVXGenMCCodeEmitter.inc -gen-emitter)
# @{CMakeLists_AsmMatcher
tablegen(LLVM MYRISCVXGenAsmMatcher.inc -gen-asm-matcher)  # For MYRISCVX AsmParser
# @}CMakeLists_AsmMatcher

tablegen(LLVM MYRISCVXGenAsmWriter.inc -gen-asm-writer)

# MYRISCVXCommonTableGen must be defined
add_public_tablegen_target(MYRISCVXCommonTableGen)

# MYRISCVXCodeGen should match with LLVMBuild.txt MYRISCVXCodeGen
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

# Should match with "subdirectories =  MCTargetDesc TargetInfo" in LLVMBuild.txt
add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(AsmParser)
# @{MYRISCVX_CMakeLists.txt
