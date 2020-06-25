//===-- MYRISCVXFrameLowering.cpp - MYRISCVX Frame Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXFrameLowering.h"

#include "MYRISCVXInstrInfo.h"
#include "MYRISCVXMachineFunction.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-frame"

// @{ MYRISCVXFrameLowering_emitPrologue_Impl
// @{ MYRISCVXFrameLowering_emitPrologue
// @{ MYRISCVXFrameLowering_emitPrologue_Name
// @{ emitPrologue_ComputeStackSize
// 関数のプロローグを生成する：関数フレームの確保とCallee Savedレジスタの退避
void MYRISCVXFrameLowering::emitPrologue(MachineFunction &MF,
                                         MachineBasicBlock &MBB) const {
  // @} MYRISCVXFrameLowering_emitPrologue_Name
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo MFI = MF.getFrameInfo();

  // @{ emitPrologue_ComputeStackSize ...
  const MYRISCVXInstrInfo &TII =
      *static_cast<const MYRISCVXInstrInfo *>(STI.getInstrInfo());

  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned SP = MYRISCVX::SP;

  // @} emitPrologue_ComputeStackSize ...
  // まず最終的なスタックフレームのサイズを取得する
  uint64_t StackSize = MFI.getStackSize();

  // もしスタックフレームのサイズが0ならば, スタックフレームの調整は必要ない
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  // @{ emitPrologue_ComputeStackSize ...
  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();
  // @} emitPrologue_ComputeStackSize ...

  // スタックフレームの調整を行う. マイナス方向
  TII.adjustStackPtr(SP, -StackSize, MBB, MBBI);
  // @} emitPrologue_ComputeStackSize

  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, -StackSize));
  BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

  if (CSI.size()) {
    // もし退避しなければならないCallee Savedレジスタが存在していれば
    // スタックに退避するための命令を生成する
    for (unsigned i = 0; i < CSI.size(); ++i)
      ++MBBI;

    // 全ての退避しなければならないCallee Savedレジスタに対して
    // スタックに退避するための命令を生成する
    for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
                                                      E = CSI.end();
         I != E; ++I) {
      int64_t Offset = MFI.getObjectOffset(I->getFrameIdx());
      unsigned Reg = I->getReg();
      {
        // Reg is in CPURegs.
        unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
            nullptr, MRI->getDwarfRegNum(Reg, 1), Offset));
        BuildMI(MBB, MBBI, dl, TII.get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
    }
  }
}
// @} MYRISCVXFrameLowering_emitPrologue
// @} MYRISCVXFrameLowering_emitPrologue_Impl

// @{ MYRISCVXFrameLowering_emitEpilogue_Impl
// @{ MYRISCVXFrameLowering_emitEpilogue
// 関数のスタックフレームを開放する
void MYRISCVXFrameLowering::emitEpilogue(MachineFunction &MF,
                                         MachineBasicBlock &MBB) const {
  // @{ MYRISCVXFrameLowering_emitEpilogue_Impl ...
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo MFI = MF.getFrameInfo();

  const MYRISCVXInstrInfo &TII =
      *static_cast<const MYRISCVXInstrInfo *>(STI.getInstrInfo());

  DebugLoc dl = MBBI->getDebugLoc();
  // @} MYRISCVXFrameLowering_emitEpilogue_Impl ...

  // スタックポインタSPを使用する
  unsigned SP = MYRISCVX::SP;

  // スタックフレームのサイズを取得する
  uint64_t StackSize = MFI.getStackSize();

  if (!StackSize)
    return;

  // スタックポインタ(sp)の調整を行う. プラス方向
  TII.adjustStackPtr(SP, StackSize, MBB, MBBI);
}
// @} MYRISCVXFrameLowering_emitEpilogue
// @} MYRISCVXFrameLowering_emitEpilogue_Impl

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas,
// if it needs dynamic stack realignment, if frame pointer elimination is
// disabled, or if the frame address is taken.
bool MYRISCVXFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken() ||
         TRI->hasStackRealignment(MF);
}


// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions
// @{ MYRISCVXFrameLowering_eliminateCallFramePseudoInstr
MachineBasicBlock::iterator MYRISCVXFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  unsigned SP = MYRISCVX::SP;
  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();
    if (I->getOpcode() == MYRISCVX::ADJCALLSTACKDOWN)
      Amount = -Amount;

    STI.getInstrInfo()->adjustStackPtr(SP, Amount, MBB, I);
  }

  return MBB.erase(I);
}
// @} MYRISCVXFrameLowering_eliminateCallFramePseudoInstr
