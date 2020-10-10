//===-- MYRISCVXInstrInfo.cpp - MYRISCVX Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MYRISCVX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MYRISCVXInstrInfo.h"

#include "MYRISCVXMachineFunction.h"
#include "MYRISCVXTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "MYRISCVXGenInstrInfo.inc"

// Pin the vtable to this file.
void MYRISCVXInstrInfo::anchor() {}

MYRISCVXInstrInfo::MYRISCVXInstrInfo()
    : MYRISCVXGenInstrInfo(MYRISCVX::ADJCALLSTACKDOWN, MYRISCVX::ADJCALLSTACKUP)
{}

/// Return the number of bytes of code the specified instruction may be.
unsigned MYRISCVXInstrInfo::GetInstSizeInBytes(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default:
    return MI.getDesc().getSize();
  }
}

// @{ MYRISCVXInstrInfo_GetMemOperand
MachineMemOperand *
MYRISCVXInstrInfo::GetMemOperand(MachineBasicBlock &MBB, int FI,
                                 MachineMemOperand::Flags Flags) const {

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  Align align = MFI.getObjectAlign(FI);

  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI),
                                 Flags, MFI.getObjectSize(FI), align);
}
// @} MYRISCVXInstrInfo_GetMemOperand


// @{ MYRISCVXInstrInfo_storeRegToStack
void MYRISCVXInstrInfo::
storeRegToStack(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                Register SrcReg, bool isKill, int FI,
                const TargetRegisterClass *RC, const TargetRegisterInfo *TRI,
                int64_t Offset) const {
  DebugLoc DL;
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOStore);

  unsigned Opc;
  if (MYRISCVX::GPRRegClass.hasSubClassEq(RC))
    Opc = TRI->getRegSizeInBits(MYRISCVX::GPRRegClass) == 32 ?
        MYRISCVX::SW : MYRISCVX::SD;
  else if (MYRISCVX::FPR_SRegClass.hasSubClassEq(RC))
    Opc = MYRISCVX::FSW;
  else if (MYRISCVX::FPR_DRegClass.hasSubClassEq(RC))
    Opc = MYRISCVX::FSD;
  else
    llvm_unreachable("Can't store this register to stack slot");

  BuildMI(MBB, I, DL, get(Opc)).addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI).addImm(Offset).addMemOperand(MMO);
}
// @} MYRISCVXInstrInfo_storeRegToStack


// @{ MYRISCVXInstrInfo_loadRegFromStack
void MYRISCVXInstrInfo::
loadRegFromStack(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                 Register DestReg, int FI, const TargetRegisterClass *RC,
                 const TargetRegisterInfo *TRI, int64_t Offset) const {
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  MachineMemOperand *MMO = GetMemOperand(MBB, FI, MachineMemOperand::MOLoad);

  unsigned Opc;
  if (MYRISCVX::GPRRegClass.hasSubClassEq(RC))
    Opc = TRI->getRegSizeInBits(MYRISCVX::GPRRegClass) == 32 ?
        MYRISCVX::LW : MYRISCVX::LD;
  else if (MYRISCVX::FPR_SRegClass.hasSubClassEq(RC))
    Opc = MYRISCVX::FLW;
  else if (MYRISCVX::FPR_DRegClass.hasSubClassEq(RC))
    Opc = MYRISCVX::FLD;
  else
    llvm_unreachable("Can't store this register to stack slot");

  BuildMI(MBB, I, DL, get(Opc), DestReg).addFrameIndex(FI).addImm(Offset)
      .addMemOperand(MMO);
}
// @} MYRISCVXInstrInfo_loadRegFromStack


// @{ MYRISCVXInstrInfo_copyPhysReg
// copyPhysRegsはレジスタ間コピーを行うためのノードを生成する
// MYRISCVXの場合はADDI rd,rs,0で所望のコピーが行える
void MYRISCVXInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    const DebugLoc &DL, MCRegister DstReg,
                                    MCRegister SrcReg, bool KillSrc) const {
  if (MYRISCVX::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(MYRISCVX::ADDI), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
    return;
  }
  llvm_unreachable("Doesn't support Floating Point");
}
// @} MYRISCVXInstrInfo_copyPhysReg

//@{ MYRISCVXInstrInfo_expandPostRA
/// 疑似命令を本物の命令に変換するための関数
bool MYRISCVXInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();

  switch (MI.getDesc().getOpcode()) {
  default:
    return false;
  case MYRISCVX::RetRA: // MYRISCVX::RetRAノードの場合はexpandRetRA()に飛ぶ
    expandRetRA(MBB, MI);
    break;
  }

  MBB.erase(MI);
  return true;
}

// @{ MYRISCVXInstrInfo_adjustStackPtr
/// SPをスタックフレームのサイズだけ移動する
void MYRISCVXInstrInfo::adjustStackPtr(unsigned SP, int64_t Amount,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const {
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();
  // オフセットが12ビット以内に収まらない場合はADD命令を使用する
  unsigned ADD = MYRISCVX::ADD;
  // オフセットが12ビット以内に収まる場合はADDI命令を使用する
  unsigned ADDI = MYRISCVX::ADDI;

  if (isInt<12>(Amount)) {
    // スタックフレームのサイズが12ビット以内で収まるならば, 単純にaddi sp, sp,
    // amountを実行する
    BuildMI(MBB, I, DL, get(ADDI), SP).addReg(SP).addImm(Amount);
  } else {
    // スタックフレームが12ビットのサイズに収まらない場合, loadImmediate()による定数生成を行う
    MachineFunction *MF = MBB.getParent();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    // 仮想的なレジスタを作成して定数の中間値を格納
    unsigned Reg = MRI.createVirtualRegister(&MYRISCVX::GPRRegClass);
    // loaImmediate()を呼び出して12ビットよりも大きい定数を生成
    loadImmediate(Amount, MBB, I, DL, Reg, nullptr);
    // 生成した定数通りにスタックポインタを更新する
    BuildMI(MBB, I, DL, get(ADD), SP).addReg(SP).addReg(Reg, RegState::Kill);
  }
}
// @} MYRISCVXInstrInfo_adjustStackPtr

// @{ MYRISCVXInstrInfo_loadImmediate
// 即値の生成に必要な命令の生成
void MYRISCVXInstrInfo::loadImmediate(int64_t Imm, MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator II,
                                      const DebugLoc &DL, unsigned DstReg,
                                      unsigned *NewImm) const {
  // 定数を上位の20ビットと下位の12ビットに分けて命令を生成する
  uint64_t Hi20 = ((Imm + 0x800) >> 12) & 0xfffff;
  uint64_t Lo12 = SignExtend64<12>(Imm);
  // 上位の20ビットはLUI命令を使用して生成
  BuildMI(MBB, II, DL, get(MYRISCVX::LUI), DstReg).addImm(Hi20);
  // 下位の12ビットはADDI命令を使用して生成
  BuildMI(MBB, II, DL, get(MYRISCVX::ADDI), DstReg)
      .addReg(DstReg, RegState::Kill)
      .addImm(Lo12);
}
// @} MYRISCVXInstrInfo_loadImmediate

void MYRISCVXInstrInfo::expandRetRA(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I) const {
  // expandRetRAでは, MYRISCVX::RetRAノードをret命令(=JALR x0,ra,0)に変換する
  BuildMI(MBB, I, I->getDebugLoc(), get(MYRISCVX::JALR))
      .addReg(MYRISCVX::ZERO)
      .addReg(MYRISCVX::RA)
      .addImm(0);
}
//@} MYRISCVXInstrInfo_expandPostRA
