//===-- MYRISCVXISelLowering.h - MYRISCVX DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MYRISCVX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MYRISCVX_MYRISCVXISELLOWERING_H
#define LLVM_LIB_TARGET_MYRISCVX_MYRISCVXISELLOWERING_H

#include "MCTargetDesc/MYRISCVXABIInfo.h"
#include "MCTargetDesc/MYRISCVXBaseInfo.h"
#include "MYRISCVX.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/TargetLowering.h"
#include <deque>

namespace llvm {
  namespace MYRISCVXISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // @{ MYRISCVXISelLowering_h_CALL
      // Jump and link (call)
      CALL,
      // @} MYRISCVXISelLowering_h_CALL

      // Tail call
      TailCall,

      // Get the Higher 16 bits from a 32-bit immediate
      // No relation with MYRISCVX Hi register
      Hi,
      // Get the Lower 16 bits from a 32-bit immediate
      // No relation with MYRISCVX Lo register
      Lo,

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Thread Pointer
      ThreadPointer,

      // Return
      Ret,

      SELECT_CC,

      EH_RETURN,

      // DivRem(u)
      DivRem,
      DivRemU,

      Wrapper,
      DynAlloc,

      Sync
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class MYRISCVXFunctionInfo;
  class MYRISCVXSubtarget;


  // @{ MYRISCVXTargeTLowering_Class
  // TargetLoweringはLLVM IRをターゲット向けの命令に変換するために必要な
  // 情報と機能を定義する
  class MYRISCVXTargetLowering : public TargetLowering  {
 public:
    explicit MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                    const MYRISCVXSubtarget &STI);
  // @} MYRISCVXTargeTLowering_Class

    static const MYRISCVXTargetLowering *create(const MYRISCVXTargetMachine &TM,
                                                const MYRISCVXSubtarget &STI);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    /// getTargetNodeName - This method returns the name of a target specific
    //  DAG node.
    const char *getTargetNodeName(unsigned Opcode) const override;

    // @{ MYRISCVXTargetLowering_getAddrStatic
    template<class NodeTy>
    SDValue getAddrStatic(NodeTy *N, EVT Ty, SelectionDAG &DAG) const {
      SDLoc DL(N);
      switch (getTargetMachine().getCodeModel()) {
        default:
          report_fatal_error("Unsupported code model for lowering");
        case CodeModel::Small: {
          // MedLow コードモデルの場合：
          // LUI + ADDI命令のシーケンスを出力する
          SDValue AddrHi = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_HI20);
          SDValue AddrLo = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_LO12_I);
          SDValue MNHi = SDValue(DAG.getMachineNode(MYRISCVX::LUI, DL, Ty, AddrHi), 0);
          return SDValue(DAG.getMachineNode(MYRISCVX::ADDI, DL, Ty, MNHi, AddrLo), 0);
        }
        case CodeModel::Medium: {
          // Medany コードモデルの場合：
          // LLA疑似命令をそのまま出力する
          SDValue Addr = getTargetNode(N, Ty, DAG, 0);
          return SDValue(DAG.getMachineNode(MYRISCVX::PseudoLLA, DL, Ty, Addr), 0);
        }
      }
    }
    // @} MYRISCVXTargetLowering_getAddrStatic

   protected:

    /// ByValArgInfo - Byval argument information.
    struct ByValArgInfo {
      unsigned FirstIdx; // Index of the first register used.
      unsigned NumRegs;  // Number of registers used for this argument.
      unsigned Address;  // Offset of the stnack area used to pass this argument.

      ByValArgInfo() : FirstIdx(0), NumRegs(0), Address(0) {}
    };

 protected:
    // Subtarget Info
    const MYRISCVXSubtarget &Subtarget;
    // Cache the ABI from the TargetMachine, we use it everywhere.
    const MYRISCVXABIInfo &ABI;

 private:

    // Create a TargetGlobalAddress node.
    SDValue getTargetNode(GlobalAddressSDNode *N, EVT Ty, SelectionDAG &DAG,
                          unsigned Flag) const;

    // Create a TargetBlockAddress node.
    SDValue getTargetNode(BlockAddressSDNode *N, EVT Ty, SelectionDAG &DAG,
                          unsigned Flag) const;

    // Create a TargetJumpTable node.
    SDValue getTargetNode(JumpTableSDNode *N, EVT Ty, SelectionDAG &DAG,
                          unsigned Flag) const;

    // Create a TargetExternalSymbol node.
    SDValue getTargetNode(ExternalSymbolSDNode *N, EVT Ty, SelectionDAG &DAG,
                          unsigned Flag) const;

    // Lower Operand specifics
    SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
    SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) const;


    SDValue
    LowerCall(TargetLowering::CallLoweringInfo &CLI,
              SmallVectorImpl<SDValue> &InVals) const override;
    SDValue
    LowerCallResult(SDValue Chain, SDValue InFlag,
                    CallingConv::ID CallConv, bool IsVarArg,
                    const SmallVectorImpl<ISD::InputArg> &Ins,
                    const SDLoc &DL, SelectionDAG &DAG,
                    SmallVectorImpl<SDValue> &InVals,
                    const SDNode *CallNode,
                    const Type *RetTy) const;


    SDValue
    passArgOnStack(SDValue StackPtr, unsigned Offset,
                   SDValue Chain, SDValue Arg, const SDLoc &DL,
                   bool IsTailCall, SelectionDAG &DAG) const;

    /// This function fills Ops, which is the list of operands that will later
    /// be used when a function call node is created. It also generates
    /// copyToReg nodes to set up argument registers.
    virtual void
    getOpndList(SmallVectorImpl<SDValue> &Ops,
                std::deque< std::pair<unsigned, SDValue> > &RegsToPass,
                bool IsPICCall, bool GlobalOrExternal, bool InternalLinkage,
                CallLoweringInfo &CLI, SDValue Callee, SDValue Chain) const;

    //- must be exist even without function all
    SDValue
    LowerFormalArguments(SDValue Chain,
                         CallingConv::ID CallConv, bool IsVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         const SDLoc &dl, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const override;

    SDValue LowerReturn(SDValue Chain,
                        CallingConv::ID CallConv, bool IsVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        const SDLoc &dl, SelectionDAG &DAG) const override;

    SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;

    /// writeVarArgRegs - Write variable function arguments passed in registers
    /// to the stack. Also create a stack frame object for the first variable
    /// argument.
    void writeVarArgRegs(std::vector<SDValue> &OutChains, SDValue Chain,
                         const SDLoc &DL, SelectionDAG &DAG,
                         CCState &State) const;

    static unsigned getBranchOpcodeForIntCondCode (ISD::CondCode CC);

    unsigned getInlineAsmMemConstraint(StringRef ConstraintCode) const override;

    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 StringRef Constraint, MVT VT) const override;

    void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const override;

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *BB) const override;

    static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                               MachineBasicBlock *BB);
    bool isEligibleForTailCallOptimization(
        CCState &CCInfo,
        unsigned NextStackOffset, const MYRISCVXFunctionInfo& FI) const;
  };
}

#endif // MYRISCVXISELLOWERING_H
