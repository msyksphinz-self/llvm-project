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

  //@class MYRISCVXTargetLowering
  class MYRISCVXTargetLowering : public TargetLowering  {
 public:
    explicit MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                    const MYRISCVXSubtarget &STI);

    static const MYRISCVXTargetLowering *create(const MYRISCVXTargetMachine &TM,
                                                const MYRISCVXSubtarget &STI);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    /// getTargetNodeName - This method returns the name of a target specific
    //  DAG node.
    const char *getTargetNodeName(unsigned Opcode) const override;

    // @{ MYRISCVXTargetLowering_getAddrGlobalGOT
    template<class NodeTy>
    SDValue getAddrGlobalGOT(NodeTy *N, EVT Ty, SelectionDAG &DAG,
                             unsigned HiFlag, unsigned LoFlag,
                             SDValue Chain,
                             const MachinePointerInfo &PtrInfo) const {
      SDLoc DL(N);
      SDValue AddrHi = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_GOT_HI20);
      SDValue AddrLo = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_PCREL_LO12_I);
      SDValue MNHi = SDValue(DAG.getMachineNode(MYRISCVX::AUIPC, DL, Ty, AddrHi), 0);
      return SDValue(DAG.getMachineNode(MYRISCVX::ADDI, DL, Ty, MNHi, AddrLo), 0);
    }
    // @} MYRISCVXTargetLowering_getAddrGlobalGOT

    // @{ MYRISCVXTargetLowering_getAddrStatic
    template<class NodeTy>
    SDValue getAddrStatic(NodeTy *N, EVT Ty, SelectionDAG &DAG) const {
      SDLoc DL(N);
      switch (getTargetMachine().getCodeModel()) {
        default:
          report_fatal_error("Unsupported code model for lowering");
        case CodeModel::Small: {
          // MedLow C-Model
          SDValue AddrHi = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_HI20);
          SDValue AddrLo = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_LO12_I);
          SDValue MNHi = SDValue(DAG.getMachineNode(MYRISCVX::LUI, DL, Ty, AddrHi), 0);
          return SDValue(DAG.getMachineNode(MYRISCVX::ADDI, DL, Ty, MNHi, AddrLo), 0);
        }
        case CodeModel::Medium: {
          // MedLow C-Model
          SDValue AddrHi = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_PCREL_HI20);
          SDValue AddrLo = getTargetNode(N, Ty, DAG, MYRISCVXII::MO_LO12_I);
          SDValue MNHi = SDValue(DAG.getMachineNode(MYRISCVX::AUIPC, DL, Ty, AddrHi), 0);
          return SDValue(DAG.getMachineNode(MYRISCVX::ADDI, DL, Ty, MNHi, AddrLo), 0);
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

    static unsigned getBranchOpcodeForIntCondCode (ISD::CondCode CC);

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr &MI,
                                MachineBasicBlock *BB) const override;

    static MachineBasicBlock *emitSelectPseudo(MachineInstr &MI,
                                               MachineBasicBlock *BB);
  };
}

#endif // MYRISCVXISELLOWERING_H
