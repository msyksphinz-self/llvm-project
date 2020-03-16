//===-- MYRISCVXISelLowering.cpp - MYRISCVX DAG Lowering Implementation ---===//
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
#include "MYRISCVXISelLowering.h"

#include "MYRISCVXMachineFunction.h"
#include "MYRISCVXTargetMachine.h"
#include "MYRISCVXTargetObjectFile.h"
#include "MYRISCVXSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "MYRISCVX-lower"

//@3_1 1 {
const char *MYRISCVXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    case MYRISCVXISD::JmpLink:           return "MYRISCVXISD::JmpLink";
    case MYRISCVXISD::TailCall:          return "MYRISCVXISD::TailCall";
    case MYRISCVXISD::Hi:                return "MYRISCVXISD::Hi";
    case MYRISCVXISD::Lo:                return "MYRISCVXISD::Lo";
    case MYRISCVXISD::GPRel:             return "MYRISCVXISD::GPRel";
    case MYRISCVXISD::Ret:               return "MYRISCVXISD::Ret";
    case MYRISCVXISD::EH_RETURN:         return "MYRISCVXISD::EH_RETURN";
    case MYRISCVXISD::DivRem:            return "MYRISCVXISD::DivRem";
    case MYRISCVXISD::DivRemU:           return "MYRISCVXISD::DivRemU";
    case MYRISCVXISD::Wrapper:           return "MYRISCVXISD::Wrapper";
    default:                             return NULL;
  }
}
//@3_1 1 }

//@MYRISCVXTargetLowering {
MYRISCVXTargetLowering::MYRISCVXTargetLowering(const MYRISCVXTargetMachine &TM,
                                               const MYRISCVXSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI), ABI(TM.getABI()) {
  MVT XLenVT = Subtarget.getXLenVT();

  // Set up the register classes.
  addRegisterClass(XLenVT, &MYRISCVX::GPRRegClass);

  //- Set .align 2
  // It will emit .align 2 later
  setMinFunctionAlignment(2);
  // must, computeRegisterProperties - Once all of the register classes are
  //  added, this allows us to compute derived properties we expose.
  computeRegisterProperties(STI.getRegisterInfo());

  //@{MYRISCVXTargetLowering_setOperationAction_DontGenerate
  setOperationAction(ISD::ROTL, XLenVT, Expand);
  setOperationAction(ISD::ROTR, XLenVT, Expand);
  //@}MYRISCVXTargetLowering_setOperationAction_DontGenerate

  //@{MYRISCVXTargetLowering_setOperationAction_GlobalAddress
  setOperationAction(ISD::GlobalAddress, XLenVT, Custom);
  //@}MYRISCVXTargetLowering_setOperationAction_GlobalAddress

  // @{MYRISCVXTargetLowering_setOperationAction_Branch_Select
  // Branch Instructions
  setOperationAction(ISD::BR_CC,     XLenVT,     Expand);
  setOperationAction(ISD::BR_JT,     MVT::Other, Expand);

  setOperationAction(ISD::SELECT,    XLenVT,     Custom);
  setOperationAction(ISD::SELECT_CC, XLenVT,     Expand);
  // @}MYRISCVXTargetLowering_setOperationAction_Branch_Select
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

// addLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned
addLiveIn(MachineFunction &MF, unsigned PReg, const TargetRegisterClass *RC)
{
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

#include "MYRISCVXGenCallingConv.inc"

//@{MYRISCVXTargetLowering_LowerOperation
SDValue MYRISCVXTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const
{
  switch (Op.getOpcode())
  {
    case ISD::GlobalAddress: return lowerGlobalAddress(Op, DAG);
  }
  return SDValue();
}
//@}MYRISCVXTargetLowering_LowerOperation


//@{MYRISCVXTargetLowering_lowerGlobalAddress
SDValue MYRISCVXTargetLowering::lowerGlobalAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  EVT Ty = Op.getValueType();
  GlobalAddressSDNode *N = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = N->getGlobal();

  //@{MYRISCVXTargetLowering_lowerGlobalAddress_nonPIC
  if (!isPositionIndependent()) {
    //@ %hi/%lo relocation
    return getAddrNonPIC(N, Ty, DAG);
  }
  //@}MYRISCVXTargetLowering_lowerGlobalAddress_nonPIC

  if (GV->hasInternalLinkage() || (GV->hasLocalLinkage() && !isa<Function>(GV)))
    return getAddrLocal(N, Ty, DAG);

  //@{MYRISCVXTargetLowering_lowerGlobalAddress_PIC
  return getAddrGlobalGOT(
      N, Ty, DAG, MYRISCVXII::MO_GOT_HI20, MYRISCVXII::MO_GOT_LO12,
      DAG.getEntryNode(),
      MachinePointerInfo::getGOT(DAG.getMachineFunction()));
  //@}MYRISCVXTargetLowering_lowerGlobalAddress_PIC
}
//@}MYRISCVXTargetLowering_lowerGlobalAddress


SDValue MYRISCVXTargetLowering::getGlobalReg(SelectionDAG &DAG, EVT Ty) const {
  MYRISCVXFunctionInfo *FI = DAG.getMachineFunction().getInfo<MYRISCVXFunctionInfo>();
  return DAG.getRegister(FI->getGlobalBaseReg(), Ty);
}

//@getTargetNode(GlobalAddressSDNode
SDValue MYRISCVXTargetLowering::getTargetNode(GlobalAddressSDNode *N, EVT Ty,
                                              SelectionDAG &DAG,
                                              unsigned Flag) const {
  return DAG.getTargetGlobalAddress(N->getGlobal(), SDLoc(N), Ty, 0, Flag);
}

//@getTargetNode(ExternalSymbolSDNode
SDValue MYRISCVXTargetLowering::getTargetNode(ExternalSymbolSDNode *N, EVT Ty,
                                              SelectionDAG &DAG,
                                              unsigned Flag) const {
  return DAG.getTargetExternalSymbol(N->getSymbol(), Ty, Flag);
}


// @{MYRISCVXISelLowering_LowerFormalArguments
// @{MYRISCVXISelLowering_LowerFormalArguments_Head
//===----------------------------------------------------------------------===//
//@            Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//
/// LowerFormalArguments - transform physical registers into virtual registers
/// and generate load operations for arguments places on the stack.
SDValue
MYRISCVXTargetLowering::LowerFormalArguments(SDValue Chain,
                                             CallingConv::ID CallConv,
                                             bool IsVarArg,
                                             const SmallVectorImpl<ISD::InputArg> &Ins,
                                             const SDLoc &DL, SelectionDAG &DAG,
                                             SmallVectorImpl<SDValue> &InVals)
// @}MYRISCVXISelLowering_LowerFormalArguments_Head
const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MYRISCVXFunctionInfo *MYRISCVXFI = MF.getInfo<MYRISCVXFunctionInfo>();

  MYRISCVXFI->setVarArgsFrameIndex(0);

  // @{MYRISCVXISelLowering_LowerFormalArguments_AnalyzeFormalarguments
  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments (Ins, CC_MYRISCVX);
  // @}MYRISCVXISelLowering_LowerFormalArguments_AnalyzeFormalarguments

  Function::const_arg_iterator FuncArg =
      DAG.getMachineFunction().getFunction().arg_begin();

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  unsigned CurArgIdx = 0;
  CCInfo.rewindByValRegsInfo();

  // @{MYRISCVXISelLowering_LowerFormalArguments_Loop
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (Ins[i].isOrigArg()) {
      std::advance(FuncArg, Ins[i].getOrigArgIndex() - CurArgIdx);
      CurArgIdx = Ins[i].getOrigArgIndex();
    }
    // @}MYRISCVXISelLowering_LowerFormalArguments_Loop
    EVT ValVT = VA.getValVT();

    // @{MYRISCVXISelLowering_LowerFormalArguments_RegLoc
    bool IsRegLoc = VA.isRegLoc();

    // Arguments stored on registers
    if (IsRegLoc) {
      MVT RegVT = VA.getLocVT();
      unsigned ArgReg = VA.getLocReg();
      const TargetRegisterClass *RC = getRegClassFor(RegVT);

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Reg = addLiveIn(DAG.getMachineFunction(), ArgReg, RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue,
                                 DAG.getValueType(ValVT));
        ArgValue = DAG.getNode(ISD::TRUNCATE, DL, ValVT, ArgValue);
      }
      InVals.push_back(ArgValue);
      // @}MYRISCVXISelLowering_LowerFormalArguments_RegLoc
      // @{MYRISCVXISelLowering_LowerFormalArguments_MemLoc
    } else { // VA.isRegLoc()
      MVT LocVT = VA.getLocVT();

      // sanity check
      assert(VA.isMemLoc());

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI.CreateFixedObject(ValVT.getSizeInBits()/8,
                                     VA.getLocMemOffset(), true);

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Load = DAG.getLoad(
          LocVT, DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
      InVals.push_back(Load);
      OutChains.push_back(Load.getValue(1));
    }
    // @}MYRISCVXISelLowering_LowerFormalArguments_MemLoc
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}
// @}MYRISCVXISelLowering_LowerFormalArguments


// @{MYRISCVXISelLowering_LowerReturn
//===----------------------------------------------------------------------===//
//@              Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//
// @{MYRISCVXISelLowering_LowerReturn_Header
SDValue
MYRISCVXTargetLowering::LowerReturn(SDValue Chain,
                                    CallingConv::ID CallConv, bool IsVarArg,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SDLoc &DL, SelectionDAG &DAG) const {
  // @}MYRISCVXISelLowering_LowerReturn_Header
  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  MachineFunction &MF = DAG.getMachineFunction();


  // @{MYRISCVXISelLowering_LowerReturn_AnalyzeReturn
  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_MYRISCVX);
  // @}MYRISCVXISelLowering_LowerReturn_AnalyzeReturn

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    SDValue Val = OutVals[i];
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // @{MYRISCVXISelLowering_LowerReturn_BITCAST
    if (RVLocs[i].getValVT() != RVLocs[i].getLocVT())
      Val = DAG.getNode(ISD::BITCAST, DL, RVLocs[i].getLocVT(), Val);
    // @}MYRISCVXISelLowering_LowerReturn_BITCAST

    // @{MYRISCVXISelLowering_LowerReturn_getCopyToReg
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Val, Flag);
    // @}MYRISCVXISelLowering_LowerReturn_getCopyToReg

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  // @{MYRISCVXISelLowering_LowerReturn_RET
  return DAG.getNode(MYRISCVXISD::Ret, DL, MVT::Other, RetOps);
  // @}MYRISCVXISelLowering_LowerReturn_RET
}
// @}MYRISCVXISelLowering_LowerReturn
