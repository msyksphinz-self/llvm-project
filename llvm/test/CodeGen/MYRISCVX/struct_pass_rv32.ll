; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDANY %s

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

%struct.S128 = type { %struct.S64, i64 }
%struct.S64 = type { %struct.S32, i32 }
%struct.S32 = type { i8, i8, i16 }
%struct.S256 = type { %struct.S128, [4 x i32] }

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @func_S32(i32 %elem.coerce) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDANY-LABEL:func_S32:
; MYRVX32I_STATIC_MEDANY:# %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	andi	x11, x10, 255
; MYRVX32I_STATIC_MEDANY-NEXT:	srli	x12, x10, 16
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x12, x11
; MYRVX32I_STATIC_MEDANY-NEXT:	srli	x10, x10, 8
; MYRVX32I_STATIC_MEDANY-NEXT:	andi	x10, x10, 255
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x11, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x11, x0
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

entry:
  %elem.sroa.2.0.extract.shift = lshr i32 %elem.coerce, 8
  %elem.sroa.3.0.extract.shift = lshr i32 %elem.coerce, 16
  %conv = and i32 %elem.coerce, 255
  %conv1 = and i32 %elem.sroa.2.0.extract.shift, 255
  %add = add nuw nsw i32 %elem.sroa.3.0.extract.shift, %conv
  %add3 = add nuw nsw i32 %add, %conv1
  %conv4 = zext i32 %add3 to i64
  ret i64 %conv4
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @func_S64([2 x i32] %elem.coerce) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDANY-LABEL:func_S64:
; MYRVX32I_STATIC_MEDANY:# %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, -16
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x1, 12(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x2, 8(x2)               # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x9, 4(x2)               # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x9, x11
; MYRVX32I_STATIC_MEDANY-NEXT:	call	func_S32
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x12, x9
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x15, x10, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x12, x0, 1
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x13, x0
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x14, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x15, x13, $BB1_2
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.1:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x14, x13
; MYRVX32I_STATIC_MEDANY-NEXT:$BB1_2:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x15, x10, x9
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x15, x13, $BB1_4
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.3:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB1_4:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x9, 4(x2)               # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x2, 8(x2)               # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x1, 12(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, 16
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

entry:
  %elem.coerce.fca.0.extract = extractvalue [2 x i32] %elem.coerce, 0
  %elem.coerce.fca.1.extract = extractvalue [2 x i32] %elem.coerce, 1
  %call = call i64 @func_S32(i32 %elem.coerce.fca.0.extract)
  %conv = zext i32 %elem.coerce.fca.1.extract to i64
  %add = add i64 %call, %conv
  ret i64 %add
}

; Function Attrs: norecurse nounwind readonly
define dso_local i64 @func_S128(%struct.S128* nocapture readonly %elem) local_unnamed_addr #1 {

; MYRVX32I_STATIC_MEDANY-LABEL:func_S128:
; MYRVX32I_STATIC_MEDANY:# %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, -16
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x1, 12(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x2, 8(x2)               # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x9, 4(x2)               # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x9, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x11, 4(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x10, 0(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	call	func_S64
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x13, 8(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x13, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x16, x10, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x13, x0, 1
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x14, x0
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x15, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x16, x14, $BB2_2
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.1:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x15, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB2_2:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x12, x10, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x12, x14, $BB2_4
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.3:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x13, x15
; MYRVX32I_STATIC_MEDANY-NEXT:$BB2_4:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x12, 12(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x12, x11
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x9, 4(x2)               # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x2, 8(x2)               # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x1, 12(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, 16
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

entry:
  %.elt = bitcast %struct.S128* %elem to i32*
  %.unpack = load i32, i32* %.elt, align 8
  %0 = insertvalue [2 x i32] undef, i32 %.unpack, 0
  %1 = getelementptr inbounds %struct.S128, %struct.S128* %elem, i32 0, i32 0, i32 1
  %.unpack2 = load i32, i32* %1, align 4
  %2 = insertvalue [2 x i32] %0, i32 %.unpack2, 1
  %call = call i64 @func_S64([2 x i32] %2)
  %e = getelementptr inbounds %struct.S128, %struct.S128* %elem, i32 0, i32 1
  %3 = load i64, i64* %e, align 8, !tbaa !3
  %add = add i64 %3, %call
  ret i64 %add
}

; Function Attrs: nounwind
define dso_local i64 @func_S256(%struct.S256* nocapture readonly %elem) local_unnamed_addr #2 {

; MYRVX32I_STATIC_MEDANY-LABEL:func_S256:
; MYRVX32I_STATIC_MEDANY:# %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, -32
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x1, 28(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x2, 24(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x9, 20(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x9, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x10, x2
; MYRVX32I_STATIC_MEDANY-NEXT:	ori	x11, x10, 4
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x12, 4(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x12, 0(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x11, 12(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 12(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x11, 8(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 8(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x11, 0(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 0(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	call	func_S128
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x15, 28(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x6, 24(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x5, 20(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x16, 16(x9)
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x13, x0, 1
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x14, x0
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x17, x12, x16
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x7, x17, x5
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x28, x7, x6
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x28, x15
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x30, x10, x28
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x29, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x30, x14, $BB3_2
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.1:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x29, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_2:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x30, x10, x15
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x15, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x30, x14, $BB3_4
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.3:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x15, x29
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_4:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x29, x28, x6
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x6, x28, x7
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x28, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x6, x14, $BB3_6
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.5:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x28, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_6:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x6, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x29, x14, $BB3_8
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.7:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x6, x28
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_8:                                 # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x28, x7, x5
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x5, x7, x17
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x7, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x5, x14, $BB3_10
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.9:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x7, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_10:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x5, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x28, x14, $BB3_12
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.11:                               # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x5, x7
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_12:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x16, x17, x16
; MYRVX32I_STATIC_MEDANY-NEXT:	sltu	x17, x17, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x17, x14, $BB3_14
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.13:                               # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x12, x14
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_14:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	bne	x16, x14, $BB3_16
; MYRVX32I_STATIC_MEDANY-NEXT:# %bb.15:                               # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x13, x12
; MYRVX32I_STATIC_MEDANY-NEXT:$BB3_16:                                # %entry
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x13
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x5
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x6
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x11, x15
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x9, 20(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x2, 24(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x1, 28(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, 32
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

entry:
  %byval-temp = alloca %struct.S128, align 8
  %0 = getelementptr inbounds %struct.S128, %struct.S128* %byval-temp, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %0) #4
  %1 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i32 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* nonnull align 8 dereferenceable(16) %0, i8* nonnull align 8 dereferenceable(16) %1, i32 16, i1 false), !tbaa.struct !12
  %call = call i64 @func_S128(%struct.S128* nonnull %byval-temp)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %0) #4
  %arrayidx = getelementptr inbounds %struct.S256, %struct.S256* %elem, i32 0, i32 1, i32 0
  %2 = load i32, i32* %arrayidx, align 8, !tbaa !15
  %conv = zext i32 %2 to i64
  %add = add i64 %call, %conv
  %arrayidx2 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i32 0, i32 1, i32 1
  %3 = load i32, i32* %arrayidx2, align 4, !tbaa !15
  %conv3 = zext i32 %3 to i64
  %add4 = add i64 %add, %conv3
  %arrayidx6 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i32 0, i32 1, i32 2
  %4 = load i32, i32* %arrayidx6, align 8, !tbaa !15
  %conv7 = zext i32 %4 to i64
  %add8 = add i64 %add4, %conv7
  %arrayidx10 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i32 0, i32 1, i32 3
  %5 = load i32, i32* %arrayidx10, align 4, !tbaa !15
  %conv11 = zext i32 %5 to i64
  %add12 = add i64 %add8, %conv11
  ret i64 %add12
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg) #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: nounwind
define dso_local i32 @func_S256_caller() local_unnamed_addr #2 {

; MYRVX32I_STATIC_MEDANY-:func_S256_caller:
; MYRVX32I_STATIC_MEDANY:# %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, -48
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x1, 44(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x2, 40(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x10, x2, 8
; MYRVX32I_STATIC_MEDANY-NEXT:	ori	x11, x10, 4
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x12, x0, 400
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x12, 0(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:	mv	x11, x0
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 20(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x0, 900
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 36(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x0, 800
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 32(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x0, 700
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 28(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x0, 600
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 24(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	lui	x11, 912092
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x11, -273
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 16(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	lui	x11, 4813
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x11, x11, -1948
; MYRVX32I_STATIC_MEDANY-NEXT:	sw	x11, 8(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:	call	func_S256
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x2, 40(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x1, 44(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDANY-NEXT:	addi	x2, x2, 48
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

entry:
  %byval-temp = alloca %struct.S256, align 8
  %0 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #4
  store i8 100, i8* %0, align 8, !tbaa.struct !17
  %elem.sroa.4.0..sroa_idx = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 0, i32 0, i32 0, i32 1
  store i8 -56, i8* %elem.sroa.4.0..sroa_idx, align 1, !tbaa.struct !17
  %elem.sroa.5.0..sroa_idx16 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 0, i32 0, i32 0, i32 2
  store i16 300, i16* %elem.sroa.5.0..sroa_idx16, align 2, !tbaa.struct !17
  %elem.sroa.6.0..sroa_idx18 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 0, i32 0, i32 1
  store i32 400, i32* %elem.sroa.6.0..sroa_idx18, align 4, !tbaa.struct !17
  %elem.sroa.7.0..sroa_idx20 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 0, i32 1
  store i64 3735928559, i64* %elem.sroa.7.0..sroa_idx20, align 8, !tbaa.struct !17
  %elem.sroa.8.0..sroa_idx22 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 1, i32 0
  store i32 600, i32* %elem.sroa.8.0..sroa_idx22, align 8, !tbaa.struct !17
  %elem.sroa.9.0..sroa_idx24 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 1, i32 1
  store i32 700, i32* %elem.sroa.9.0..sroa_idx24, align 4, !tbaa.struct !17
  %elem.sroa.10.0..sroa_idx26 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 1, i32 2
  store i32 800, i32* %elem.sroa.10.0..sroa_idx26, align 8, !tbaa.struct !17
  %elem.sroa.11.0..sroa_idx28 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i32 0, i32 1, i32 3
  store i32 900, i32* %elem.sroa.11.0..sroa_idx28, align 4, !tbaa.struct !17
  %call = call i64 @func_S256(%struct.S256* nonnull %byval-temp)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #4
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 16c6cdf270cec0ae30ef18641075a36f6f17b0b1)"}
!3 = !{!4, !11, i64 8}
!4 = !{!"S128", !5, i64 0, !11, i64 8}
!5 = !{!"S64", !6, i64 0, !10, i64 4}
!6 = !{!"S32", !7, i64 0, !7, i64 1, !9, i64 2}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!"short", !7, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!"long long", !7, i64 0}
!12 = !{i64 0, i64 1, !13, i64 1, i64 1, !13, i64 2, i64 2, !14, i64 4, i64 4, !15, i64 8, i64 8, !16}
!13 = !{!7, !7, i64 0}
!14 = !{!9, !9, i64 0}
!15 = !{!10, !10, i64 0}
!16 = !{!11, !11, i64 0}
!17 = !{i64 0, i64 1, !13, i64 1, i64 1, !13, i64 2, i64 2, !14, i64 4, i64 4, !15, i64 8, i64 8, !16, i64 16, i64 16, !13}
