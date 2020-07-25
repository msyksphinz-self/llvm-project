; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDANY %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.S256 = type { %struct.S128, [4 x i32] }
%struct.S128 = type { %struct.S64, i64 }
%struct.S64 = type { %struct.S32, i32 }
%struct.S32 = type { i8, i8, i16 }

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @func_S32(i64 %elem.coerce) local_unnamed_addr #0 {

; MYRVX64I_STATIC_MEDANY-LABEL:func_S32:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	andi    x11, x10, 255
; MYRVX64I_STATIC_MEDANY-NEXT:	slli    x10, x10, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	srli    x10, x10, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	srli    x12, x10, 16
; MYRVX64I_STATIC_MEDANY-NEXT:	add     x11, x12, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	srli    x10, x10, 8
; MYRVX64I_STATIC_MEDANY-NEXT:	andi    x10, x10, 255
; MYRVX64I_STATIC_MEDANY-NEXT:	add     x10, x11, x10
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %tmp.0.extract.trunc = trunc i64 %elem.coerce to i32
  %elem.sroa.2.0.extract.shift = lshr i32 %tmp.0.extract.trunc, 8
  %elem.sroa.3.0.extract.shift = lshr i32 %tmp.0.extract.trunc, 16
  %conv = and i32 %tmp.0.extract.trunc, 255
  %conv1 = and i32 %elem.sroa.2.0.extract.shift, 255
  %add = add nuw nsw i32 %elem.sroa.3.0.extract.shift, %conv
  %add3 = add nuw nsw i32 %add, %conv1
  %conv4 = zext i32 %add3 to i64
  ret i64 %conv4
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @func_S64(i64 %elem.coerce) local_unnamed_addr #0 {

; MYRVX64I_STATIC_MEDANY-LABEL:func_S64:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, -32
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x1, 24(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x2, 16(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x9, 8(x2)               # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	srli	x9, x10, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x10, x10, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	srli	x10, x10, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	call	func_S32
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x9
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x9, 8(x2)               # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x2, 16(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x1, 24(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %elem.sroa.2.0.extract.shift = lshr i64 %elem.coerce, 32
  %tmp.0.insert.ext = and i64 %elem.coerce, 4294967295
  %call = call i64 @func_S32(i64 %tmp.0.insert.ext)
  %add = add i64 %call, %elem.sroa.2.0.extract.shift
  ret i64 %add
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @func_S128([2 x i64] %elem.coerce) local_unnamed_addr #0 {

; MYRVX64I_STATIC_MEDANY-LABEL:func_S128:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, -32
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x1, 24(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x2, 16(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x9, 8(x2)               # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x9, x11, 0
; MYRVX64I_STATIC_MEDANY-NEXT:	call	func_S64
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x9
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x9, 8(x2)               # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x2, 16(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x1, 24(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %elem.coerce.fca.0.extract = extractvalue [2 x i64] %elem.coerce, 0
  %elem.coerce.fca.1.extract = extractvalue [2 x i64] %elem.coerce, 1
  %call = call i64 @func_S64(i64 %elem.coerce.fca.0.extract)
  %add = add i64 %call, %elem.coerce.fca.1.extract
  ret i64 %add
}

; Function Attrs: norecurse nounwind readonly
define dso_local i64 @func_S256(%struct.S256* nocapture readonly %elem) local_unnamed_addr #1 {

; MYRVX64I_STATIC_MEDANY-LABEL:func_S256:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, -32
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x1, 24(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x2, 16(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x9, 8(x2)               # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x9, x10, 0
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x11, 8(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x10, 0(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	call	func_S128
; MYRVX64I_STATIC_MEDANY-NEXT:	lwu	x11, 16(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	lwu	x11, 20(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	lwu	x11, 24(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	lwu	x11, 28(x9)
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x9, 8(x2)               # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x2, 16(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x1, 24(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, 32
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %.elt = bitcast %struct.S256* %elem to i64*
  %.unpack = load i64, i64* %.elt, align 8
  %0 = insertvalue [2 x i64] undef, i64 %.unpack, 0
  %1 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i64 0, i32 0, i32 1
  %.unpack14 = load i64, i64* %1, align 8
  %2 = insertvalue [2 x i64] %0, i64 %.unpack14, 1
  %call = call i64 @func_S128([2 x i64] %2)
  %arrayidx = getelementptr inbounds %struct.S256, %struct.S256* %elem, i64 0, i32 1, i64 0
  %3 = load i32, i32* %arrayidx, align 8, !tbaa !3
  %conv = zext i32 %3 to i64
  %add = add i64 %call, %conv
  %arrayidx2 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i64 0, i32 1, i64 1
  %4 = load i32, i32* %arrayidx2, align 4, !tbaa !3
  %conv3 = zext i32 %4 to i64
  %add4 = add i64 %add, %conv3
  %arrayidx6 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i64 0, i32 1, i64 2
  %5 = load i32, i32* %arrayidx6, align 8, !tbaa !3
  %conv7 = zext i32 %5 to i64
  %add8 = add i64 %add4, %conv7
  %arrayidx10 = getelementptr inbounds %struct.S256, %struct.S256* %elem, i64 0, i32 1, i64 3
  %6 = load i32, i32* %arrayidx10, align 4, !tbaa !3
  %conv11 = zext i32 %6 to i64
  %add12 = add i64 %add8, %conv11
  ret i64 %add12
}

; Function Attrs: nounwind readonly
define dso_local signext i32 @func_S256_caller() local_unnamed_addr #2 {

; MYRVX64I_STATIC_MEDANY-LABEL:func_S256_caller:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, -48
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x1, 40(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x2, 32(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x0, 225
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x10, x10, 34
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x10, 800
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x10, 24(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x0, 175
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x10, x10, 34
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x10, 600
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x10, 16(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:	lui	x10, 56
; MYRVX64I_STATIC_MEDANY-NEXT:	addiw	x10, x10, -1353
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x10, x10, 14
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x10, -273
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x10, 8(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:	lui	x10, 102401
; MYRVX64I_STATIC_MEDANY-NEXT:	addiw	x10, x10, 717
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x10, x10, 12
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x10, -1948
; MYRVX64I_STATIC_MEDANY-NEXT:	sd	x10, 0(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x10, x2, 0
; MYRVX64I_STATIC_MEDANY-NEXT:	call	func_S256
; MYRVX64I_STATIC_MEDANY-NEXT:	addiw	x10, x10, 0
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x2, 32(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	ld	x1, 40(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDANY-NEXT:	addi	x2, x2, 48
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %byval-temp = alloca %struct.S256, align 8
  %0 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 0, i32 0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #4
  store i8 100, i8* %0, align 8, !tbaa.struct !7
  %elem.sroa.4.0..sroa_idx = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 0, i32 0, i32 0, i32 1
  store i8 -56, i8* %elem.sroa.4.0..sroa_idx, align 1, !tbaa.struct !7
  %elem.sroa.5.0..sroa_idx16 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 0, i32 0, i32 0, i32 2
  store i16 300, i16* %elem.sroa.5.0..sroa_idx16, align 2, !tbaa.struct !7
  %elem.sroa.6.0..sroa_idx18 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 0, i32 0, i32 1
  store i32 400, i32* %elem.sroa.6.0..sroa_idx18, align 4, !tbaa.struct !7
  %elem.sroa.7.0..sroa_idx20 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 0, i32 1
  store i64 3735928559, i64* %elem.sroa.7.0..sroa_idx20, align 8, !tbaa.struct !7
  %elem.sroa.8.0..sroa_idx22 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 1, i64 0
  store i32 600, i32* %elem.sroa.8.0..sroa_idx22, align 8, !tbaa.struct !7
  %elem.sroa.9.0..sroa_idx24 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 1, i64 1
  store i32 700, i32* %elem.sroa.9.0..sroa_idx24, align 4, !tbaa.struct !7
  %elem.sroa.10.0..sroa_idx26 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 1, i64 2
  store i32 800, i32* %elem.sroa.10.0..sroa_idx26, align 8, !tbaa.struct !7
  %elem.sroa.11.0..sroa_idx28 = getelementptr inbounds %struct.S256, %struct.S256* %byval-temp, i64 0, i32 1, i64 3
  store i32 900, i32* %elem.sroa.11.0..sroa_idx28, align 4, !tbaa.struct !7
  %call = call i64 @func_S256(%struct.S256* nonnull %byval-temp)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #4
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 16c6cdf270cec0ae30ef18641075a36f6f17b0b1)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{i64 0, i64 1, !8, i64 1, i64 1, !8, i64 2, i64 2, !9, i64 4, i64 4, !3, i64 8, i64 8, !11, i64 16, i64 16, !8}
!8 = !{!5, !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"short", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"long", !5, i64 0}
