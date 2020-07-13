; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDANY %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDANY %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

define dso_local signext i32 @sum_array(i32* nocapture readonly %array, i32 signext %idx0, i32 signext %idx1, i32 signext %idx2) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:sum_array:
; MYRVX32I_STATIC_MEDLOW:        # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	slli	x11, x11, 2
; MYRVX32I_STATIC_MEDLOW-NEXT:	add	x11, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT:	lw	x11, 0(x11)
; MYRVX32I_STATIC_MEDLOW-NEXT:	slli	x12, x12, 2
; MYRVX32I_STATIC_MEDLOW-NEXT:	add	x12, x10, x12
; MYRVX32I_STATIC_MEDLOW-NEXT:	lw	x12, 0(x12)
; MYRVX32I_STATIC_MEDLOW-NEXT:	add	x11, x12, x11
; MYRVX32I_STATIC_MEDLOW-NEXT:	slli	x12, x13, 2
; MYRVX32I_STATIC_MEDLOW-NEXT:	add	x10, x10, x12
; MYRVX32I_STATIC_MEDLOW-NEXT:	lw	x10, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	add	x10, x11, x10
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX32I_STATIC_MEDANY-LABEL:sum_array:
; MYRVX32I_STATIC_MEDANY:        # %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:	slli	x11, x11, 2
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x10, x11
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x11, 0(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:	slli	x12, x12, 2
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x12, x10, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x12, 0(x12)
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x11, x12, x11
; MYRVX32I_STATIC_MEDANY-NEXT:	slli	x12, x13, 2
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x10, x12
; MYRVX32I_STATIC_MEDANY-NEXT:	lw	x10, 0(x10)
; MYRVX32I_STATIC_MEDANY-NEXT:	add	x10, x11, x10
; MYRVX32I_STATIC_MEDANY-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:sum_array:
; MYRVX64I_STATIC_MEDLOW:        # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	slli	x11, x11, 2
; MYRVX64I_STATIC_MEDLOW-NEXT:	add	x11, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT:	lw	x11, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:	slli	x12, x12, 2
; MYRVX64I_STATIC_MEDLOW-NEXT:	add	x12, x10, x12
; MYRVX64I_STATIC_MEDLOW-NEXT:	lw	x12, 0(x12)
; MYRVX64I_STATIC_MEDLOW-NEXT:	add	x11, x12, x11
; MYRVX64I_STATIC_MEDLOW-NEXT:	slli	x12, x13, 2
; MYRVX64I_STATIC_MEDLOW-NEXT:	add	x10, x10, x12
; MYRVX64I_STATIC_MEDLOW-NEXT:	lw	x10, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:	addw	x10, x11, x10
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDANY-LABEL:sum_array:
; MYRVX64I_STATIC_MEDANY:        # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x11, x11, 2
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x11, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	lw	x11, 0(x11)
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x12, x12, 2
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x12, x10, x12
; MYRVX64I_STATIC_MEDANY-NEXT:	lw	x12, 0(x12)
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x11, x12, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	slli	x12, x13, 2
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x10, x12
; MYRVX64I_STATIC_MEDANY-NEXT:	lw	x10, 0(x10)
; MYRVX64I_STATIC_MEDANY-NEXT:	addw	x10, x11, x10
; MYRVX64I_STATIC_MEDANY-NEXT:	ret


entry:
  %idxprom = sext i32 %idx0 to i64
  %arrayidx = getelementptr inbounds i32, i32* %array, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !3
  %idxprom1 = sext i32 %idx1 to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %array, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4, !tbaa !3
  %add = add nsw i32 %1, %0
  %idxprom3 = sext i32 %idx2 to i64
  %arrayidx4 = getelementptr inbounds i32, i32* %array, i64 %idxprom3
  %2 = load i32, i32* %arrayidx4, align 4, !tbaa !3
  %add5 = add nsw i32 %add, %2
  ret i32 %add5
}

attributes #0 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git e947bd789bb551e8ed77e4eda9d0c10a40e2f863)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
