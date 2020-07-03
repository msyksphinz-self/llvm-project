; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @test_fp_math(float* nocapture readonly %a, float* nocapture readonly %b, float* nocapture readonly %c, float* nocapture %result) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_fp_math:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	flw	f0, 0(x12)
; MYRVX32I_STATIC_MEDLOW-NEXT:	flw	f1, 0(x11)
; MYRVX32I_STATIC_MEDLOW-NEXT:	flw	f2, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fmadd.s	f3, f2, f1, f0
; MYRVX32I_STATIC_MEDLOW-NEXT:	fmsub.s	f3, f2, f0, f3
; MYRVX32I_STATIC_MEDLOW-NEXT:	fnmsub.s	f1, f1, f0, f3
; MYRVX32I_STATIC_MEDLOW-NEXT:	fnmadd.s	f0, f2, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	fsw	f0, 0(x13)
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_fp_math:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	flw	f0, 0(x12)
; MYRVX64I_STATIC_MEDLOW-NEXT:	flw	f1, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:	flw	f2, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fmadd.s	f3, f2, f1, f0
; MYRVX64I_STATIC_MEDLOW-NEXT:	fmsub.s	f3, f2, f0, f3
; MYRVX64I_STATIC_MEDLOW-NEXT:	fnmsub.s	f1, f1, f0, f3
; MYRVX64I_STATIC_MEDLOW-NEXT:	fnmadd.s	f0, f2, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	fsw	f0, 0(x13)
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %0 = load float, float* %a, align 4, !tbaa !2
  %1 = load float, float* %b, align 4, !tbaa !2
  %mul = fmul float %0, %1
  %2 = load float, float* %c, align 4, !tbaa !2
  %add = fadd float %mul, %2
  %mul1 = fmul float %0, %2
  %sub = fsub float %mul1, %add
  %3 = fmul float %1, %2
  %add3 = fsub float %sub, %3
  %fneg4 = fneg float %0
  %mul5 = fmul float %2, %fneg4
  %sub6 = fsub float %mul5, %add3
  store float %sub6, float* %result, align 4, !tbaa !2
  ret void
}

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @test_dp_math(double* nocapture readonly %a, double* nocapture readonly %b, double* nocapture readonly %c, double* nocapture %result) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_dp_math:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	fld	f0, 0(x12)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fld	f1, 0(x11)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fld	f2, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fmadd.d	f3, f2, f1, f0
; MYRVX32I_STATIC_MEDLOW-NEXT:	fmsub.d	f3, f2, f1, f3
; MYRVX32I_STATIC_MEDLOW-NEXT:	fnmsub.d	f1, f1, f0, f3
; MYRVX32I_STATIC_MEDLOW-NEXT:	fnmadd.d	f0, f2, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	fsd	f0, 0(x13)
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_dp_math:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	fld	f0, 0(x12)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fld	f1, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fld	f2, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fmadd.d	f3, f2, f1, f0
; MYRVX64I_STATIC_MEDLOW-NEXT:	fmsub.d	f3, f2, f1, f3
; MYRVX64I_STATIC_MEDLOW-NEXT:	fnmsub.d	f1, f1, f0, f3
; MYRVX64I_STATIC_MEDLOW-NEXT:	fnmadd.d	f0, f2, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	fsd	f0, 0(x13)
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %0 = load double, double* %a, align 8, !tbaa !6
  %1 = load double, double* %b, align 8, !tbaa !6
  %mul = fmul double %0, %1
  %2 = load double, double* %c, align 8, !tbaa !6
  %add = fadd double %mul, %2
  %sub = fsub double %mul, %add
  %3 = fmul double %1, %2
  %add3 = fsub double %sub, %3
  %fneg4 = fneg double %0
  %mul5 = fmul double %2, %fneg4
  %sub6 = fsub double %mul5, %add3
  store double %sub6, double* %result, align 8, !tbaa !6
  ret void
}

attributes #0 = { nofree norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git f0f234f33736f4340a30af831fe89649ae1e19aa)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"float", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}
