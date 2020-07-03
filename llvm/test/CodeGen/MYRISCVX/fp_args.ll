; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local float @test_fp_arg(float %a, float %b, float %c) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_fp_arg:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f10, f11
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f10, f0, f12
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_fp_arg:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f10, f11
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f10, f0, f12
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %add = fadd float %a, %b
  %add1 = fadd float %add, %c
  ret float %add1
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @test_dp_arg(double %a, double %b, double %c) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_dp_arg:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f10, f11
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f10, f0, f12
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_dp_arg:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f10, f11
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f10, f0, f12
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %add = fadd double %a, %b
  %add1 = fadd double %add, %c
  ret double %add1
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local float @test_fp_longarg(float %f0, float %f1, float %f2, float %f3, float %f4, float %f5, float %f6, float %f7, float %f8, float %f9) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_fp_longarg:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f10, f11
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f12
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f13
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f14
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f15
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f16
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f17
; MYRVX32I_STATIC_MEDLOW-NEXT:	flw	f1, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	flw	f1, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.s	f10, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_fp_longarg:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f10, f11
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f12
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f13
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f14
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f15
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f16
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f17
; MYRVX64I_STATIC_MEDLOW-NEXT:	flw	f1, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f0, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	flw	f1, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.s	f10, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %add = fadd float %f0, %f1
  %add1 = fadd float %add, %f2
  %add2 = fadd float %add1, %f3
  %add3 = fadd float %add2, %f4
  %add4 = fadd float %add3, %f5
  %add5 = fadd float %add4, %f6
  %add6 = fadd float %add5, %f7
  %add7 = fadd float %add6, %f8
  %add8 = fadd float %add7, %f9
  ret float %add8
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local double @test_dp_longarg(double %f0, double %f1, double %f2, double %f3, double %f4, double %f5, double %f6, double %f7, double %f8, double %f9) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:test_dp_longarg:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f10, f11
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f12
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f13
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f14
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f15
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f16
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f17
; MYRVX32I_STATIC_MEDLOW-NEXT:	fld	f1, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	fld	f1, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	fadd.d	f10, f0, f1
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret

; MYRVX64I_STATIC_MEDLOW-LABEL:test_dp_longarg:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f10, f11
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f12
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f13
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f14
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f15
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f16
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f17
; MYRVX64I_STATIC_MEDLOW-NEXT:	fld	f1, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f0, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	fld	f1, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:	fadd.d	f10, f0, f1
; MYRVX64I_STATIC_MEDLOW-NEXT:	ret

entry:
  %add = fadd double %f0, %f1
  %add1 = fadd double %add, %f2
  %add2 = fadd double %add1, %f3
  %add3 = fadd double %add2, %f4
  %add4 = fadd double %add3, %f5
  %add5 = fadd double %add4, %f6
  %add6 = fadd double %add5, %f7
  %add7 = fadd double %add6, %f8
  %add8 = fadd double %add7, %f9
  ret double %add8
}

attributes #0 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git f3702bb4f36df0b58e5c73a0b9c57ed68c67c898)"}
