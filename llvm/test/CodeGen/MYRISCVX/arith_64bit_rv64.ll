; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @add64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: add64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   add     x10, x11, x10
; MYRVX64I-NEXT:   ret

entry:
  %add = add nsw i64 %b, %a
  ret i64 %add
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @sub64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: sub64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   sub     x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %sub = sub nsw i64 %a, %b
  ret i64 %sub
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @mul64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: mul64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   mul     x10, x11, x10
; MYRVX64I-NEXT:   ret

entry:
  %mul = mul nsw i64 %b, %a
  ret i64 %mul
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @div64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: div64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   div     x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %div = sdiv i64 %a, %b
  ret i64 %div
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @div64u(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: div64u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   divu    x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %div = udiv i64 %a, %b
  ret i64 %div
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_left64(i64 %a) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_left64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   slli    x10, x10, 10
; MYRVX64I-NEXT:   ret

entry:
  %shl = shl i64 %a, 10
  ret i64 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_left64u(i64 %ua) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_left64u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   slli     x10, x10, 11
; MYRVX64I-NEXT:   ret

entry:
  %shl = shl i64 %ua, 11
  ret i64 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_right64(i64 %a) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_right64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   srai     x10, x10, 2
; MYRVX64I-NEXT:   ret

entry:
  %shr = ashr i64 %a, 2
  ret i64 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_right64u(i64 %ua) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_right64u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   srli    x10, x10, 30
; MYRVX64I-NEXT:   ret

entry:
  %shr = lshr i64 %ua, 30
  ret i64 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_leftv64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_leftv64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   sll     x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %shl = shl i64 %a, %b
  ret i64 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_rightv64(i64 %a, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_rightv64:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   sra     x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %shr = ashr i64 %a, %b
  ret i64 %shr
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_leftv64u(i64 %ua, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_leftv64u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   sll     x10, x10, x11
; MYRVX64I-NEXT:   ret

entry:
  %shl = shl i64 %ua, %b
  ret i64 %shl
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @shift_rightv64u(i64 %ua, i64 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_rightv64u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:    srl     x10, x10, x11
; MYRVX64I-NEXT:    ret

entry:
  %shr = lshr i64 %ua, %b
  ret i64 %shr
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (git@msyksphinz-self:msyksphinz-self/llvm-project.git be06a078b440bc136fea17f2cc85addadedce840)"}
