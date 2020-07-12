; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

define dso_local i32 @add32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: add32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT: 	add	x10, x11, x10
; MYRVX32I-NEXT:  	ret

entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define dso_local i32 @sub32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: sub32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT: 	sub	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

define dso_local i32 @mul32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: mul32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT: 	mul	x10, x11, x10
; MYRVX32I-NEXT: 	ret

entry:
  %mul = mul nsw i32 %b, %a
  ret i32 %mul
}

define dso_local i32 @div32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: div32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	div	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define dso_local i32 @div32u(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: div32u:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	divu	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %div = udiv i32 %a, %b
  ret i32 %div
}

define dso_local i32 @shift_left32(i32 %a) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_left32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	slli	x10, x10, 10
; MYRVX32I-NEXT:  	ret

entry:
  %shl = shl i32 %a, 10
  ret i32 %shl
}

define dso_local i32 @shift_left32u(i32 %ua) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_left32u:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	slli	x10, x10, 11
; MYRVX32I-NEXT:  	ret

entry:
  %shl = shl i32 %ua, 11
  ret i32 %shl
}

define dso_local i32 @shift_right32(i32 %a) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_right32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	srai	x10, x10, 2
; MYRVX32I-NEXT:  	ret

entry:
  %shr = ashr i32 %a, 2
  ret i32 %shr
}

define dso_local i32 @shift_right32u(i32 %ua) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_right32u:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	srli	x10, x10, 30
; MYRVX32I-NEXT:  	ret

entry:
  %shr = lshr i32 %ua, 30
  ret i32 %shr
}

define dso_local i32 @shift_leftv32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_leftv32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	sll	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %shl = shl i32 %a, %b
  ret i32 %shl
}

define dso_local i32 @shift_rightv32(i32 %a, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_rightv32:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	sra	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %shr = ashr i32 %a, %b
  ret i32 %shr
}

define dso_local i32 @shift_leftv32u(i32 %ua, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_leftv32u:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	sll	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %shl = shl i32 %ua, %b
  ret i32 %shl
}

define dso_local i32 @shift_rightv32u(i32 %ua, i32 %b) local_unnamed_addr #0 {

; MYRVX32I-LABEL: shift_rightv32u:
; MYRVX32I:       # %bb.0:
; MYRVX32I-NEXT:   	srl	x10, x10, x11
; MYRVX32I-NEXT:  	ret

entry:
  %shr = lshr i32 %ua, %b
  ret i32 %shr
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{!"clang version 10.0.1 (git@msyksphinz-self:msyksphinz-self/llvm-project.git be06a078b440bc136fea17f2cc85addadedce840)"}
