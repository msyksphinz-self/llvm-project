; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

define dso_local signext i32 @add32(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: add32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT: 	addw	x10, x11, x10
; MYRVX64I-NEXT:  	ret

entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define dso_local signext i32 @sub32(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: sub32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   	subw	x10, x10, x11
; MYRVX64I-NEXT:  	ret

entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

define dso_local signext i32 @mul32(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: mul32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   	mulw	x10, x11, x10
; MYRVX64I-NEXT:  	ret

entry:
  %mul = mul nsw i32 %b, %a
  ret i32 %mul
}

define dso_local signext i32 @div32(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {
; MYRVX64I-LABEL: div32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:   	divw	x10, x10, x11
; MYRVX64I-NEXT:  	ret


entry:
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define dso_local i32 @div32u(i32 %a, i32 %b) local_unnamed_addr #0 {
; MYRVX64I-LABEL: div32u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:    	slli	x11, x11, 32
; MYRVX64I-NEXT:  	srli	x11, x11, 32
; MYRVX64I-NEXT: 	slli	x10, x10, 32
; MYRVX64I-NEXT: 	srli	x10, x10, 32
; MYRVX64I-NEXT: 	divu	x10, x10, x11
; MYRVX64I-NEXT: 	ret

entry:
  %div = udiv i32 %a, %b
  ret i32 %div
}

define dso_local signext i32 @shift_left32(i32 %a) local_unnamed_addr #0 {
; MYRVX64I-LABEL: shift_left32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	slliw	x10, x10, 10
; MYRVX64I-NEXT:  	ret

entry:
  %shl = shl i32 %a, 10
  ret i32 %shl
}

define dso_local signext i32 @shift_left32u(i32 %ua) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_left32u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	slliw	x10, x10, 11
; MYRVX64I-NEXT:	ret

entry:
  %shl = shl i32 %ua, 11
  ret i32 %shl
}

define dso_local signext i32 @shift_right32(i32 signext %a) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_right32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	srai     x10, x10, 2
; MYRVX64I-NEXT:  	ret

entry:
  %shr = ashr i32 %a, 2
  ret i32 %shr
}

define dso_local i32 @shift_right32u(i32 %ua) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_right32u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:    addi    x11, x0, 3
; MYRVX64I-NEXT:    slli    x11, x11, 30
; MYRVX64I-NEXT:    and     x10, x10, x11
; MYRVX64I-NEXT:    srli     x10, x10, 30
; MYRVX64I-NEXT:    ret

entry:
  %shr = lshr i32 %ua, 30
  ret i32 %shr
}

define dso_local signext i32 @shift_leftv32(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_leftv32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	slli	x11, x11, 32
; MYRVX64I-NEXT:  	srli	x11, x11, 32
; MYRVX64I-NEXT: 	sllw	x10, x10, x11
; MYRVX64I-NEXT: 	ret

entry:
  %shl = shl i32 %a, %b
  ret i32 %shl
}

define dso_local signext i32 @shift_rightv32(i32 signext %a, i32 signext %b)local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_rightv32:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	slli	x11, x11, 32
; MYRVX64I-NEXT:  	srli	x11, x11, 32
; MYRVX64I-NEXT: 	sra	x10, x10, x11
; MYRVX64I-NEXT: 	ret

entry:
  %shr = ashr i32 %a, %b
  ret i32 %shr
}

define dso_local i32 @shift_leftv32u(i32 %ua, i32 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_leftv32u:
; MYRVX64I:       # %bb.0:
; MYRVX64I-NEXT:  	slli	x11, x11, 32
; MYRVX64I-NEXT:  	srli	x11, x11, 32
; MYRVX64I-NEXT: 	sll 	x10, x10, x11
; MYRVX64I-NEXT: 	ret

entry:
  %shl = shl i32 %ua, %b
  ret i32 %shl
}

define dso_local i32 @shift_rightv32u(i32 %ua, i32 %b) local_unnamed_addr #0 {

; MYRVX64I-LABEL: shift_rightv32u:
; MYRVX64I:      # %bb.0:
; MYRVX64I-NEXT: 	slli	x11, x11, 32
; MYRVX64I-NEXT: 	srli	x11, x11, 32
; MYRVX64I-NEXT: 	slli	x10, x10, 32
; MYRVX64I-NEXT: 	srli	x10, x10, 32
; MYRVX64I-NEXT: 	srl     x10, x10, x11
; MYRVX64I-NEXT: 	ret

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
