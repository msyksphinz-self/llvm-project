; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s


target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-elf"

; Function Attrs: norecurse nounwind readnone
define dso_local signext i32 @rotate_left(i32 signext %a) local_unnamed_addr #0 {

; MYRVX32I-LABEL:rotate_left:
; MYRVX32I:     # %bb.0:
; MYRVX32I-NEXT:	srai	x11, x10, 2
; MYRVX32I-NEXT:	slli	x10, x10, 30
; MYRVX32I-NEXT:	or	x10, x10, x11
; MYRVX32I-NEXT:	ret


; MYRVX64I-LABEL:rotate_left:
; MYRVX64I:     # %bb.0:
; MYRVX64I-NEXT:	srli	x11, x10, 2
; MYRVX64I-NEXT:	slli	x10, x10, 30
; MYRVX64I-NEXT:	or	x10, x10, x11
; MYRVX64I-NEXT:	sext.w	x10, x10
; MYRVX64I-NEXT:	ret

entry:
  %shl = shl i32 %a, 30
  %shr = ashr i32 %a, 2
  %or = or i32 %shl, %shr
  ret i32 %or
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
