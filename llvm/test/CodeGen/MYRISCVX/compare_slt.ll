; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

source_filename = "compare_slt.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: norecurse nounwind readnone
define dso_local signext i32 @compare_slt(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {

;MYRVX32I-LABEL:compare_slt:
;MYRVX32I:     # %bb.0:
;MYRVX32I-NEXT:	xor	x12, x10, x11
;MYRVX32I-NEXT:	sltu	x13, x0, x12
;MYRVX32I-NEXT:	andi	x13, x13, 1
;MYRVX32I-NEXT:	sltiu	x12, x12, 1
;MYRVX32I-NEXT:	andi	x12, x12, 1
;MYRVX32I-NEXT:	add	x12, x12, x13
;MYRVX32I-NEXT:	slt	x13, x10, x11
;MYRVX32I-NEXT:	andi	x14, x13, 1
;MYRVX32I-NEXT:	add	x12, x12, x14
;MYRVX32I-NEXT:	slt	x10, x11, x10
;MYRVX32I-NEXT:	xori	x11, x10, 1
;MYRVX32I-NEXT:	andi	x11, x11, 1
;MYRVX32I-NEXT:	add	x11, x12, x11
;MYRVX32I-NEXT:	andi	x10, x10, 1
;MYRVX32I-NEXT:	add	x10, x11, x10
;MYRVX32I-NEXT:	xori	x11, x13, 1
;MYRVX32I-NEXT:	andi	x11, x11, 1
;MYRVX32I-NEXT:	add	x10, x10, x11
;MYRVX32I-NEXT:	ret


;MYRVX64I-LABEL:compare_slt:
;MYRVX64I:      # %bb.0:
;MYRVX64I-NEXT:	xor	x12, x10, x11
;MYRVX64I-NEXT:	sltu	x13, x0, x12
;MYRVX64I-NEXT:	andi	x13, x13, 1
;MYRVX64I-NEXT:	sltiu	x12, x12, 1
;MYRVX64I-NEXT:	andi	x12, x12, 1
;MYRVX64I-NEXT:	add	x12, x12, x13
;MYRVX64I-NEXT:	slt	x13, x10, x11
;MYRVX64I-NEXT:	andi	x14, x13, 1
;MYRVX64I-NEXT:	add	x12, x12, x14
;MYRVX64I-NEXT:	slt	x10, x11, x10
;MYRVX64I-NEXT:	xori	x11, x10, 1
;MYRVX64I-NEXT:	andi	x11, x11, 1
;MYRVX64I-NEXT:	add	x11, x12, x11
;MYRVX64I-NEXT:	andi	x10, x10, 1
;MYRVX64I-NEXT:	add	x10, x11, x10
;MYRVX64I-NEXT:	xori	x11, x13, 1
;MYRVX64I-NEXT:	andi	x11, x11, 1
;MYRVX64I-NEXT:	add	x10, x10, x11
;MYRVX64I-NEXT:	ret

entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  %cmp1 = icmp ne i32 %a, %b
  %conv2 = zext i1 %cmp1 to i32
  %cmp3 = icmp slt i32 %a, %b
  %conv4 = zext i1 %cmp3 to i32
  %cmp5 = icmp sle i32 %a, %b
  %conv6 = zext i1 %cmp5 to i32
  %cmp7 = icmp sgt i32 %a, %b
  %conv8 = zext i1 %cmp7 to i32
  %cmp9 = icmp sge i32 %a, %b
  %conv10 = zext i1 %cmp9 to i32
  %add = add nuw nsw i32 %conv, %conv2
  %add11 = add nuw nsw i32 %add, %conv4
  %add12 = add nuw nsw i32 %add11, %conv6
  %add13 = add nuw nsw i32 %add12, %conv8
  %add14 = add nuw nsw i32 %add13, %conv10
  ret i32 %add14
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
