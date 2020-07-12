; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"


define dso_local signext i32 @arithmetics() #0 {

;MYRVX32I-LABEL:arithmetics:
;MYRVX32I:      # %bb.0:
;MYRVX32I-NEXT:	addi	x2, x2, -56
;MYRVX32I-NEXT:	addi	x10, x0, 5
;MYRVX32I-NEXT:	sw	x10, 52(x2)
;MYRVX32I-NEXT:	addi	x10, x0, 2
;MYRVX32I-NEXT:	sw	x10, 48(x2)
;MYRVX32I-NEXT:	addi	x10, x0, -5
;MYRVX32I-NEXT:	sw	x10, 44(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	lw	x11, 48(x2)
;MYRVX32I-NEXT:	add	x10, x10, x11
;MYRVX32I-NEXT:	sw	x10, 40(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	lw	x11, 48(x2)
;MYRVX32I-NEXT:	sub	x10, x10, x11
;MYRVX32I-NEXT:	sw	x10, 36(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	lw	x11, 48(x2)
;MYRVX32I-NEXT:	mul	x10, x10, x11
;MYRVX32I-NEXT:	sw	x10, 32(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	slli	x10, x10, 2
;MYRVX32I-NEXT:	sw	x10, 28(x2)
;MYRVX32I-NEXT:	lw	x10, 44(x2)
;MYRVX32I-NEXT:	slli	x10, x10, 1
;MYRVX32I-NEXT:	sw	x10, 12(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	srai	x10, x10, 2
;MYRVX32I-NEXT:	sw	x10, 24(x2)
;MYRVX32I-NEXT:	lw	x10, 44(x2)
;MYRVX32I-NEXT:	srli	x10, x10, 30
;MYRVX32I-NEXT:	sw	x10, 8(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	addi	x11, x0, 1
;MYRVX32I-NEXT:	sll	x10, x11, x10
;MYRVX32I-NEXT:	sw	x10, 20(x2)
;MYRVX32I-NEXT:	lw	x10, 48(x2)
;MYRVX32I-NEXT:	sll	x10, x11, x10
;MYRVX32I-NEXT:	sw	x10, 4(x2)
;MYRVX32I-NEXT:	lw	x10, 52(x2)
;MYRVX32I-NEXT:	addi	x11, x0, 128
;MYRVX32I-NEXT:	srl	x10, x11, x10
;MYRVX32I-NEXT:	sw	x10, 16(x2)
;MYRVX32I-NEXT:	lw	x10, 48(x2)
;MYRVX32I-NEXT:	lw	x11, 52(x2)
;MYRVX32I-NEXT:	sra	x10, x10, x11
;MYRVX32I-NEXT:	sw	x10, 0(x2)
;MYRVX32I-NEXT:	addi	x10, x0, 0
;MYRVX32I-NEXT:	addi	x2, x2, 56
;MYRVX32I-NEXT:	ret


;MYRVX64I-LABEL:arithmetics:
;MYRVX64I:      # %bb.0:
;MYRVX64I-NEXT:	addi	x2, x2, -56
;MYRVX64I-NEXT:	addi	x10, x0, 5
;MYRVX64I-NEXT:	sw	x10, 52(x2)
;MYRVX64I-NEXT:	addi	x10, x0, 2
;MYRVX64I-NEXT:	sw	x10, 48(x2)
;MYRVX64I-NEXT:	addi	x10, x0, 1
;MYRVX64I-NEXT:	slli	x11, x10, 32
;MYRVX64I-NEXT:	addi	x11, x11, -5
;MYRVX64I-NEXT:	sw	x11, 44(x2)
;MYRVX64I-NEXT:	lw	x11, 52(x2)
;MYRVX64I-NEXT:	lw	x12, 48(x2)
;MYRVX64I-NEXT:	add	x11, x11, x12
;MYRVX64I-NEXT:	sw	x11, 40(x2)
;MYRVX64I-NEXT:	lw	x11, 52(x2)
;MYRVX64I-NEXT:	lw	x12, 48(x2)
;MYRVX64I-NEXT:	sub	x11, x11, x12
;MYRVX64I-NEXT:	sw	x11, 36(x2)
;MYRVX64I-NEXT:	lw	x11, 52(x2)
;MYRVX64I-NEXT:	lw	x12, 48(x2)
;MYRVX64I-NEXT:	mul	x11, x11, x12
;MYRVX64I-NEXT:	sw	x11, 32(x2)
;MYRVX64I-NEXT:	lw	x11, 52(x2)
;MYRVX64I-NEXT:	slli	x11, x11, 2
;MYRVX64I-NEXT:	sw	x11, 28(x2)
;MYRVX64I-NEXT:	lw	x11, 44(x2)
;MYRVX64I-NEXT:	slli	x11, x11, 1
;MYRVX64I-NEXT:	sw	x11, 12(x2)
;MYRVX64I-NEXT:	lw	x11, 52(x2)
;MYRVX64I-NEXT:	srli	x11, x11, 2
;MYRVX64I-NEXT:	sw	x11, 24(x2)
;MYRVX64I-NEXT:	lwu	x11, 44(x2)
;MYRVX64I-NEXT:	srli	x11, x11, 30
;MYRVX64I-NEXT:	sw	x11, 8(x2)
;MYRVX64I-NEXT:	lwu	x11, 52(x2)
;MYRVX64I-NEXT:	sll	x11, x10, x11
;MYRVX64I-NEXT:	sw	x11, 20(x2)
;MYRVX64I-NEXT:	lwu	x11, 48(x2)
;MYRVX64I-NEXT:	sll	x10, x10, x11
;MYRVX64I-NEXT:	sw	x10, 4(x2)
;MYRVX64I-NEXT:	lwu	x10, 52(x2)
;MYRVX64I-NEXT:	addi	x11, x0, 128
;MYRVX64I-NEXT:	srl	x10, x11, x10
;MYRVX64I-NEXT:	sw	x10, 16(x2)
;MYRVX64I-NEXT:	lw	x10, 48(x2)
;MYRVX64I-NEXT:	lwu	x11, 52(x2)
;MYRVX64I-NEXT:	sra	x10, x10, x11
;MYRVX64I-NEXT:	sw	x10, 0(x2)
;MYRVX64I-NEXT:	addi	x10, x0, 0
;MYRVX64I-NEXT:	addi	x2, x2, 56
;MYRVX64I-NEXT:	ret


entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %a1 = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  %i = alloca i32, align 4
  %f1 = alloca i32, align 4
  %g1 = alloca i32, align 4
  %h1 = alloca i32, align 4
  %i1 = alloca i32, align 4
  store i32 5, i32* %a, align 4
  store i32 2, i32* %b, align 4
  store i32 -5, i32* %a1, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, i32* %c, align 4
  %2 = load i32, i32* %a, align 4
  %3 = load i32, i32* %b, align 4
  %sub = sub nsw i32 %2, %3
  store i32 %sub, i32* %d, align 4
  %4 = load i32, i32* %a, align 4
  %5 = load i32, i32* %b, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, i32* %e, align 4
  %6 = load i32, i32* %a, align 4
  %shl = shl i32 %6, 2
  store i32 %shl, i32* %f, align 4
  %7 = load i32, i32* %a1, align 4
  %shl1 = shl i32 %7, 1
  store i32 %shl1, i32* %f1, align 4
  %8 = load i32, i32* %a, align 4
  %shr = ashr i32 %8, 2
  store i32 %shr, i32* %g, align 4
  %9 = load i32, i32* %a1, align 4
  %shr2 = lshr i32 %9, 30
  store i32 %shr2, i32* %g1, align 4
  %10 = load i32, i32* %a, align 4
  %shl3 = shl i32 1, %10
  store i32 %shl3, i32* %h, align 4
  %11 = load i32, i32* %b, align 4
  %shl4 = shl i32 1, %11
  store i32 %shl4, i32* %h1, align 4
  %12 = load i32, i32* %a, align 4
  %shr5 = ashr i32 128, %12
  store i32 %shr5, i32* %i, align 4
  %13 = load i32, i32* %b, align 4
  %14 = load i32, i32* %a, align 4
  %shr6 = ashr i32 %13, %14
  store i32 %shr6, i32* %i1, align 4
  ret i32 0
}


attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
