; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @test_switch(i32 signext %operation, i32 signext %a, i32 signext %b) #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: test_switch:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -16
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x11, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x12, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %entry
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_5
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_2
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %entry
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 3
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_6
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %entry
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_7
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_8
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_4:                                 # %sw.bb
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	add	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_5:                                 # %sw.bb1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sub	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_6:                                 # %sw.bb2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_7:                                 # %sw.bb3
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	div	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_8:                                 # %sw.default
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_9:                                 # %sw.epilog
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 16
; MYRVX32I_STATIC_MEDLOW-NEXT: 	ret

; MYRVX64I_STATIC_MEDLOW-LABEL: test_switch:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -16
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x11, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x12, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %entry
; MYRVX64I_STATIC_MEDLOW-NEXT: 	slli	x11, x10, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	srli	x11, x11, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x12, x0, 2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beq	x11, x12, $BB0_5
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_2
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %entry
; MYRVX64I_STATIC_MEDLOW-NEXT: 	slli	x11, x10, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	srli	x11, x11, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x12, x0, 3
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beq	x11, x12, $BB0_6
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %entry
; MYRVX64I_STATIC_MEDLOW-NEXT: 	slli	x10, x10, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	srli	x10, x10, 32
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beq	x10, x11, $BB0_7
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_8
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_4:                                 # %sw.bb
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_5:                                 # %sw.bb1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sub	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_6:                                 # %sw.bb2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_7:                                 # %sw.bb3
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	div	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_8:                                 # %sw.default
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_9:                                 # %sw.epilog
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 16
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ret

entry:
  %operation.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  store i32 %operation, i32* %operation.addr, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %operation.addr, align 4
  switch i32 %0, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i32, i32* %a.addr, align 4
  %2 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %1, %2
  store i32 %add, i32* %ret, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %3 = load i32, i32* %a.addr, align 4
  %4 = load i32, i32* %b.addr, align 4
  %sub = sub nsw i32 %3, %4
  store i32 %sub, i32* %ret, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %5 = load i32, i32* %a.addr, align 4
  %6 = load i32, i32* %b.addr, align 4
  %mul = mul nsw i32 %5, %6
  store i32 %mul, i32* %ret, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %7 = load i32, i32* %a.addr, align 4
  %8 = load i32, i32* %b.addr, align 4
  %div = sdiv i32 %7, %8
  store i32 %div, i32* %ret, align 4
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  %9 = load i32, i32* %a.addr, align 4
  store i32 %9, i32* %ret, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %10 = load i32, i32* %ret, align 4
  ret i32 %10
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 9bd55c2f21c13001f663d8357eac3083e3e21005)"}
