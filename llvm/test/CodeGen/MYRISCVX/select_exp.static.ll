; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @test_ternary(i32 signext %operation, i32 signext %a, i32 signext %b) #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: test_ternary:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -16
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x11, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x12, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %cond.true
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	add	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_12
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %cond.false
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %cond.true2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sub	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_11
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_4:                                 # %cond.false3
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 3
; MYRVX32I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_6
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_5
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_5:                                 # %cond.true5
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_10
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_6:                                 # %cond.false6
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 12(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_8
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_7
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_7:                                 # %cond.true8
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	div	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_8:                                 # %cond.false9
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_9:                                 # %cond.end
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_10
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_10:                                # %cond.end10
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_11
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_11:                                # %cond.end12
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_12
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_12:                                # %cond.end14
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 16
; MYRVX32I_STATIC_MEDLOW-NEXT: 	ret


; MYRVX64I_STATIC_MEDLOW-LABEL: test_ternary:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -16
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x11, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x12, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %cond.true
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_12
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %cond.false
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %cond.true2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sub	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_11
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_4:                                 # %cond.false3
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 3
; MYRVX64I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_6
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_5
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_5:                                 # %cond.true5
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_10
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_6:                                 # %cond.false6
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	bne	x10, x11, $BB0_8
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_7
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_7:                                 # %cond.true8
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	div	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_8:                                 # %cond.false9
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_9
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_9:                                 # %cond.end
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_10
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_10:                                # %cond.end10
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_11
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_11:                                # %cond.end12
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_12
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_12:                                # %cond.end14
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
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
  %cmp = icmp eq i32 %0, 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %1 = load i32, i32* %a.addr, align 4
  %2 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %1, %2
  br label %cond.end14

cond.false:                                       ; preds = %entry
  %3 = load i32, i32* %operation.addr, align 4
  %cmp1 = icmp eq i32 %3, 2
  br i1 %cmp1, label %cond.true2, label %cond.false3

cond.true2:                                       ; preds = %cond.false
  %4 = load i32, i32* %a.addr, align 4
  %5 = load i32, i32* %b.addr, align 4
  %sub = sub nsw i32 %4, %5
  br label %cond.end12

cond.false3:                                      ; preds = %cond.false
  %6 = load i32, i32* %operation.addr, align 4
  %cmp4 = icmp eq i32 %6, 3
  br i1 %cmp4, label %cond.true5, label %cond.false6

cond.true5:                                       ; preds = %cond.false3
  %7 = load i32, i32* %a.addr, align 4
  %8 = load i32, i32* %b.addr, align 4
  %mul = mul nsw i32 %7, %8
  br label %cond.end10

cond.false6:                                      ; preds = %cond.false3
  %9 = load i32, i32* %operation.addr, align 4
  %cmp7 = icmp eq i32 %9, 4
  br i1 %cmp7, label %cond.true8, label %cond.false9

cond.true8:                                       ; preds = %cond.false6
  %10 = load i32, i32* %a.addr, align 4
  %11 = load i32, i32* %b.addr, align 4
  %div = sdiv i32 %10, %11
  br label %cond.end

cond.false9:                                      ; preds = %cond.false6
  %12 = load i32, i32* %a.addr, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false9, %cond.true8
  %cond = phi i32 [ %div, %cond.true8 ], [ %12, %cond.false9 ]
  br label %cond.end10

cond.end10:                                       ; preds = %cond.end, %cond.true5
  %cond11 = phi i32 [ %mul, %cond.true5 ], [ %cond, %cond.end ]
  br label %cond.end12

cond.end12:                                       ; preds = %cond.end10, %cond.true2
  %cond13 = phi i32 [ %sub, %cond.true2 ], [ %cond11, %cond.end10 ]
  br label %cond.end14

cond.end14:                                       ; preds = %cond.end12, %cond.true
  %cond15 = phi i32 [ %add, %cond.true ], [ %cond13, %cond.end12 ]
  store i32 %cond15, i32* %ret, align 4
  %13 = load i32, i32* %ret, align 4
  ret i32 %13
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 9bd55c2f21c13001f663d8357eac3083e3e21005)"}
