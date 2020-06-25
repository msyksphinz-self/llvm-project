; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @do_while_count(i32 signext %init) #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: do_while_count:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %do.body
; MYRVX32I_STATIC_MEDLOW-NEXT:                                         # =>This Inner Loop Header: Depth=1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x10
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_2
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %do.cond
; MYRVX32I_STATIC_MEDLOW-NEXT:                                         #   in Loop: Header=BB0_1 Depth=1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1000
; MYRVX32I_STATIC_MEDLOW-NEXT: 	slt	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	bne	x10, x0, $BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %do.end
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 4
; MYRVX32I_STATIC_MEDLOW-NEXT: 	ret

; MYRVX64I_STATIC_MEDLOW-LABEL: do_while_count:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %do.body
; MYRVX64I_STATIC_MEDLOW-NEXT:                                         # =>This Inner Loop Header: Depth=1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	mul	x10, x10, x10
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_2
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %do.cond
; MYRVX64I_STATIC_MEDLOW-NEXT:                                         #   in Loop: Header=BB0_1 Depth=1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x11, x0, 1000
; MYRVX64I_STATIC_MEDLOW-NEXT: 	slt	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	bne	x10, x0, $BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %do.end
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 4
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ret

entry:
  %init.addr = alloca i32, align 4
  store i32 %init, i32* %init.addr, align 4
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %0 = load i32, i32* %init.addr, align 4
  %1 = load i32, i32* %init.addr, align 4
  %mul = mul nsw i32 %0, %1
  store i32 %mul, i32* %init.addr, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %2 = load i32, i32* %init.addr, align 4
  %cmp = icmp slt i32 %2, 1000
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  %3 = load i32, i32* %init.addr, align 4
  ret i32 %3
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 9bd55c2f21c13001f663d8357eac3083e3e21005)"}
