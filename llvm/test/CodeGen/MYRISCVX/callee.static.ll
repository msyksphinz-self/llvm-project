; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @func_callee(i32 signext %c, i32 signext %a, i32 signext %b) #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: func_callee:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -12
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x12, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beqz	x10, $BB0_2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %cond.true
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %cond.false
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %cond.end
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 12
; MYRVX32I_STATIC_MEDLOW-NEXT: 	ret

; MYRVX64I_STATIC_MEDLOW-LABEL: func_callee:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -12
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x12, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lwu	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beqz	x10, $BB0_2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %cond.true
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %cond.false
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %cond.end
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sext.w	x10, x10
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 12
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ret


entry:
  %c.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %c.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  %1 = load i32, i32* %a.addr, align 4
  br label %cond.end

cond.false:                                       ; preds = %entry
  %2 = load i32, i32* %b.addr, align 4
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %1, %cond.true ], [ %2, %cond.false ]
  ret i32 %cond
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 95852c6428ed8c4521fe112f521ad201ccd17095)"}
