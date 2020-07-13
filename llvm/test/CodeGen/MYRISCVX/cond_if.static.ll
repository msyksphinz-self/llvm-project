; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @if_ctrl(i32 signext %a, i32 signext %b) #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: if_ctrl:
; MYRVX32I_STATIC_MEDLOW:        # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -12
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	slt	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT: 	beqz	x10, $BB0_2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %if.then
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x10, x0, 1
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %if.else
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x10, x0, 2
; MYRVX32I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX32I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %if.end
; MYRVX32I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 12
; MYRVX32I_STATIC_MEDLOW-NEXT: 	ret

; MYRVX64I_STATIC_MEDLOW-LABEL: if_ctrl:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -12
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	slt	x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT: 	beqz	x10, $BB0_2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_1
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_1:                                 # %if.then
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x0, 1
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_2:                                 # %if.else
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x0, 2
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	j	$BB0_3
; MYRVX64I_STATIC_MEDLOW-NEXT: $BB0_3:                                 # %if.end
; MYRVX64I_STATIC_MEDLOW-NEXT: 	lw	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 12
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ret

entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %ret = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %1 = load i32, i32* %b.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, i32* %ret, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32* %ret, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* %ret, align 4
  ret i32 %2
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 9bd55c2f21c13001f663d8357eac3083e3e21005)"}
