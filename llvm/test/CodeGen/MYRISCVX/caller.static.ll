; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

@a = common dso_local global i32 0, align 4
@b = common dso_local global i32 0, align 4
@c = common dso_local global i32 0, align 4

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @func_caller() #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL: func_caller:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:         addi    x2, x2, -16
; MYRVX32I_STATIC_MEDLOW-NEXT:         sw      x1, 12(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:         sw      x2, 8(x2)               # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:         lla     x10, a
; MYRVX32I_STATIC_MEDLOW-NEXT:         lw      x10, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:         lla     x11, b
; MYRVX32I_STATIC_MEDLOW-NEXT:         lw      x11, 0(x11)
; MYRVX32I_STATIC_MEDLOW-NEXT:         lla     x12, c
; MYRVX32I_STATIC_MEDLOW-NEXT:         lw      x12, 0(x12)
; MYRVX32I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:         call    func_callee
; MYRVX32I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:         lw      x2, 8(x2)               # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:         lw      x1, 12(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 16
; MYRVX32I_STATIC_MEDLOW-NEXT:         ret


; MYRVX64I_STATIC_MEDLOW-LABEL: func_caller:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:         addi    x2, x2, -16
; MYRVX64I_STATIC_MEDLOW-NEXT:         sd      x1, 8(x2)               # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT:         sd      x2, 0(x2)               # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT:         lla     x10, a
; MYRVX64I_STATIC_MEDLOW-NEXT:         lw      x10, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:         lla     x11, b
; MYRVX64I_STATIC_MEDLOW-NEXT:         lw      x11, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:         lla     x12, c
; MYRVX64I_STATIC_MEDLOW-NEXT:         lw      x12, 0(x12)
; MYRVX64I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 0
; MYRVX64I_STATIC_MEDLOW-NEXT:         call    func_callee
; MYRVX64I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 0
; MYRVX64I_STATIC_MEDLOW-NEXT:         ld      x2, 0(x2)               # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT:         ld      x1, 8(x2)               # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT:         addi    x2, x2, 16
; MYRVX64I_STATIC_MEDLOW-NEXT:         ret

entry:
  %0 = load i32, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %2 = load i32, i32* @c, align 4
  %call = call signext i32 @func_callee(i32 signext %0, i32 signext %1, i32 signext %2)
  ret i32 %call
}

declare dso_local signext i32 @func_callee(i32 signext, i32 signext, i32 signext) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 95852c6428ed8c4521fe112f521ad201ccd17095)"}
