; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDANY %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDANY %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @local_pointer() #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:	local_pointer:
; MYRVX32I_STATIC_MEDLOW:	    # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:		addi	x2, x2, -8
; MYRVX32I_STATIC_MEDLOW-NEXT:		addi	x10, x0, 3
; MYRVX32I_STATIC_MEDLOW-NEXT:		sw	x10, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:		addi	x10, x2, 4
; MYRVX32I_STATIC_MEDLOW-NEXT:		sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:		lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:		lw	x10, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:		addi	x2, x2, 8
; MYRVX32I_STATIC_MEDLOW-NEXT:		ret

; MYRVX32I_STATIC_MEDANY-LABEL:	local_pointer:
; MYRVX32I_STATIC_MEDANY:	    # %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:		addi	x2, x2, -8
; MYRVX32I_STATIC_MEDANY-NEXT:		addi	x10, x0, 3
; MYRVX32I_STATIC_MEDANY-NEXT:		sw	x10, 4(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:		addi	x10, x2, 4
; MYRVX32I_STATIC_MEDANY-NEXT:		sw	x10, 0(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:		lw	x10, 0(x2)
; MYRVX32I_STATIC_MEDANY-NEXT:		lw	x10, 0(x10)
; MYRVX32I_STATIC_MEDANY-NEXT:		addi	x2, x2, 8
; MYRVX32I_STATIC_MEDANY-NEXT:		ret

; MYRVX64I_STATIC_MEDLOW-LABEL:	local_pointer:
; MYRVX64I_STATIC_MEDLOW:	    # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:		addi	x2, x2, -16
; MYRVX64I_STATIC_MEDLOW-NEXT:		addi	x10, x0, 3
; MYRVX64I_STATIC_MEDLOW-NEXT:		sw	x10, 12(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:		addi	x10, x2, 12
; MYRVX64I_STATIC_MEDLOW-NEXT:		sd	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:		ld	x10, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:		lw	x10, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:		addi	x2, x2, 16
; MYRVX64I_STATIC_MEDLOW-NEXT:		ret

; MYRVX64I_STATIC_MEDANY-LABEL:	local_pointer:
; MYRVX64I_STATIC_MEDANY:	    # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:		addi	x2, x2, -16
; MYRVX64I_STATIC_MEDANY-NEXT:		addi	x10, x0, 3
; MYRVX64I_STATIC_MEDANY-NEXT:		sw	x10, 12(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:		addi	x10, x2, 12
; MYRVX64I_STATIC_MEDANY-NEXT:		sd	x10, 0(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:		ld	x10, 0(x2)
; MYRVX64I_STATIC_MEDANY-NEXT:		lw	x10, 0(x10)
; MYRVX64I_STATIC_MEDANY-NEXT:		addi	x2, x2, 16
; MYRVX64I_STATIC_MEDANY-NEXT:		ret

entry:
  %b = alloca i32, align 4
  %local_ptr_b = alloca i32*, align 8
  store i32 3, i32* %b, align 4
  store i32* %b, i32** %local_ptr_b, align 8
  %0 = load i32*, i32** %local_ptr_b, align 8
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git e947bd789bb551e8ed77e4eda9d0c10a40e2f863)"}
