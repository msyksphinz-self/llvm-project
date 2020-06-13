; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

define dso_local signext i32 @add_int_O0(i32 signext %arg1, i32 signext %arg2) #0 {

; MYRVX32I-LABEL:add_int_O0:
; MYRVX32I:         # %bb.0:
; MYRVX32I-NEXT:addi    x2, x2, -16
; MYRVX32I-NEXT:addi    x12, zero, 10
; MYRVX32I-NEXT:sw      x12, 4(x2)
; MYRVX32I-NEXT:addi    x12, zero, 20
; MYRVX32I-NEXT:sw      x12, 0(x2)
; MYRVX32I-NEXT:sw      x11, 8(x2)
; MYRVX32I-NEXT:sw      x10, 12(x2)
; MYRVX32I-NEXT:add     x10, x10, x11
; MYRVX32I-NEXT:addi    x10, x10, 30
; MYRVX32I-NEXT:addi    x2, x2, 16
; MYRVX32I-NEXT:ret

; MYRVX64I-LABEL:add_int_O0:
; MYRVX64I:             # %bb.0:                                # %entry
; MYRVX64I-NEXT:        addi    x2, x2, -16
; MYRVX64I-NEXT:        addi    x12, zero, 10
; MYRVX64I-NEXT:        sw      x12, 4(x2)
; MYRVX64I-NEXT:        addi    x12, zero, 20
; MYRVX64I-NEXT:        sw      x12, 0(x2)
; MYRVX64I-NEXT:        sw      x11, 8(x2)
; MYRVX64I-NEXT:        sw      x10, 12(x2)
; MYRVX64I-NEXT:        add     x10, x10, x11
; MYRVX64I-NEXT:        addiw   x10, x10, 30
; MYRVX64I-NEXT:        addi    x2, x2, 16
; MYRVX64I-NEXT:        ret


entry:
  %arg1.addr = alloca i32, align 4
  %arg2.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %arg1, i32* %arg1.addr, align 4
  store i32 %arg2, i32* %arg2.addr, align 4
  store i32 10, i32* %a, align 4
  store i32 20, i32* %b, align 4
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* %arg1.addr, align 4
  %add1 = add nsw i32 %add, %2
  %3 = load i32, i32* %arg2.addr, align 4
  %add2 = add nsw i32 %add1, %3
  ret i32 %add2
}


define dso_local signext i32 @add_int_O3(i32 signext %arg1, i32 signext %arg2) local_unnamed_addr #0 {

; MYRV32I-LABEL:add_int_O3:
; MYRV32I:      # %bb.0:                                # %entry
; MYRV32I-NEXT:	add	x10, x10, x11
; MYRV32I-NEXT:	addi	x10, x10, 30
; MYRV32I-NEXT:	ret

; MYRV64I-LABEL:add_int_O3:
; MYRV64I:      # %bb.0:                                # %entry
; MYRV64I-NEXT:        add     x10, x10, x11
; MYRV64I-NEXT:        addiw   x10, x10, 30
; MYRV64I-NEXT:        ret

entry:
  %add1 = add nsw i32 %arg1, 30
  %add2 = add nsw i32 %add1, %arg2
  ret i32 %add2
}
