; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

define i32 @zero_return() nounwind {
; MYRVX32I-LABEL: zero_return:
; MYRVX32I:       # %bb.0:
; MYRVX32I:         addi    x10, x0, 0
; MYRVX32I-NEXT:    ret

; MYRVX64I-LABEL: zero_return:
; MYRVX64I:       # %bb.0:
; MYRVX64I:         addi    x10, x0, 0
; MYRVX64I-NEXT:    ret
entry:
  ret i32 0
}
