; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

define dso_local { i32, i32 } @constants2() {
; MYRVX32I-LABEL:constants2:
; MYRVX32I:     # %bb.0:                                # %entry
; MYRVX32I-NEXT:        addi    x10, x0, 4
; MYRVX32I-NEXT:        addi    x11, x0, 5
; MYRVX32I-NEXT:        ret

; MYRVX64I-LABEL:constants2:
; MYRVX64I:     # %bb.0:                                # %entry
; MYRVX64I-NEXT:        addi    x10, x0, 4
; MYRVX64I-NEXT:        addi    x11, x0, 5
; MYRVX64I-NEXT:        ret

entry:
  ret { i32, i32 } { i32 4, i32 5 }
}
