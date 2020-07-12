; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

define dso_local signext i32 @simm_const() #0 {
; MYRVX32I-LABEL:simm_const:
; MYRVX32I:         # %bb.0:
; MYRVX32I-NEXT:	addi	x10, x0, 291
; MYRVX32I-NEXT:	ret

; MYRVX64I-LABEL:simm_const:
; MYRVX64I:         # %bb.0:
; MYRVX64I-NEXT:	addi	x10, x0, 291
; MYRVX64I-NEXT:	ret

entry:
  ret i32 291
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @lui_const() #0 {
; MYRVX32I-LABEL:lui_const:
; MYRVX32I:         # %bb.0:
; MYRVX32I-NEXT:	lui	x10, 74565
; MYRVX32I-NEXT:	ret

; MYRVX64I-LABEL:lui_const:
; MYRVX64I:         # %bb.0:
; MYRVX64I-NEXT:	lui	x10, 74565
; MYRVX64I-NEXT:	ret

entry:
  ret i32 305418240
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @word_const() #0 {
; MYRVX32I-LABEL:word_const:
; MYRVX32I:         # %bb.0:
; MYRVX32I-NEXT:	lui x10, 74565
; MYRVX32I-NEXT:	addi    x10, x10, 1656
; MYRVX32I-NEXT:	ret

; MYRVX64I-LABEL:word_const:
; MYRVX64I:         # %bb.0:
; MYRVX64I-NEXT:	lui x10, 74565
; MYRVX64I-NEXT:	addiw    x10, x10, 1656
; MYRVX64I-NEXT:	ret

entry:
  ret i32 305419896
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @word_const2() #0 {
; MYRVX32I-LABEL:word_const2:
; MYRVX32I:         # %bb.0:
; MYRVX32I-NEXT:	lui x10, 74566
; MYRVX32I-NEXT:	addi    x10, x10, -1348
; MYRVX32I-NEXT:	ret

; MYRVX64I-LABEL:word_const2:
; MYRVX64I:         # %bb.0:
; MYRVX64I-NEXT:	lui x10, 74566
; MYRVX64I-NEXT:	addiw    x10, x10, -1348
; MYRVX64I-NEXT:	ret

entry:
  ret i32 305420988
}
