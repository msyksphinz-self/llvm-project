; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s


define dso_local void @make_frame64() #0 {
; MYRVX32I-LABEL:make_frame64:
; MYRVX32I:      # %bb.0:
; MYRVX32I-NEXT: 	addi    x2, x2, -24
; MYRVX32I-NEXT: 	addi    x10, x2, 8
; MYRVX32I-NEXT: 	ori x10, x10, 4
; MYRVX32I-NEXT: 	addi    x11, x0, 0
; MYRVX32I-NEXT: 	sw  x11, 0(x10)
; MYRVX32I-NEXT: 	addi    x10, x2, 16
; MYRVX32I-NEXT: 	ori x10, x10, 4
; MYRVX32I-NEXT: 	sw  x11, 0(x10)
; MYRVX32I-NEXT: 	addi    x10, x2, 0
; MYRVX32I-NEXT: 	ori x10, x10, 4
; MYRVX32I-NEXT: 	sw  x11, 0(x10)
; MYRVX32I-NEXT: 	sw  x11, 8(x2)
; MYRVX32I-NEXT: 	sw  x11, 16(x2)
; MYRVX32I-NEXT: 	sw  x11, 0(x2)
; MYRVX32I-NEXT: 	addi    x2, x2, 24
; MYRVX32I-NEXT: 	ret

; MYRVX64I-LABEL:make_frame64:
; MYRVX64I:      # %bb.0:
; MYRVX64I-NEXT: addi    x2, x2, -24
; MYRVX64I-NEXT: addi    x10, x0, 0
; MYRVX64I-NEXT: sd  x10, 8(x2)
; MYRVX64I-NEXT: sd  x10, 16(x2)
; MYRVX64I-NEXT: sd  x10, 0(x2)
; MYRVX64I-NEXT: addi    x2, x2, 24
; MYRVX64I-NEXT: ret

entry:
  %a = alloca i64, align 8
  %b = alloca i64, align 8
  %c = alloca i64, align 8
  store i64 0, i64* %a, align 8
  store i64 0, i64* %b, align 8
  store i64 0, i64* %c, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @make_frame32() #0 {
; MYRVX32I-LABEL:make_frame32:
; MYRVX32I:      # %bb.0:
; MYRVX32I-NEXT: addi    x2, x2, -12
; MYRVX32I-NEXT: addi    x10, x0, 0
; MYRVX32I-NEXT: sw  x10, 4(x2)
; MYRVX32I-NEXT: sw  x10, 8(x2)
; MYRVX32I-NEXT: sw  x10, 0(x2)
; MYRVX32I-NEXT: addi    x2, x2, 12
; MYRVX32I-NEXT: ret

; MYRVX64I-LABEL:make_frame32:
; MYRVX64I:      # %bb.0:
; MYRVX64I-NEXT: addi    x2, x2, -12
; MYRVX64I-NEXT: addi    x10, x0, 0
; MYRVX64I-NEXT: sw  x10, 4(x2)
; MYRVX64I-NEXT: sw  x10, 8(x2)
; MYRVX64I-NEXT: sw  x10, 0(x2)
; MYRVX64I-NEXT: addi    x2, x2, 12
; MYRVX64I-NEXT: ret

entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, i32* %a, align 4
  store i32 0, i32* %b, align 4
  store i32 0, i32* %c, align 4
  ret void
}
