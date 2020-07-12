; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s


; Function Attrs: noinline nounwind optnone
define dso_local void @load_memory_test() #0 {
; MYRVX32I-LABEL:load_memory_test:
; MYRVX32I:     # %bb.0:
; MYRVX32I-NEXT:     lui x10, 1
; MYRVX32I-NEXT:     addi    x11, x10, 568
; MYRVX32I-NEXT:     lw  x12, 0(x11)
; MYRVX32I-NEXT:     addi   x10, x10, 564
; MYRVX32I-NEXT:     lw  x12, 0(x10)
; MYRVX32I-NEXT:     lw  x12, 0(x10)
; MYRVX32I-NEXT:     lh  x12, 0(x10)
; MYRVX32I-NEXT:     lb  x12, 0(x10)
; MYRVX32I-NEXT:     lw  x11, 0(x11)
; MYRVX32I-NEXT:     lw  x11, 0(x10)
; MYRVX32I-NEXT:     lw  x11, 0(x10)
; MYRVX32I-NEXT:     lh  x11, 0(x10)
; MYRVX32I-NEXT:     lb  x10, 0(x10)
; MYRVX32I-NEXT:     ret

; MYRVX64I-LABEL:load_memory_test:
; MYRVX64I:          # %bb.0:
; MYRVX64I-NEXT:     lui x10, 1
; MYRVX64I-NEXT:     addiw   x10, x10, 564
; MYRVX64I-NEXT:     ld  x11, 0(x10)
; MYRVX64I-NEXT:     lw  x11, 0(x10)
; MYRVX64I-NEXT:     lh  x11, 0(x10)
; MYRVX64I-NEXT:     lb  x11, 0(x10)
; MYRVX64I-NEXT:     ld  x11, 0(x10)
; MYRVX64I-NEXT:     lw  x11, 0(x10)
; MYRVX64I-NEXT:     lh  x11, 0(x10)
; MYRVX64I-NEXT:     lb  x10, 0(x10)
; MYRVX64I-NEXT:     ret

entry:
  %0 = load volatile i64, i64* inttoptr (i64 4660 to i64*), align 8
  %1 = load volatile i32, i32* inttoptr (i64 4660 to i32*), align 4
  %2 = load volatile i16, i16* inttoptr (i64 4660 to i16*), align 2
  %3 = load volatile i8, i8* inttoptr (i64 4660 to i8*), align 1
  %4 = load volatile i64, i64* inttoptr (i64 4660 to i64*), align 8
  %5 = load volatile i32, i32* inttoptr (i64 4660 to i32*), align 4
  %6 = load volatile i16, i16* inttoptr (i64 4660 to i16*), align 2
  %7 = load volatile i8, i8* inttoptr (i64 4660 to i8*), align 1
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @store_memory_test() #0 {
; MYRVX32I-LABEL:store_memory_test:
; MYRVX32I:     # %bb.0:
; MYRVX32I-NEXT:    lui x10, 1
; MYRVX32I-NEXT:	addi    x11, x10, 568
; MYRVX32I-NEXT:	addi    x12, x0, 0
; MYRVX32I-NEXT:	sw  x12, 0(x11)
; MYRVX32I-NEXT:	addi    x10, x10, 564
; MYRVX32I-NEXT:	addi    x13, x0, 1
; MYRVX32I-NEXT:	sw  x13, 0(x10)
; MYRVX32I-NEXT:	sw  x13, 0(x10)
; MYRVX32I-NEXT:	sh  x13, 0(x10)
; MYRVX32I-NEXT:	sb  x13, 0(x10)
; MYRVX32I-NEXT:	sw  x12, 0(x11)
; MYRVX32I-NEXT:	sw  x13, 0(x10)
; MYRVX32I-NEXT:	sw  x13, 0(x10)
; MYRVX32I-NEXT:	sh  x13, 0(x10)
; MYRVX32I-NEXT:	sb  x13, 0(x10)
; MYRVX32I-NEXT:	ret

; MYRVX64I-LABEL:store_memory_test:
; MYRVX64I:         # %bb.0:
; MYRVX64I-NEXT:    lui x10, 1
; MYRVX64I-NEXT:    addiw   x10, x10, 564
; MYRVX64I-NEXT:    addi    x11, x0, 1
; MYRVX64I-NEXT:    sd  x11, 0(x10)
; MYRVX64I-NEXT:    sw  x11, 0(x10)
; MYRVX64I-NEXT:    sh  x11, 0(x10)
; MYRVX64I-NEXT:    sb  x11, 0(x10)
; MYRVX64I-NEXT:    sd  x11, 0(x10)
; MYRVX64I-NEXT:    sw  x11, 0(x10)
; MYRVX64I-NEXT:    sh  x11, 0(x10)
; MYRVX64I-NEXT:    sb  x11, 0(x10)
; MYRVX64I-NEXT:    ret

entry:
  store volatile i64 1, i64* inttoptr (i64 4660 to i64*), align 8
  store volatile i32 1, i32* inttoptr (i64 4660 to i32*), align 4
  store volatile i16 1, i16* inttoptr (i64 4660 to i16*), align 2
  store volatile i8 1, i8* inttoptr (i64 4660 to i8*), align 1
  store volatile i64 1, i64* inttoptr (i64 4660 to i64*), align 8
  store volatile i32 1, i32* inttoptr (i64 4660 to i32*), align 4
  store volatile i16 1, i16* inttoptr (i64 4660 to i16*), align 2
  store volatile i8 1, i8* inttoptr (i64 4660 to i8*), align 1
  ret void
}
