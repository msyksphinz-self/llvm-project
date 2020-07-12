; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

define dso_local signext i32 @func_lot_arguments_O0(i32 signext %val1, i32 signext %val2, i32 signext %val3, i32 signext %val4, i32 signext %val5, i32 signext %val6, i32 signext %val7, i32 signext %val8, i32 %val9, i32 %val10) #0 {

;MYRVX32I-LABEL:func_lot_arguments_O0:
;MYRVX32I:             # %bb.0:                                # %entry
;MYRVX32I-NEXT:        addi    x2, x2, -32
;MYRVX32I-NEXT:        lw      x5, 36(x2)
;MYRVX32I-NEXT:        lw      x5, 32(x2)
;MYRVX32I-NEXT:        sw      x10, 28(x2)
;MYRVX32I-NEXT:        sw      x11, 24(x2)
;MYRVX32I-NEXT:        sw      x12, 20(x2)
;MYRVX32I-NEXT:        sw      x13, 16(x2)
;MYRVX32I-NEXT:        sw      x14, 12(x2)
;MYRVX32I-NEXT:        sw      x15, 8(x2)
;MYRVX32I-NEXT:        sw      x16, 4(x2)
;MYRVX32I-NEXT:        sw      x17, 0(x2)
;MYRVX32I-NEXT:        lw      x10, 28(x2)
;MYRVX32I-NEXT:        lw      x11, 24(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 20(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 16(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 12(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 8(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 4(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 0(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 32(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        lw      x11, 36(x2)
;MYRVX32I-NEXT:        add     x10, x10, x11
;MYRVX32I-NEXT:        addi    x2, x2, 32
;MYRVX32I-NEXT:        ret


;MYRVX64I-LABEL:func_lot_arguments_O0:
;MYRVX64I:             # %bb.0:                                # %entry
;MYRVX64I-NEXT:        addi    x2, x2, -40
;MYRVX64I-NEXT:        ld      x5, 48(x2)
;MYRVX64I-NEXT:        ld      x6, 40(x2)
;MYRVX64I-NEXT:        sw      x10, 36(x2)
;MYRVX64I-NEXT:        sw      x11, 32(x2)
;MYRVX64I-NEXT:        sw      x12, 28(x2)
;MYRVX64I-NEXT:        sw      x13, 24(x2)
;MYRVX64I-NEXT:        sw      x14, 20(x2)
;MYRVX64I-NEXT:        sw      x15, 16(x2)
;MYRVX64I-NEXT:        sw      x16, 12(x2)
;MYRVX64I-NEXT:        sw      x17, 8(x2)
;MYRVX64I-NEXT:        sw      x6, 4(x2)
;MYRVX64I-NEXT:        sw      x5, 0(x2)
;MYRVX64I-NEXT:        lw      x10, 36(x2)
;MYRVX64I-NEXT:        lw      x11, 32(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 28(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 24(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 20(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 16(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 12(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 8(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 4(x2)
;MYRVX64I-NEXT:        add     x10, x10, x11
;MYRVX64I-NEXT:        lw      x11, 0(x2)
;MYRVX64I-NEXT:        addw    x10, x10, x11
;MYRVX64I-NEXT:        addi    x2, x2, 40
;MYRVX64I-NEXT:        ret

entry:
  %val1.addr = alloca i32, align 4
  %val2.addr = alloca i32, align 4
  %val3.addr = alloca i32, align 4
  %val4.addr = alloca i32, align 4
  %val5.addr = alloca i32, align 4
  %val6.addr = alloca i32, align 4
  %val7.addr = alloca i32, align 4
  %val8.addr = alloca i32, align 4
  %val9.addr = alloca i32, align 4
  %val10.addr = alloca i32, align 4
  store i32 %val1, i32* %val1.addr, align 4
  store i32 %val2, i32* %val2.addr, align 4
  store i32 %val3, i32* %val3.addr, align 4
  store i32 %val4, i32* %val4.addr, align 4
  store i32 %val5, i32* %val5.addr, align 4
  store i32 %val6, i32* %val6.addr, align 4
  store i32 %val7, i32* %val7.addr, align 4
  store i32 %val8, i32* %val8.addr, align 4
  store i32 %val9, i32* %val9.addr, align 4
  store i32 %val10, i32* %val10.addr, align 4
  %0 = load i32, i32* %val1.addr, align 4
  %1 = load i32, i32* %val2.addr, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* %val3.addr, align 4
  %add1 = add nsw i32 %add, %2
  %3 = load i32, i32* %val4.addr, align 4
  %add2 = add nsw i32 %add1, %3
  %4 = load i32, i32* %val5.addr, align 4
  %add3 = add nsw i32 %add2, %4
  %5 = load i32, i32* %val6.addr, align 4
  %add4 = add nsw i32 %add3, %5
  %6 = load i32, i32* %val7.addr, align 4
  %add5 = add nsw i32 %add4, %6
  %7 = load i32, i32* %val8.addr, align 4
  %add6 = add nsw i32 %add5, %7
  %8 = load i32, i32* %val9.addr, align 4
  %add7 = add nsw i32 %add6, %8
  %9 = load i32, i32* %val10.addr, align 4
  %add8 = add nsw i32 %add7, %9
  ret i32 %add8
}


define dso_local signext i32 @func_lot_arguments_O3(i32 signext %val1, i32 signext %val2, i32 signext %val3, i32 signext %val4, i32 signext %val5, i32 signext %val6, i32 signext %val7, i32 signext %val8, i32 %val9, i32 %val10) local_unnamed_addr #0 {

;MYRVX32I-LABEL:func_lot_arguments_O3:
;MYRVX32I:             # %bb.0:
;MYRVX32I-NEXT:        lw      x5, 4(x2)
;MYRVX32I-NEXT:        lw      x6, 0(x2)
;MYRVX32I-NEXT:        add     x10, x11, x10
;MYRVX32I-NEXT:        add     x10, x10, x12
;MYRVX32I-NEXT:        add     x10, x10, x13
;MYRVX32I-NEXT:        add     x10, x10, x14
;MYRVX32I-NEXT:        add     x10, x10, x15
;MYRVX32I-NEXT:        add     x10, x10, x16
;MYRVX32I-NEXT:        add     x10, x10, x17
;MYRVX32I-NEXT:        add     x10, x10, x6
;MYRVX32I-NEXT:        add     x10, x10, x5
;MYRVX32I-NEXT:        ret

;MYRVX64I-LABEL:func_lot_arguments_O3:
;MYRVX64I:           # %bb.0:                                # %entry
;MYRVX64I-NEXT:        ld      x5, 8(x2)
;MYRVX64I-NEXT:        ld      x6, 0(x2)
;MYRVX64I-NEXT:        add     x10, x11, x10
;MYRVX64I-NEXT:        add     x10, x10, x12
;MYRVX64I-NEXT:        add     x10, x10, x13
;MYRVX64I-NEXT:        add     x10, x10, x14
;MYRVX64I-NEXT:        add     x10, x10, x15
;MYRVX64I-NEXT:        add     x10, x10, x16
;MYRVX64I-NEXT:        add     x10, x10, x17
;MYRVX64I-NEXT:        add     x10, x10, x6
;MYRVX64I-NEXT:        addw    x10, x10, x5
;MYRVX64I-NEXT:        ret


entry:
  %add = add nsw i32 %val2, %val1
  %add1 = add nsw i32 %add, %val3
  %add2 = add nsw i32 %add1, %val4
  %add3 = add nsw i32 %add2, %val5
  %add4 = add nsw i32 %add3, %val6
  %add5 = add nsw i32 %add4, %val7
  %add6 = add nsw i32 %add5, %val8
  %add7 = add nsw i32 %add6, %val9
  %add8 = add nsw i32 %add7, %val10
  ret i32 %add8
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false"
                                            "disable-tail-calls"="false"
                                            "frame-pointer"="all"
                                            "less-precise-fpmad"="false"
                                            "min-legal-vector-width"="0"
                                            "no-infs-fp-math"="false"
                                            "no-jump-tables"="false"
                                            "no-nans-fp-math"="false"
                                            "no-signed-zeros-fp-math"="false"
                                            "no-trapping-math"="false"
                                            "stack-protector-buffer-size"="8"
                                            "target-features"="+a,+c,+m,+relax"
                                            "unsafe-fp-math"="false" "use-soft-float"="false" }
