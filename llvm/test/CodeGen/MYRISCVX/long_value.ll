; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"


; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @long_value() #0 {

;MYRVX32I-LABEL:long_value:
;MYRVX32I:      # %bb.0:
;MYRVX32I-NEXT:	addi	x2, x2, -8
;MYRVX32I-NEXT:	addi	x10, x2, 0
;MYRVX32I-NEXT:	ori	x10, x10, 4
;MYRVX32I-NEXT:	lui	x11, 4660
;MYRVX32I-NEXT:	addi	x11, x11, 1383
;MYRVX32I-NEXT:	sw	x11, 0(x10)
;MYRVX32I-NEXT:	lui	x10, 563901
;MYRVX32I-NEXT:	addi	x10, x10, -529
;MYRVX32I-NEXT:	sw	x10, 0(x2)
;MYRVX32I-NEXT:	addi	x10, x0, 0
;MYRVX32I-NEXT:	addi	x2, x2, 8
;MYRVX32I-NEXT:	ret


;MYRVX64I-LABEL:long_value:
;MYRVX64I:      # %bb.0:
;MYRVX64I-NEXT:	addi	x2, x2, -8
;MYRVX64I-NEXT:	lui	x10, 146
;MYRVX64I-NEXT:	addiw	x10, x10, -1493
;MYRVX64I-NEXT:	slli	x10, x10, 12
;MYRVX64I-NEXT:	addi	x10, x10, 965
;MYRVX64I-NEXT:	slli	x10, x10, 13
;MYRVX64I-NEXT:	addi	x10, x10, -1347
;MYRVX64I-NEXT:	slli	x10, x10, 12
;MYRVX64I-NEXT:	addi	x10, x10, -529
;MYRVX64I-NEXT:	sd	x10, 0(x2)
;MYRVX64I-NEXT:	addi	x10, x0, 0
;MYRVX64I-NEXT:	addi	x2, x2, 8
;MYRVX64I-NEXT:	ret

entry:
  %long_const = alloca i64, align 8
  store i64 81985529216486895, i64* %long_const, align 8
  ret i32 0
}


attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
