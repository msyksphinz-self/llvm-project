; RUN: llc --march=myriscvx32 --relocation-model=pic --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_PIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=pic --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_PIC_MEDLOW %s
; RUN: llc --march=myriscvx32 --relocation-model=pic --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_PIC_MEDANY %s
; RUN: llc --march=myriscvx64 --relocation-model=pic --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_PIC_MEDANY %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

@__const.int_array.array = private unnamed_addr constant [4 x i32] [i32 100, i32 200, i32 300, i32 400], align 4

define signext i32 @int_array() #0 {

; MYRVX32I_PIC_MEDLOW-LABEL:int_array:
; MYRVX32I_PIC_MEDLOW:       # %bb.0:
; MYRVX32I_PIC_MEDLOW-NEXT:	addi	x2, x2, -24
; MYRVX32I_PIC_MEDLOW-NEXT:	la	x10, $__const.int_array.array
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x11, 12(x10)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x11, 20(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x11, 8(x10)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x11, 16(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x11, 4(x10)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x11, 12(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x10, 0(x10)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x10, 8(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x10, 8(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x10, 4(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	lw	x10, 12(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x10, 0(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	addi	x10, x0, 301
; MYRVX32I_PIC_MEDLOW-NEXT:	sw	x10, 16(x2)
; MYRVX32I_PIC_MEDLOW-NEXT:	mv	x10, x0
; MYRVX32I_PIC_MEDLOW-NEXT:	addi	x2, x2, 24
; MYRVX32I_PIC_MEDLOW-NEXT:	ret
; MYRVX32I_PIC_MEDLOW-NEXT:$func_end0:
; MYRVX32I_PIC_MEDLOW-NEXT:	.size	int_array, ($func_end0)-int_array
; MYRVX32I_PIC_MEDLOW-NEXT:                                        # -- End function
; MYRVX32I_PIC_MEDLOW-NEXT:	.type	$__const.int_array.array,@object # @__const.int_array.array
; MYRVX32I_PIC_MEDLOW-NEXT:	.section	.rodata.cst16,"aM",@progbits,16
; MYRVX32I_PIC_MEDLOW-NEXT:	.p2align	2
; MYRVX32I_PIC_MEDLOW-NEXT:$__const.int_array.array:
; MYRVX32I_PIC_MEDLOW-NEXT:	.4byte	100                     # 0x64
; MYRVX32I_PIC_MEDLOW-NEXT:	.4byte	200                     # 0xc8
; MYRVX32I_PIC_MEDLOW-NEXT:	.4byte	300                     # 0x12c
; MYRVX32I_PIC_MEDLOW-NEXT:	.4byte	400                     # 0x190
; MYRVX32I_PIC_MEDLOW-NEXT:	.size	$__const.int_array.array, 16

; MYRVX32I_PIC_MEDANY-LABEL:int_array:
; MYRVX32I_PIC_MEDANY:       # %bb.0:
; MYRVX32I_PIC_MEDANY-NEXT:	addi	x2, x2, -24
; MYRVX32I_PIC_MEDANY-NEXT:	la	x10, $__const.int_array.array
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x11, 12(x10)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x11, 20(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x11, 8(x10)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x11, 16(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x11, 4(x10)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x11, 12(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x10, 0(x10)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x10, 8(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x10, 8(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x10, 4(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	lw	x10, 12(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x10, 0(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	addi	x10, x0, 301
; MYRVX32I_PIC_MEDANY-NEXT:	sw	x10, 16(x2)
; MYRVX32I_PIC_MEDANY-NEXT:	mv	x10, x0
; MYRVX32I_PIC_MEDANY-NEXT:	addi	x2, x2, 24
; MYRVX32I_PIC_MEDANY-NEXT:	ret
; MYRVX32I_PIC_MEDANY-NEXT:$func_end0:
; MYRVX32I_PIC_MEDANY-NEXT:	.size	int_array, ($func_end0)-int_array
; MYRVX32I_PIC_MEDANY-NEXT:                                        # -- End function
; MYRVX32I_PIC_MEDANY-NEXT:	.type	$__const.int_array.array,@object # @__const.int_array.array
; MYRVX32I_PIC_MEDANY-NEXT:	.section	.rodata.cst16,"aM",@progbits,16
; MYRVX32I_PIC_MEDANY-NEXT:	.p2align	2
; MYRVX32I_PIC_MEDANY-NEXT:$__const.int_array.array:
; MYRVX32I_PIC_MEDANY-NEXT:	.4byte	100                     # 0x64
; MYRVX32I_PIC_MEDANY-NEXT:	.4byte	200                     # 0xc8
; MYRVX32I_PIC_MEDANY-NEXT:	.4byte	300                     # 0x12c
; MYRVX32I_PIC_MEDANY-NEXT:	.4byte	400                     # 0x190
; MYRVX32I_PIC_MEDANY-NEXT:	.size	$__const.int_array.array, 16


; MYRVX64I_PIC_MEDLOW-LABEL:int_array:
; MYRVX64I_PIC_MEDLOW:	    # %bb.0:
; MYRVX64I_PIC_MEDLOW-NEXT:	addi    x2, x2, -24
; MYRVX64I_PIC_MEDLOW-NEXT:	la      x10, $__const.int_array.array
; MYRVX64I_PIC_MEDLOW-NEXT:	lwu     x11, 8(x10)
; MYRVX64I_PIC_MEDLOW-NEXT:	lwu     x12, 12(x10)
; MYRVX64I_PIC_MEDLOW-NEXT:	slli    x12, x12, 32
; MYRVX64I_PIC_MEDLOW-NEXT:	or      x11, x12, x11
; MYRVX64I_PIC_MEDLOW-NEXT:	sd      x11, 16(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	lwu     x11, 0(x10)
; MYRVX64I_PIC_MEDLOW-NEXT:	lwu     x10, 4(x10)
; MYRVX64I_PIC_MEDLOW-NEXT:	slli    x10, x10, 32
; MYRVX64I_PIC_MEDLOW-NEXT:	or      x10, x10, x11
; MYRVX64I_PIC_MEDLOW-NEXT:	sd      x10, 8(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	lw      x10, 8(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	sw      x10, 4(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	addi    x10, x2, 8
; MYRVX64I_PIC_MEDLOW-NEXT:	ori     x10, x10, 4
; MYRVX64I_PIC_MEDLOW-NEXT:	lw      x10, 0(x10)
; MYRVX64I_PIC_MEDLOW-NEXT:	sw      x10, 0(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	addi    x10, x0, 301
; MYRVX64I_PIC_MEDLOW-NEXT:	sw      x10, 16(x2)
; MYRVX64I_PIC_MEDLOW-NEXT:	mv      x10, x0
; MYRVX64I_PIC_MEDLOW-NEXT:	addi    x2, x2, 24
; MYRVX64I_PIC_MEDLOW-NEXT:	ret
; MYRVX64I_PIC_MEDLOW-NEXT:$func_end0:
; MYRVX64I_PIC_MEDLOW-NEXT:	.size	int_array, ($func_end0)-int_array
; MYRVX64I_PIC_MEDLOW-NEXT:                                        # -- End function
; MYRVX64I_PIC_MEDLOW-NEXT:	.type	$__const.int_array.array,@object # @__const.int_array.array
; MYRVX64I_PIC_MEDLOW-NEXT:	.section	.rodata.cst16,"aM",@progbits,16
; MYRVX64I_PIC_MEDLOW-NEXT:	.p2align	2
; MYRVX64I_PIC_MEDLOW-NEXT:$__const.int_array.array:
; MYRVX64I_PIC_MEDLOW-NEXT:	.4byte	100                     # 0x64
; MYRVX64I_PIC_MEDLOW-NEXT:	.4byte	200                     # 0xc8
; MYRVX64I_PIC_MEDLOW-NEXT:	.4byte	300                     # 0x12c
; MYRVX64I_PIC_MEDLOW-NEXT:	.4byte	400                     # 0x190
; MYRVX64I_PIC_MEDLOW-NEXT:	.size	$__const.int_array.array, 16


; MYRVX64I_PIC_MEDANY-LABEL:int_array:
; MYRVX64I_PIC_MEDANY:      # %bb.0:
; MYRVX64I_PIC_MEDANY-NEXT: addi    x2, x2, -24
; MYRVX64I_PIC_MEDANY-NEXT: la      x10, $__const.int_array.array
; MYRVX64I_PIC_MEDANY-NEXT: lwu     x11, 8(x10)
; MYRVX64I_PIC_MEDANY-NEXT: lwu     x12, 12(x10)
; MYRVX64I_PIC_MEDANY-NEXT: slli    x12, x12, 32
; MYRVX64I_PIC_MEDANY-NEXT: or      x11, x12, x11
; MYRVX64I_PIC_MEDANY-NEXT: sd      x11, 16(x2)
; MYRVX64I_PIC_MEDANY-NEXT: lwu     x11, 0(x10)
; MYRVX64I_PIC_MEDANY-NEXT: lwu     x10, 4(x10)
; MYRVX64I_PIC_MEDANY-NEXT: slli    x10, x10, 32
; MYRVX64I_PIC_MEDANY-NEXT: or      x10, x10, x11
; MYRVX64I_PIC_MEDANY-NEXT: sd      x10, 8(x2)
; MYRVX64I_PIC_MEDANY-NEXT: lw      x10, 8(x2)
; MYRVX64I_PIC_MEDANY-NEXT: sw      x10, 4(x2)
; MYRVX64I_PIC_MEDANY-NEXT: addi    x10, x2, 8
; MYRVX64I_PIC_MEDANY-NEXT: ori     x10, x10, 4
; MYRVX64I_PIC_MEDANY-NEXT: lw      x10, 0(x10)
; MYRVX64I_PIC_MEDANY-NEXT: sw      x10, 0(x2)
; MYRVX64I_PIC_MEDANY-NEXT: addi    x10, x0, 301
; MYRVX64I_PIC_MEDANY-NEXT: sw      x10, 16(x2)
; MYRVX64I_PIC_MEDANY-NEXT: mv      x10, x0
; MYRVX64I_PIC_MEDANY-NEXT: addi    x2, x2, 24
; MYRVX64I_PIC_MEDANY-NEXT: ret
; MYRVX64I_PIC_MEDANY-NEXT:$func_end0:
; MYRVX64I_PIC_MEDANY-NEXT:	.size	int_array, ($func_end0)-int_array
; MYRVX64I_PIC_MEDANY-NEXT:                                        # -- End function
; MYRVX64I_PIC_MEDANY-NEXT:	.type	$__const.int_array.array,@object # @__const.int_array.array
; MYRVX64I_PIC_MEDANY-NEXT:	.section	.rodata.cst16,"aM",@progbits,16
; MYRVX64I_PIC_MEDANY-NEXT:	.p2align	2
; MYRVX64I_PIC_MEDANY-NEXT:$__const.int_array.array:
; MYRVX64I_PIC_MEDANY-NEXT:	.4byte	100                     # 0x64
; MYRVX64I_PIC_MEDANY-NEXT:	.4byte	200                     # 0xc8
; MYRVX64I_PIC_MEDANY-NEXT:	.4byte	300                     # 0x12c
; MYRVX64I_PIC_MEDANY-NEXT:	.4byte	400                     # 0x190
; MYRVX64I_PIC_MEDANY-NEXT:	.size	$__const.int_array.array, 16


entry:
  %array = alloca [4 x i32], align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %0 = bitcast [4 x i32]* %array to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast ([4 x i32]* @__const.int_array.array to i8*), i64 16, i1 false)
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32]* %array, i64 0, i64 0
  %1 = load i32, i32* %arrayidx, align 4
  store i32 %1, i32* %a, align 4
  %arrayidx1 = getelementptr inbounds [4 x i32], [4 x i32]* %array, i64 0, i64 1
  %2 = load i32, i32* %arrayidx1, align 4
  store i32 %2, i32* %b, align 4
  %arrayidx2 = getelementptr inbounds [4 x i32], [4 x i32]* %array, i64 0, i64 2
  store i32 301, i32* %arrayidx2, align 4
  ret i32 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
