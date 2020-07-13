; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.Date = type { i16, i8, i8, i8, i8, i8 }

@__const.int_struct.date1 = private unnamed_addr constant %struct.Date { i16 2012, i8 11, i8 25, i8 9, i8 40, i8 15 }, align 2

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @int_struct() #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:int_struct:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	addi	x2, x2, -16
; MYRVX32I_STATIC_MEDLOW-NEXT:	lui	x10, %hi($__const.int_struct.date1)
; MYRVX32I_STATIC_MEDLOW-NEXT:	addi	x10, x10, %lo($__const.int_struct.date1)
; MYRVX32I_STATIC_MEDLOW-NEXT:	lhu	x11, 4(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	lhu	x12, 6(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	slli	x12, x12, 16
; MYRVX32I_STATIC_MEDLOW-NEXT:	or	x11, x12, x11
; MYRVX32I_STATIC_MEDLOW-NEXT:	addi	x12, x2, 8
; MYRVX32I_STATIC_MEDLOW-NEXT:	ori	x13, x12, 4
; MYRVX32I_STATIC_MEDLOW-NEXT:	sw	x11, 0(x13)
; MYRVX32I_STATIC_MEDLOW-NEXT:	lhu	x11, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	lhu	x10, 2(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	slli	x10, x10, 16
; MYRVX32I_STATIC_MEDLOW-NEXT:	or	x10, x10, x11
; MYRVX32I_STATIC_MEDLOW-NEXT:	sw	x10, 8(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	ori	x10, x12, 2
; MYRVX32I_STATIC_MEDLOW-NEXT:	lb	x10, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	sb	x10, 4(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	ori	x10, x12, 6
; MYRVX32I_STATIC_MEDLOW-NEXT:	lb	x10, 0(x10)
; MYRVX32I_STATIC_MEDLOW-NEXT:	sb	x10, 0(x2)
; MYRVX32I_STATIC_MEDLOW-NEXT:	addi	x10, x0, 100
; MYRVX32I_STATIC_MEDLOW-NEXT:	sb	x10, 0(x13)
; MYRVX32I_STATIC_MEDLOW-NEXT:	mv	x10, x0
; MYRVX32I_STATIC_MEDLOW-NEXT:	addi	x2, x2, 16
; MYRVX32I_STATIC_MEDLOW-NEXT:	ret
; MYRVX32I_STATIC_MEDLOW-NEXT:$func_end0:
; MYRVX32I_STATIC_MEDLOW-NEXT:	.size	int_struct, ($func_end0)-int_struct
; MYRVX32I_STATIC_MEDLOW-NEXT:                                        # -- End function
; MYRVX32I_STATIC_MEDLOW-NEXT:	.type	$__const.int_struct.date1,@object # @__const.int_struct.date1
; MYRVX32I_STATIC_MEDLOW-NEXT:	.section	.rodata.cst8,"aM",@progbits,8
; MYRVX32I_STATIC_MEDLOW-NEXT:	.p2align	1
; MYRVX32I_STATIC_MEDLOW-NEXT:$__const.int_struct.date1:
; MYRVX32I_STATIC_MEDLOW-NEXT:	.2byte	2012                    # 0x7dc
; MYRVX32I_STATIC_MEDLOW-NEXT:	.byte	11                      # 0xb
; MYRVX32I_STATIC_MEDLOW-NEXT:	.byte	25                      # 0x19
; MYRVX32I_STATIC_MEDLOW-NEXT:	.byte	9                       # 0x9
; MYRVX32I_STATIC_MEDLOW-NEXT:	.byte	40                      # 0x28
; MYRVX32I_STATIC_MEDLOW-NEXT:	.byte	15                      # 0xf
; MYRVX32I_STATIC_MEDLOW-NEXT:	.space	1
; MYRVX32I_STATIC_MEDLOW-NEXT:	.size	$__const.int_struct.date1, 8


; MYRVX64I_STATIC_MEDLOW-LABEL:int_struct:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT:  addi    x2, x2, -16
; MYRVX64I_STATIC_MEDLOW-NEXT:  lui     x10, %hi($__const.int_struct.date1)
; MYRVX64I_STATIC_MEDLOW-NEXT:  addi    x10, x10, %lo($__const.int_struct.date1)
; MYRVX64I_STATIC_MEDLOW-NEXT:  lhu     x11, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:  lhu     x12, 2(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:  slli    x12, x12, 16
; MYRVX64I_STATIC_MEDLOW-NEXT:  or      x11, x12, x11
; MYRVX64I_STATIC_MEDLOW-NEXT:  lhu     x12, 4(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:  lhu     x10, 6(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:  slli    x10, x10, 16
; MYRVX64I_STATIC_MEDLOW-NEXT:  or      x10, x10, x12
; MYRVX64I_STATIC_MEDLOW-NEXT:  slli    x10, x10, 32
; MYRVX64I_STATIC_MEDLOW-NEXT:  or      x10, x10, x11
; MYRVX64I_STATIC_MEDLOW-NEXT:  sd      x10, 8(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:  addi    x10, x2, 8
; MYRVX64I_STATIC_MEDLOW-NEXT:  ori     x11, x10, 2
; MYRVX64I_STATIC_MEDLOW-NEXT:  lb      x11, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:  sb      x11, 4(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:  ori     x11, x10, 6
; MYRVX64I_STATIC_MEDLOW-NEXT:  lb      x11, 0(x11)
; MYRVX64I_STATIC_MEDLOW-NEXT:  sb      x11, 0(x2)
; MYRVX64I_STATIC_MEDLOW-NEXT:  ori     x10, x10, 4
; MYRVX64I_STATIC_MEDLOW-NEXT:  addi    x11, x0, 100
; MYRVX64I_STATIC_MEDLOW-NEXT:  sb      x11, 0(x10)
; MYRVX64I_STATIC_MEDLOW-NEXT:  mv      x10, x0
; MYRVX64I_STATIC_MEDLOW-NEXT:  addi    x2, x2, 16
; MYRVX64I_STATIC_MEDLOW-NEXT:  ret
; MYRVX64I_STATIC_MEDLOW-NEXT:$func_end0:
; MYRVX64I_STATIC_MEDLOW-NEXT:	.size	int_struct, ($func_end0)-int_struct
; MYRVX64I_STATIC_MEDLOW-NEXT:                                        # -- End function
; MYRVX64I_STATIC_MEDLOW-NEXT:	.type	$__const.int_struct.date1,@object # @__const.int_struct.date1
; MYRVX64I_STATIC_MEDLOW-NEXT:	.section	.rodata.cst8,"aM",@progbits,8
; MYRVX64I_STATIC_MEDLOW-NEXT:	.p2align	1
; MYRVX64I_STATIC_MEDLOW-NEXT:$__const.int_struct.date1:
; MYRVX64I_STATIC_MEDLOW-NEXT:	.2byte	2012                    # 0x7dc
; MYRVX64I_STATIC_MEDLOW-NEXT:	.byte	11                      # 0xb
; MYRVX64I_STATIC_MEDLOW-NEXT:	.byte	25                      # 0x19
; MYRVX64I_STATIC_MEDLOW-NEXT:	.byte	9                       # 0x9
; MYRVX64I_STATIC_MEDLOW-NEXT:	.byte	40                      # 0x28
; MYRVX64I_STATIC_MEDLOW-NEXT:	.byte	15                      # 0xf
; MYRVX64I_STATIC_MEDLOW-NEXT:	.space	1
; MYRVX64I_STATIC_MEDLOW-NEXT:	.size	$__const.int_struct.date1, 8



entry:
  %date1 = alloca %struct.Date, align 2
  %m = alloca i8, align 1
  %s = alloca i8, align 1
  %0 = bitcast %struct.Date* %date1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 %0, i8* align 2 bitcast (%struct.Date* @__const.int_struct.date1 to i8*), i64 8, i1 false)
  %month = getelementptr inbounds %struct.Date, %struct.Date* %date1, i32 0, i32 1
  %1 = load i8, i8* %month, align 2
  store i8 %1, i8* %m, align 1
  %second = getelementptr inbounds %struct.Date, %struct.Date* %date1, i32 0, i32 5
  %2 = load i8, i8* %second, align 2
  store i8 %2, i8* %s, align 1
  %hour = getelementptr inbounds %struct.Date, %struct.Date* %date1, i32 0, i32 3
  store i8 100, i8* %hour, align 2
  ret i32 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git e947bd789bb551e8ed77e4eda9d0c10a40e2f863)"}
