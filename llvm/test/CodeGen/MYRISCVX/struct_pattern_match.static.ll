; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDANY %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=medium < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDANY %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.color = type { i8, i8, i8 }

; Function Attrs: norecurse nounwind readonly
define dso_local signext i32 @pattern_match(%struct.color* nocapture readonly %c0, %struct.color* nocapture readonly %c1, i32 signext %threshold) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDANY-LABEL:pattern_match:
; MYRVX32I_STATIC_MEDANY:       # %bb.0:
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x12, 1(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x13, 1(x10)
; MYRVX32I_STATIC_MEDANY-NEXT:  sub     x12, x13, x12
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x13, 0(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x14, 0(x10)
; MYRVX32I_STATIC_MEDANY-NEXT:  sub     x13, x14, x13
; MYRVX32I_STATIC_MEDANY-NEXT:  mul     x13, x13, x13
; MYRVX32I_STATIC_MEDANY-NEXT:  mul     x12, x12, x12
; MYRVX32I_STATIC_MEDANY-NEXT:  add     x12, x12, x13
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x11, 2(x11)
; MYRVX32I_STATIC_MEDANY-NEXT:  lbu     x10, 2(x10)
; MYRVX32I_STATIC_MEDANY-NEXT:  sub     x10, x10, x11
; MYRVX32I_STATIC_MEDANY-NEXT:  mul     x10, x10, x10
; MYRVX32I_STATIC_MEDANY-NEXT:  add     x10, x12, x10
; MYRVX32I_STATIC_MEDANY-NEXT:  ret


; MYRVX64I_STATIC_MEDANY-LABEL:pattern_match:
; MYRVX64I_STATIC_MEDANY:       # %bb.0:
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x12, 1(x11)
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x13, 1(x10)
; MYRVX64I_STATIC_MEDANY-NEXT:	sub	x12, x13, x12
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x13, 0(x11)
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x14, 0(x10)
; MYRVX64I_STATIC_MEDANY-NEXT:	sub	x13, x14, x13
; MYRVX64I_STATIC_MEDANY-NEXT:	mul	x13, x13, x13
; MYRVX64I_STATIC_MEDANY-NEXT:	mul	x12, x12, x12
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x12, x12, x13
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x11, 2(x11)
; MYRVX64I_STATIC_MEDANY-NEXT:	lbu	x10, 2(x10)
; MYRVX64I_STATIC_MEDANY-NEXT:	sub	x10, x10, x11
; MYRVX64I_STATIC_MEDANY-NEXT:	mul	x10, x10, x10
; MYRVX64I_STATIC_MEDANY-NEXT:	add	x10, x12, x10
; MYRVX64I_STATIC_MEDANY-NEXT:	ret

entry:
  %red = getelementptr inbounds %struct.color, %struct.color* %c0, i64 0, i32 0
  %0 = load i8, i8* %red, align 1, !tbaa !3
  %conv = zext i8 %0 to i32
  %red1 = getelementptr inbounds %struct.color, %struct.color* %c1, i64 0, i32 0
  %1 = load i8, i8* %red1, align 1, !tbaa !3
  %conv2 = zext i8 %1 to i32
  %sub = sub nsw i32 %conv, %conv2
  %green = getelementptr inbounds %struct.color, %struct.color* %c0, i64 0, i32 1
  %2 = load i8, i8* %green, align 1, !tbaa !7
  %conv3 = zext i8 %2 to i32
  %green4 = getelementptr inbounds %struct.color, %struct.color* %c1, i64 0, i32 1
  %3 = load i8, i8* %green4, align 1, !tbaa !7
  %conv5 = zext i8 %3 to i32
  %sub6 = sub nsw i32 %conv3, %conv5
  %blue = getelementptr inbounds %struct.color, %struct.color* %c0, i64 0, i32 2
  %4 = load i8, i8* %blue, align 1, !tbaa !8
  %conv7 = zext i8 %4 to i32
  %blue8 = getelementptr inbounds %struct.color, %struct.color* %c1, i64 0, i32 2
  %5 = load i8, i8* %blue8, align 1, !tbaa !8
  %conv9 = zext i8 %5 to i32
  %sub10 = sub nsw i32 %conv7, %conv9
  %mul = mul nsw i32 %sub, %sub
  %mul11 = mul nsw i32 %sub6, %sub6
  %add = add nuw nsw i32 %mul11, %mul
  %mul12 = mul nsw i32 %sub10, %sub10
  %add13 = add nuw nsw i32 %add, %mul12
  ret i32 %add13
}

attributes #0 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+a,+c,+m,+relax" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"lp64"}
!2 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git 4e9d0aea340415167da7ac771f1bdb373d62da80)"}
!3 = !{!4, !5, i64 0}
!4 = !{!"color", !5, i64 0, !5, i64 1, !5, i64 2}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!4, !5, i64 1}
!8 = !{!4, !5, i64 2}
