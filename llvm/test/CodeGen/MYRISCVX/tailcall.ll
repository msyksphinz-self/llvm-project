; RUN: llc --march=myriscvx32 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I_STATIC_MEDLOW %s
; RUN: llc --march=myriscvx64 --relocation-model=static --code-model=small < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I_STATIC_MEDLOW %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @tail_call(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:tail_call:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x2, x2, -32
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x1, 28(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x2, 24(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x9, 20(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x18, 16(x2)             # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x19, 12(x2)             # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x20, 8(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x9, x13, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x18, x12, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x19, x11, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x20, x10, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x19, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   add     x19, x10, x20
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x18, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x18, x10, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x9, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   add     x11, x10, x18
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x19, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    tail_call_func
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x20, 8(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x19, 12(x2)             # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x18, 16(x2)             # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x9, 20(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x2, 24(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x1, 28(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x2, x2, 32
; MYRVX32I_STATIC_MEDLOW-NEXT:   ret

; MYRVX64I_STATIC_MEDLOW-LABEL: tail_call:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -48
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x1, 40(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x2, 32(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x9, 24(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x18, 16(x2)             # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x19, 8(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x20, 0(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x9, x13, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x18, x12, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x19, x11, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x20, x10, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x19, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x19, x10, x20
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x18, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x18, x10, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x9, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x11, x10, x18
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x19, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	tail_call_func
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x20, 0(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x19, 8(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x18, 16(x2)             # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x9, 24(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x2, 32(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x1, 40(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 48
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ret

entry:
  %call = call i32 @inc(i32 %a)
  %call1 = call i32 @inc(i32 %b)
  %add = add nsw i32 %call1, %call
  %call2 = call i32 @inc(i32 %c)
  %call3 = call i32 @inc(i32 %d)
  %add4 = add nsw i32 %call3, %call2
  %call5 = call i32 @tail_call_func(i32 %add, i32 %add4)
  ret i32 %call5
}


; Function Attrs: nounwind uwtable
define dso_local i32 @tail_call_tail(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {

; MYRVX32I_STATIC_MEDLOW-LABEL:tail_call_tail:
; MYRVX32I_STATIC_MEDLOW:       # %bb.0:
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x2, x2, -32
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x1, 28(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x2, 24(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x9, 20(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x18, 16(x2)             # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x19, 12(x2)             # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   sw      x20, 8(x2)              # 4-byte Folded Spill
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x9, x13, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x18, x12, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x19, x11, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x20, x10, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x19, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   add     x19, x10, x20
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x18, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x18, x10, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x9, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   call    inc
; MYRVX32I_STATIC_MEDLOW-NEXT:   add     x11, x10, x18
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x10, x19, 0
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x20, 8(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x19, 12(x2)             # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x18, 16(x2)             # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x9, 20(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x2, 24(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   lw      x1, 28(x2)              # 4-byte Folded Reload
; MYRVX32I_STATIC_MEDLOW-NEXT:   addi    x2, x2, 32
; MYRVX32I_STATIC_MEDLOW-NEXT:   tail    tail_call_func


; MYRVX64I_STATIC_MEDLOW-LABEL: tail_call_tail:
; MYRVX64I_STATIC_MEDLOW:       # %bb.0:
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, -48
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x1, 40(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x2, 32(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x9, 24(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x18, 16(x2)             # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x19, 8(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	sd	x20, 0(x2)              # 8-byte Folded Spill
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x9, x13, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x18, x12, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x19, x11, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x20, x10, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x19, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x19, x10, x20
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x18, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x18, x10, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x9, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	call	inc
; MYRVX64I_STATIC_MEDLOW-NEXT: 	add	x11, x10, x18
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x10, x19, 0
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x20, 0(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x19, 8(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x18, 16(x2)             # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x9, 24(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x2, 32(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	ld	x1, 40(x2)              # 8-byte Folded Reload
; MYRVX64I_STATIC_MEDLOW-NEXT: 	addi	x2, x2, 48
; MYRVX64I_STATIC_MEDLOW-NEXT: 	tail	tail_call_func

entry:
  %call = call i32 @inc(i32 %a)
  %call1 = call i32 @inc(i32 %b)
  %add = add nsw i32 %call1, %call
  %call2 = call i32 @inc(i32 %c)
  %call3 = call i32 @inc(i32 %d)
  %add4 = add nsw i32 %call3, %call2
  %call5 = tail call i32 @tail_call_func(i32 %add, i32 %add4)
  ret i32 %call5
}

declare dso_local i32 @inc(i32) local_unnamed_addr #1
declare dso_local i32 @tail_call_func(i32, i32) local_unnamed_addr #1


attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.1 (https://github.com/llvm/llvm-project.git a436ae5e999d88734216352da31025b1435d934e)"}
