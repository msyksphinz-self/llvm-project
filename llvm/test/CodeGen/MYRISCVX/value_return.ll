; @{ value_return_test
; MYRISCVX32用のテストオプション: MYRVX32Iのラベル部分を使用してテスト
; RUN: llc --march=myriscvx32 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX32I %s

; MYRISCVX64用のテストオプション: MYRVX64Iのラベル部分を使用してテスト
; RUN: llc --march=myriscvx64 < %s \
; RUN:   | FileCheck -check-prefix=MYRVX64I %s

; @{ value_return_run
define dso_local i32 @value_return() #0 {

; @{ value_return_label_myriscvx32
;; MYRVX32Iの場合はこのアセンブリ命令と比較が行われる
; MYRVX32I-LABEL: value_return:                           # @value_return
; MYRVX32I:      # %bb.0:                                # %entry
; MYRVX32I-NEXT:         addi    x10, x0, 100
; MYRVX32I-NEXT:         ret
; @} value_return_label_myriscvx32

; @{ value_return_label_myriscvx64
;; MYRVX64Iの場合はこのアセンブリ命令と比較が行われる
; MYRVX64I-LABEL: value_return:                           # @value_return
; MYRVX64I:      # %bb.0:                                # %entry
; MYRVX64I-NEXT:         addi    x10, x0, 100
; MYRVX64I-NEXT:         ret
; @} value_return_label_myriscvx64

; テスト本体
entry:
  ret i32 100
}
; @} value_return_run
; @} value_return_test
