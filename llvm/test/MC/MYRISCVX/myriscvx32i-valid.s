# RUN: llvm-mc %s -triple=myriscvx64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=myriscvx32 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s

.equ CONST, 30

# CHECK-ASM-AND-OBJ: auipc   x10, %pcrel_hi(foo)
# CHECK-ASM:         encoding: [0x17,0bAAAA0101,A,A]
auipc a0, %pcrel_hi(foo)

# CHECK-ASM-AND-OBJ: lui     x10, 2
# CHECK-ASM:         encoding: [0x37,0x25,0x00,0x00]
# CHECK-ASM-AND-OBJ: lui     x27, 552960
# CHECK-ASM:         encoding: [0xb7,0x0d,0x00,0x87]
# IGNORE-CHECK-ASM-AND-OBJ: lui     x10, %hi(2)
# IGNORE-CHECK-ASM:         encoding: [0x37,0x05,0x00,0x00]
# CHECK-ASM-AND-OBJ: lui     x27, 552960
# CHECK-ASM:         encoding: [0xb7,0x0d,0x00,0x87]
# IGNORE-CHECK-ASM-AND-OBJ: lui     x27, %hi(0x87000000)
# IGNORE-CHECK-ASM:         encoding: [0xb7,0x0d,0x00,0x87]
# CHECK-ASM-AND-OBJ: lui     x5, 1048575
# CHECK-ASM:         encoding: [0xb7,0xf2,0xff,0xff]
# CHECK-ASM-AND-OBJ: lui     x3, 0
# CHECK-ASM:         encoding: [0xb7,0x01,0x00,0x00]
# CHECK-ASM-AND-OBJ: lui     x10, %hi(foo)
# CHECK-ASM:         encoding: [0x37,0bAAAA0101,A,A]
# CHECK-ASM-AND-OBJ: lui     x10, 30
# CHECK-ASM:         encoding: [0x37,0xe5,0x01,0x00]
# CHECK-ASM-AND-OBJ: lui     x10, 31
# CHECK-ASM:         encoding: [0x37,0xf5,0x01,0x00]
lui a0, 2
lui s11, (0x87000000>>12)
// lui a0, %hi(2)
lui s11, (0x87000000>>12)
// lui s11, %hi(0x87000000)
lui t0, 1048575
lui gp, 0
lui a0, %hi(foo)
lui a0, CONST
lui a0, CONST+1

# CHECK-ASM-AND-OBJ: auipc   x10, 2
# CHECK-ASM:         encoding: [0x17,0x25,0x00,0x00]
# CHECK-ASM-AND-OBJ: auipc   x27, 552960
# CHECK-ASM:         encoding: [0x97,0x0d,0x00,0x87]
# CHECK-ASM-AND-OBJ: auipc   x5, 1048575
# CHECK-ASM:         encoding: [0x97,0xf2,0xff,0xff]
# CHECK-ASM-AND-OBJ: auipc   x3, 0
# CHECK-ASM:         encoding: [0x97,0x01,0x00,0x00]
# CHECK-ASM-AND-OBJ: auipc   x10, %pcrel_hi(foo)
# CHECK-ASM:         encoding: [0x17,0bAAAA0101,A,A]
# CHECK-ASM-AND-OBJ: auipc   x10, 30
# CHECK-ASM:         encoding: [0x17,0xe5,0x01,0x00]
auipc a0, 2
auipc s11, (0x87000000>>12)
auipc t0, 1048575
auipc gp, 0
auipc a0, %pcrel_hi(foo)
auipc a0, CONST

# CHECK-ASM-AND-OBJ: jal     x12, 1048574
# CHECK-ASM:         encoding: [0x6f,0xf6,0xff,0x7f]
# CHECK-ASM-AND-OBJ: jal     x13, 256
# CHECK-ASM:         encoding: [0xef,0x06,0x00,0x10]
# CHECK-ASM-AND-OBJ: jal     x10, foo
# CHECK-ASM:         encoding: [0x6f,0bAAAA0101,A,A]
# IGNORE-CHECK-ASM-AND-OBJ: jal     x10, a0
# IGNORE-CHECK-ASM:         encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-ASM-AND-OBJ: jal     x10, 30
# CHECK-ASM:         encoding: [0x6f,0x05,0xe0,0x01]
# CHECK-ASM-AND-OBJ: jal     x8, 0
# CHECK-ASM:         encoding: [0x6f,0x04,0x00,0x00]
# CHECK-ASM-AND-OBJ: jal     x8, 156
# CHECK-ASM:         encoding: [0x6f,0x04,0xc0,0x09]
# IGNORE-CHECK-ASM-AND-OBJ: jal     x0, .Ltmp0
# IGNORE-CHECK-ASM:         encoding: [0x6f,0bAAAA0000,A,A]
jal a2, 1048574
jal a3, 256
jal a0, foo
// jal a0, a0
jal a0, CONST
jal s0, (0)
jal s0, (0xff-99)
// jal zero, .


# CHECK-ASM-AND-OBJ: jalr    x10, x11, -2048
# CHECK-ASM:         encoding: [0x67,0x85,0x05,0x80]
# IGNORE-CHECK-ASM-AND-OBJ: jalr    x10, -2048(x11)
# IGNORE-CHECK-ASM:         encoding: [0x67,0x85,0x05,0x80]
# CHECK-ASM-AND-OBJ: jalr    x7, x6, 2047
# CHECK-ASM:         encoding: [0xe7,0x03,0xf3,0x7f]
# CHECK-ASM-AND-OBJ: jalr    x2, x0, 256
# CHECK-ASM:         encoding: [0x67,0x01,0x00,0x10]
# IGNORE-CHECK-ASM-AND-OBJ: jalr    x11, 30(x12)
# IGNORE-CHECK-ASM:         encoding: [0xe7,0x05,0xe6,0x01]
jalr a0, a1, -2048
// jalr a0, %lo(2048)(a1)
jalr t2, t1, 2047
jalr sp, zero, 256
// jalr a1, CONST(a2)


# CHECK-ASM-AND-OBJ: beq     x9, x9, 102
# CHECK-ASM:         encoding: [0x63,0x83,0x94,0x06]
# CHECK-ASM-AND-OBJ: bne     x14, x15, -4096
# CHECK-ASM:         encoding: [0x63,0x10,0xf7,0x80]
# CHECK-ASM-AND-OBJ: blt     x2, x3, 4094
# CHECK-ASM:         encoding: [0xe3,0x4f,0x31,0x7e]
# CHECK-ASM-AND-OBJ: bge     x18, x1, -224
# CHECK-ASM:         encoding: [0xe3,0x50,0x19,0xf2]
# CHECK-ASM-AND-OBJ: bgtu    x0, x0, 0
# CHECK-ASM:         encoding: [0x63,0x60,0x00,0x00]
# CHECK-ASM-AND-OBJ: bleu    x2, x24, 512
# CHECK-ASM:         encoding: [0x63,0x70,0x2c,0x20]
# CHECK-ASM-AND-OBJ: bleu    x6, x5, 30
# CHECK-ASM:         encoding: [0x63,0xff,0x62,0x00]
beq s1, s1, 102
bne a4, a5, -4096
blt sp, gp, 4094
bge s2, ra, -224
bltu zero, zero, 0
bgeu s8, sp, 512
bgeu t0, t1, CONST


# CHECK-ASM-AND-OBJ: lb      x19, 4(x1)
# CHECK-ASM:         encoding: [0x83,0x89,0x40,0x00]
# CHECK-ASM-AND-OBJ: lb      x19, 4(x1)
# CHECK-ASM:         encoding: [0x83,0x89,0x40,0x00]
# CHECK-ASM-AND-OBJ: lh      x6, -2048(x0)
# CHECK-ASM:         encoding: [0x03,0x13,0x00,0x80]
# IGNORE-CHECK-ASM-AND-OBJ: lh      x6, -2048(x0)
# IGNORE-CHECK-ASM:         encoding: [0x03,0x13,0x00,0x80]
# IGNORE-CHECK-ASM-AND-OBJ: lh      x6, 0(x0)
# IGNORE-CHECK-ASM:         encoding: [0x03,0x13,0x00,0x00]
# IGNORE-CHECK-ASM-AND-OBJ: lh      x6, -2048(x0)
# IGNORE-CHECK-ASM:         encoding: [0x03,0x13,0x00,0x80]
# CHECK-ASM-AND-OBJ: lh      x2, 2047(x10)
# CHECK-ASM:         encoding: [0x03,0x11,0xf5,0x7f]
# CHECK-ASM-AND-OBJ: lw      x10, 97(x12)
# CHECK-ASM:         encoding: [0x03,0x25,0x16,0x06]
# IGNORE-CHECK-ASM-AND-OBJ: lbu     x21, %lo(foo)(x22)
# IGNORE-CHECK-ASM:         encoding: [0x83,0x4a,0bAAAA1011,A]
# IGNORE-CHECK-ASM-AND-OBJ: lhu     x28, %pcrel_lo(.Lpcrel_hi0)(x28) #
# IGNORE-CHECK-ASM:         encoding: [0x03,0x5e,0bAAAA1110,A]
# IGNORE-CHECK-ASM-AND-OBJ: lb      x5, 30(x6)
# IGNORE-CHECK-ASM:         encoding: [0x83,0x02,0xe3,0x01]
# CHECK-ASM-AND-OBJ: lb      x8, 0(x9)
# CHECK-ASM:         encoding: [0x03,0x84,0x04,0x00]
# CHECK-ASM-AND-OBJ: lb      x8, 156(x9)
# CHECK-ASM:         encoding: [0x03,0x84,0xc4,0x09]
lb s3, 4(ra)
lb s3, +4(ra)
lh t1, -2048(zero)
// lh t1, ~2047(zero)
// lh t1, !1(zero)
// lh t1, %lo(2048)(zero)
lh sp, 2047(a0)
lw a0, 97(a2)
// lbu s5, %lo(foo)(s6)
// lhu t3, %pcrel_lo(.Lpcrel_hi0)(t3)
// lb t0, CONST(t1)
lb s0, (0)(s1)
lb s0, (0xff-99)(s1)


# CHECK-ASM-AND-OBJ: sb      x10, 2047(x12)
# CHECK-ASM:         encoding: [0xa3,0x0f,0xa6,0x7e]
# CHECK-ASM-AND-OBJ: sh      x28, -2048(x30)
# CHECK-ASM:         encoding: [0x23,0x10,0xcf,0x81]
# IGNORE-CHECK-ASM-AND-OBJ: sh      x28, -2048(x30)
# IGNORE-CHECK-ASM:         encoding: [0x23,0x10,0xcf,0x81]
# IGNORE-CHECK-ASM-AND-OBJ: sh      x28, 0(x30)
# IGNORE-CHECK-ASM:         encoding: [0x23,0x10,0xcf,0x01]
# IGNORE-CHECK-ASM-AND-OBJ: sh      x28, -2048(x30)
# IGNORE-CHECK-ASM:         encoding: [0x23,0x10,0xcf,0x81]
# CHECK-ASM-AND-OBJ: sw      x1, 999(x0)
# CHECK-ASM:         encoding: [0xa3,0x23,0x10,0x3e]
# IGNORE-CHECK-ASM-AND-OBJ: sw      x10, 30(x5)
# IGNORE-CHECK-ASM:         encoding: [0x23,0xaf,0xa2,0x00]
# CHECK-ASM-AND-OBJ: sw      x8, 0(x9)
# CHECK-ASM:         encoding: [0x23,0xa0,0x84,0x00]
# CHECK-ASM-AND-OBJ: sw      x8, 156(x9)
# CHECK-ASM:         encoding: [0x23,0xae,0x84,0x08]
sb a0, 2047(a2)
sh t3, -2048(t5)
// sh t3, ~2047(t5)
// sh t3, !1(t5)
// sh t3, %lo(2048)(t5)
sw ra, 999(zero)
// sw a0, CONST(t0)
sw s0, (0)(s1)
sw s0, (0xff-99)(s1)


# CHECK-ASM-AND-OBJ: addi    x1, x2, 2
# CHECK-ASM:         encoding: [0x93,0x00,0x21,0x00]
# CHECK-ASM-AND-OBJ: addi    x1, x2, %lo(foo)
# CHECK-ASM:         encoding: [0x93,0x00,0bAAAA0001,A]
# CHECK-ASM-AND-OBJ: addi    x1, x2, 30
# CHECK-ASM:         encoding: [0x93,0x00,0xe1,0x01]
# CHECK-ASM-AND-OBJ: mv      x1, x2
# CHECK-ASM:         encoding: [0x93,0x00,0x01,0x00]
# CHECK-ASM-AND-OBJ: addi    x1, x2, 156
# CHECK-ASM:         encoding: [0x93,0x00,0xc1,0x09]
# CHECK-ASM-AND-OBJ: slti    x10, x12, -20
# CHECK-ASM:         encoding: [0x13,0x25,0xc6,0xfe]
# CHECK-ASM-AND-OBJ: sltiu   x18, x19, 80
# CHECK-ASM:         encoding: [0x13,0xb9,0x09,0x05]
# CHECK-ASM-AND-OBJ: xori    x4, x6, -99
# CHECK-ASM:         encoding: [0x13,0x42,0xd3,0xf9]
# CHECK-ASM-AND-OBJ: ori     x10, x11, -2048
# CHECK-ASM:         encoding: [0x13,0xe5,0x05,0x80]
# IGNORE-CHECK-ASM-AND-OBJ: ori     x10, x11, -2048
# IGNORE-CHECK-ASM:         encoding: [0x13,0xe5,0x05,0x80]
# IGNORE-CHECK-ASM-AND-OBJ: ori     x10, x11, 0
# IGNORE-CHECK-ASM:         encoding: [0x13,0xe5,0x05,0x00]
# IGNORE-CHECK-ASM-AND-OBJ: ori     x10, x11, -2048
# IGNORE-CHECK-ASM:         encoding: [0x13,0xe5,0x05,0x80]
# CHECK-ASM-AND-OBJ: andi    x1, x2, 2047
# CHECK-ASM:         encoding: [0x93,0x70,0xf1,0x7f]
# CHECK-ASM-AND-OBJ: andi    x1, x2, 2047
# CHECK-ASM:         encoding: [0x93,0x70,0xf1,0x7f]
addi ra, sp, 2
addi ra, sp, %lo(foo)
addi ra, sp, CONST
addi ra, sp, (0)
addi ra, sp, (0xff-99)
slti a0, a2, -20
sltiu s2, s3, 0x50
xori tp, t1, -99
ori a0, a1, -2048
// ori a0, a1, ~2047
// ori a0, a1, !1
// ori a0, a1, %lo(2048)
andi ra, sp, 2047
andi x1, x2, 2047


# CHECK-ASM-AND-OBJ: slli    x28, x28, 31
# CHECK-ASM:         encoding: [0x13,0x1e,0xfe,0x01]
# CHECK-ASM-AND-OBJ: srli    x10, x14, 0
# CHECK-ASM:         encoding: [0x13,0x55,0x07,0x00]
# CHECK-ASM-AND-OBJ: srai    x12, x2, 15
# CHECK-ASM:         encoding: [0x13,0x56,0xf1,0x40]
# CHECK-ASM-AND-OBJ: slli    x28, x28, 30
# CHECK-ASM:         encoding: [0x13,0x1e,0xee,0x01]
slli t3, t3, 31
srli a0, a4, 0
srai a2, sp, 15
slli t3, t3, CONST


# CHECK-ASM-AND-OBJ: add     x1, x0, x0
# CHECK-ASM:         encoding: [0xb3,0x00,0x00,0x00]
# CHECK-ASM-AND-OBJ: add     x1, x0, x0
# CHECK-ASM:         encoding: [0xb3,0x00,0x00,0x00]
# CHECK-ASM-AND-OBJ: sub     x5, x7, x6
# CHECK-ASM:         encoding: [0xb3,0x82,0x63,0x40]
# CHECK-ASM-AND-OBJ: sll     x15, x14, x13
# CHECK-ASM:         encoding: [0xb3,0x17,0xd7,0x00]
# CHECK-ASM-AND-OBJ: slt     x8, x8, x8
# CHECK-ASM:         encoding: [0x33,0x24,0x84,0x00]
# CHECK-ASM-AND-OBJ: sltu    x3, x10, x11
# CHECK-ASM:         encoding: [0xb3,0x31,0xb5,0x00]
# CHECK-ASM-AND-OBJ: xor     x18, x18, x24
# CHECK-ASM:         encoding: [0x33,0x49,0x89,0x01]
# CHECK-ASM-AND-OBJ: xor     x18, x18, x24
# CHECK-ASM:         encoding: [0x33,0x49,0x89,0x01]
# CHECK-ASM-AND-OBJ: srl     x10, x8, x5
# CHECK-ASM:         encoding: [0x33,0x55,0x54,0x00]
# CHECK-ASM-AND-OBJ: sra     x5, x18, x0
# CHECK-ASM:         encoding: [0xb3,0x52,0x09,0x40]
# CHECK-ASM-AND-OBJ: or      x26, x6, x1
# CHECK-ASM:         encoding: [0x33,0x6d,0x13,0x00]
# CHECK-ASM-AND-OBJ: and     x10, x18, x19
# CHECK-ASM:         encoding: [0x33,0x75,0x39,0x01]
add ra, zero, zero
add x1, x0, x0
sub t0, t2, t1
sll a5, a4, a3
slt s0, s0, s0
sltu gp, a0, a1
xor s2, s2, s8
xor x18, x18, x24
srl a0, s0, t0
sra t0, s2, zero
or s10, t1, ra
and a0, s2, s3
