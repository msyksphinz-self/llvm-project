# RUN: llvm-mc %s -triple=myriscvx64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s



# CHECK-ASM-AND-OBJ: lwu     x0, 4(x1)
# CHECK-ASM: encoding: [0x03,0xe0,0x40,0x00]
lwu     zero,4(ra)
# CHECK-ASM-AND-OBJ: lwu     x2, 4(x3)
# CHECK-ASM: encoding: [0x03,0xe1,0x41,0x00]
lwu     sp,4(gp)
# CHECK-ASM-AND-OBJ: lwu     x4, -2048(x5)
# CHECK-ASM: encoding: [0x03,0xe2,0x02,0x80]
lwu     tp,-2048(t0)

# CHECK-ASM-AND-OBJ: ld      x10, -2048(x11)
# CHECK-ASM: encoding: [0x03,0xb5,0x05,0x80]
ld      a0,-2048(a1)
# CHECK-ASM-AND-OBJ: ld      x14, 2047(x15)
# CHECK-ASM: encoding: [0x03,0xb7,0xf7,0x7f]
ld      a4,2047(a5)

# CHECK-ASM-AND-OBJ: sd      x16, -2048(x17)
# CHECK-ASM: encoding: [0x23,0xb0,0x08,0x81]
sd      a6,-2048(a7)
# CHECK-ASM-AND-OBJ: sd      x20, 2047(x21)
# CHECK-ASM: encoding: [0xa3,0xbf,0x4a,0x7f]
sd      s4,2047(s5)

# CHECK-ASM-AND-OBJ: slli    x22, x23, 45
# CHECK-ASM: encoding: [0x13,0x9b,0xdb,0x02]
slli    s6,s7,0x2d
# CHECK-ASM-AND-OBJ: srli    x24, x25, 0
# CHECK-ASM: encoding: [0x13,0xdc,0x0c,0x00]
srli    s8,s9,0x0
# CHECK-ASM-AND-OBJ: srai    x26, x27, 31
# CHECK-ASM: encoding: [0x13,0xdd,0xfd,0x41]
srai    s10,s11,0x1f
# CHECK-ASM-AND-OBJ: srai    x26, x27, 31
# CHECK-ASM: encoding: [0x13,0xdd,0xfd,0x41]
srai    s10,s11,0x1f

# CHECK-ASM-AND-OBJ: addiw   x28, x29, -2048
# CHECK-ASM: encoding: [0x1b,0x8e,0x0e,0x80]
addiw   t3,t4,-2048
# CHECK-ASM-AND-OBJ: addiw   x30, x31, 2047
# CHECK-ASM: encoding: [0x1b,0x8f,0xff,0x7f]
addiw   t5,t6,2047

# CHECK-ASM-AND-OBJ: slliw   x0, x1, 0
# CHECK-ASM: encoding: [0x1b,0x90,0x00,0x00]
slliw   zero,ra,0x0
# CHECK-ASM-AND-OBJ: slliw   x2, x3, 31
# CHECK-ASM: encoding: [0x1b,0x91,0xf1,0x01]
slliw   sp,gp,0x1f
# CHECK-ASM-AND-OBJ: srliw   x4, x5, 0
# CHECK-ASM: encoding: [0x1b,0xd2,0x02,0x00]
srliw   tp,t0,0x0
# CHECK-ASM-AND-OBJ: srliw   x6, x7, 31
# CHECK-ASM: encoding: [0x1b,0xd3,0xf3,0x01]
srliw   t1,t2,0x1f
# CHECK-ASM-AND-OBJ: sraiw   x8, x9, 0
# CHECK-ASM: encoding: [0x1b,0xd4,0x04,0x40]
sraiw   s0,s1,0x0
# CHECK-ASM-AND-OBJ: sraiw   x10, x11, 31
# CHECK-ASM: encoding: [0x1b,0xd5,0xf5,0x41]
sraiw   a0,a1,0x1f
# CHECK-ASM-AND-OBJ: sraiw   x10, x11, 31
# CHECK-ASM: encoding: [0x1b,0xd5,0xf5,0x41]
sraiw   a0,a1,0x1f

# CHECK-ASM-AND-OBJ: addw    x12, x13, x14
# CHECK-ASM: encoding: [0x3b,0x86,0xe6,0x00]
addw    a2,a3,a4
# CHECK-ASM-AND-OBJ: addw    x15, x16, x17
# CHECK-ASM: encoding: [0xbb,0x07,0x18,0x01]
addw    a5,a6,a7
# CHECK-ASM-AND-OBJ: subw    x18, x19, x20
# CHECK-ASM: encoding: [0x3b,0x89,0x49,0x41]
subw    s2,s3,s4
# CHECK-ASM-AND-OBJ: subw    x21, x22, x23
# CHECK-ASM: encoding: [0xbb,0x0a,0x7b,0x41]
subw    s5,s6,s7
# CHECK-ASM-AND-OBJ: sllw    x24, x25, x26
# CHECK-ASM: encoding: [0x3b,0x9c,0xac,0x01]
sllw    s8,s9,s10
# CHECK-ASM-AND-OBJ: srlw    x27, x28, x29
# CHECK-ASM: encoding: [0xbb,0x5d,0xde,0x01]
srlw    s11,t3,t4
# CHECK-ASM-AND-OBJ: sraw    x30, x31, x0
# CHECK-ASM: encoding: [0x3b,0xdf,0x0f,0x40]
sraw    t5,t6,zero
