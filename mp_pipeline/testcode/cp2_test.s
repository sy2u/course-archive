.section .text
.globl _start
_start:
    # init memory
    lui  x8, 0x1ecec
    lui  x30, 0x1ee00
    addi x9, x0, 4
    addi x1, x0, 1
    addi x2, x0, 2
    addi x3, x0, 3
    nop
    nop
    nop
    nop
    nop
    sw x9, 0(x8)
    sb x1, 1(x8)
    sh x2, 2(x8)
    lb x5, 1(x8)
    sb x9, 3(x8)
    sb x2, 2047(x8)
    sb x9, 0(x30)
    sb x1, 1(x30)
    sb x2, 2(x30)
    sb x2, 8(x30)
    sb x9, 3(x30)
    sb x1, 4(x30)
    sb x2, 5(x30)
    sb x2, 66(x30)
    lw x5, 0(x8) # cache miss
    lb x9, 1(x8)
    lb x10, 1(x8)
    lb x11, 1(x8)
    lh x9, 2(x8)
    lb x9, 3(x8)

    slti x0, x0, -256 # this is the magic instruction to end the simulation