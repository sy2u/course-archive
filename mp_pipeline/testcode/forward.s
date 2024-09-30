.section .text
.globl _start
_start:
    # Decode
    addi x1, x1, 1
    nop
    nop
    addi x1, x1, 1
    # Normal
    addi x1, x0, 4
    addi x3, x1, 8
    or   x2, x1, x3
    sub  x4, x2, x3
    and  x5, x1, x3

    slti x0, x0, -256 # this is the magic instruction to end the simulation