.section .text
.globl _start
_start:
    # Load Hazard
    lui  x10, 0x1ecec
    addi x1, x0, 326
    addi x2, x0, 20
    sw   x1, 0(x10)
    lw   x3, 0(x10)
    addi x2, x3, 1
    lw   x4, 0(x10)
    add  x5, x4, x2 
    # Decode
    # Note: without implementing, this hazard also didn't occur (?)
    addi x1, x0, 0
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