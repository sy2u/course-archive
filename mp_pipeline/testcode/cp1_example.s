.section .text
.globl _start
_start:
    addi x1, x0, 4
    nop             # nops in between to prevent hazard
    nop
    nop
    nop
    nop
    addi x3, x1, 8
    nop
    nop
    nop
    nop
    nop
    lui x1, 4
    nop
    nop
    nop
    nop
    nop
    lui x2, 1
    nop
    nop
    nop
    nop
    nop    
    srl x5, x2, x1
    nop
    nop
    nop
    nop
    nop  
    slt x5, x2, x1
    nop
    nop
    nop
    nop
    nop
    sltu x5, x2, x1
    nop
    nop
    nop
    nop
    nop
    xor x3, x2, x1
    nop
    nop
    nop
    nop
    nop
    slti x5, x1, 10
    nop
    nop
    nop
    nop
    nop
    xori x5, x1, 10
    nop
    nop
    nop
    nop
    nop
    slli x5, x1, 10
    nop
    nop
    nop
    nop
    nop


    slti x0, x0, -256 # this is the magic instruction to end the simulation
