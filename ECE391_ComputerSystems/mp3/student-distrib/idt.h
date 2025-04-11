#ifndef IDT_H
#define IDT_H

#include "x86_desc.h"
#include "types.h"
#include "lib.h"
#include "asm_linkage.h"
#include "syscall.h"

#define NUM_EXC     0x14
#define NUM_INT     0x20
#define NUM_SYS     0x01
#define INT_BASE    0x20
#define SYS_BASE    0x80

/* idt initialization */
void init_idt();

/* store regsters in the form of a struct*/
struct  x86_registers {
    // General-purpose registers
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
    uint32_t esi;
    uint32_t edi;
    uint32_t ebp;
    uint32_t esp;
} __attribute__((packed));

/* define various idt functions*/
//need an assembly linkage to store registers, flags, and the cause
void exception_handler(uint32_t index, uint32_t EFLAG, struct x86_registers regs);

#endif  // IDT_H
