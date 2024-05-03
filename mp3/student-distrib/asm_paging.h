#ifndef ASM_PAGING_H
#define ASM_PAGING_H

// Declare functions implemented in assembly so that C files can call them.
extern void enable_paging();
extern void flush_tlb();

#endif // ASM_PAGING_H
