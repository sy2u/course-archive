#ifndef PAGING_H
#define PAGING_H

#include "asm_paging.h"
#include "types.h" // for using standard integer types

// Definitions for page table entries (PTE) and page directory entries (PDE)
#define ENTRY_NUM       1024
#define FOUR_KB         4096
#define FOUR_MB         0x400000

// Assume physical and virtual addresses match for simplicity
#define VRAM_ADDR       0xB8000
#define VBUF_ADDR       (VRAM_ADDR + FOUR_KB)
#define KERNEL_START    0x400000

// Fixed user program virtual address
#define PROGRAM_VIRT_ADDR   0x08000000
#define PROGRAM_PDE_IDX     (PROGRAM_VIRT_ADDR / FOUR_MB)
#define VIDMAP_ADDR         (PROGRAM_VIRT_ADDR + FOUR_MB)

/* An paging directory/table descriptor entry */
/* Structure refer to x86_desc.h, Descriptor refer to lecture slide */
typedef union page_desc_t {
    uint32_t val;
    struct {
        uint32_t present    : 1; // lowest bit
        uint32_t read_write : 1;
        uint32_t user_kernel: 1; // 0 for kernel only, 1 allow user access
        uint32_t reserved   : 2; // PWT & PCD, set to 0
        uint32_t accessed   : 1;
        uint32_t dirty      : 1;
        uint32_t page_size  : 1; // PAT for PTE
        uint32_t global     : 1;
        uint32_t available  : 3;
        uint32_t addr       : 20; // highest bits
    } __attribute__ ((packed));
} page_desc_t;

void paging_init();
void set_user_page(uint32_t phys_addr);
void enable_vidmap_page();
void set_vidmap(uint32_t phys_addr);
uint32_t get_vidmap();

#endif // PAGING_H
