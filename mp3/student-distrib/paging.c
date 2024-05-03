#include "paging.h"

// global variable
// Align arrays to 4096 (4KB) boundary
page_desc_t page_directory[ENTRY_NUM] __attribute__((aligned(FOUR_KB)));
page_desc_t page_table_vram[ENTRY_NUM] __attribute__((aligned(FOUR_KB)));
page_desc_t page_table_vidmap[ENTRY_NUM] __attribute__((aligned(FOUR_KB)));

/* paging_init
 * Description: Initialize paging and make b8000-b9000, 400000-800000 present
 * Inputs: None
 * Outputs: None
 * Side Effects: Modify paging related registers, cr0, cr3, cr4
 * Return Value: None
 */
void paging_init() {
    int i;

    // Initialize all page directory entries and all first PTE entries
    // Not present, read and write, kernel priority, not accessed, 4kb page
    for (i = 0; i < ENTRY_NUM; i++) { 
        page_directory[i].val = 0;
        page_table_vram[i].val = 0;
        page_table_vram[i].read_write = 1;
        page_table_vidmap[i].val = 0;
        page_table_vidmap[i].read_write = 1;
        page_directory[i].read_write = 1;  
    }

    // Special case for video memory
    // Calculate the page table entry for video memory by dividing the video memory address by 4096 (size of a page)
    int vram_pte_idx = VRAM_ADDR / FOUR_KB;
    page_table_vram[vram_pte_idx].addr = (VRAM_ADDR >> 12); // write base addr, 12 bits for offset
    page_table_vram[vram_pte_idx].present = 1;
    // vram buffers
    page_table_vram[vram_pte_idx + 1].addr = ( (VRAM_ADDR+FOUR_KB) >> 12); // 0xb9000-0xba000
    page_table_vram[vram_pte_idx + 1].present = 1;
    page_table_vram[vram_pte_idx + 2].addr = ( (VRAM_ADDR+2*FOUR_KB) >> 12); // 0xba000-0xbb000
    page_table_vram[vram_pte_idx + 2].present = 1;
    page_table_vram[vram_pte_idx + 3].addr = ( (VRAM_ADDR+3*FOUR_KB) >> 12); // 0xbb000-0xbc000
    page_table_vram[vram_pte_idx + 3].present = 1;

    // Set the first page table to present, writable, 4KB
    page_directory[0].addr = ((uint32_t)page_table_vram >> 12);
    page_directory[0].present = 1;

    // Directly map 4MB to 8MB for kernel, present, 4MB
    page_directory[1].addr = (KERNEL_START >> 12);  // write base addr, 12 bits for offset
    page_directory[1].present = 1;
    page_directory[1].page_size = 1;

    // add vidmap page
    page_directory[VIDMAP_ADDR / FOUR_MB].present = 1;
    page_table_vidmap[0].present = 1;
    // write base addr, 12 bits for offset
    page_directory[VIDMAP_ADDR / FOUR_MB].addr = ((uint32_t)page_table_vidmap >> 12);
    page_table_vidmap[0].addr = (VRAM_ADDR >> 12); 
    // allow user access
    page_directory[VIDMAP_ADDR / FOUR_MB].user_kernel = 1;
    page_table_vidmap[0].user_kernel = 1;

    // Load the page directory address into CR3 and enable paging via assembly functions
    enable_paging((uint32_t)page_directory);
}

/* set_user_page
 * Description: Set up new page for new called user program from execute. 4MB each.
   Inputs: phys_addr -- physical address to be mapped for the user program.
 * Outputs: None
 * Side Effects: Modify paging related registers, cr0, cr3, cr4
 * Return Value: None
 */
void set_user_page(uint32_t phys_addr){
    page_directory[PROGRAM_PDE_IDX].addr = (phys_addr >> 12);  // write base addr, 12 bits for offset
    page_directory[PROGRAM_PDE_IDX].present = 1; // enable page
    page_directory[PROGRAM_PDE_IDX].page_size = 1; // 4MB large page
    page_directory[PROGRAM_PDE_IDX].user_kernel = 1; // allow user access
    flush_tlb();
}

/* set_vidmap
 * Description: change VRAM mapping
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void set_vidmap(uint32_t phys_addr){
    page_table_vram[VRAM_ADDR/FOUR_KB].addr = (phys_addr >> 12); // write base addr, 12 bits for offset
    page_table_vidmap[0].addr = (phys_addr >> 12);  // write base addr, 12 bits for offset
    flush_tlb();
}

/* get_vidmap
 * Description: get current VRAM mapping
 * Outputs: None
 * Side Effects: None
 * Return Value: the mapped physical address for current process
 */
uint32_t get_vidmap(){ return page_table_vidmap[0].addr; }
