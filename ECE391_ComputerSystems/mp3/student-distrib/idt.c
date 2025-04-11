#include "idt.h"

/* init_idt
 * Description: Initialize IDT descriptors, set bits and insert handlers
 * Inputs: None
 * Outputs: None
 * Side Effects: Modify IDT entries
 * Return Value: None
 */
void init_idt(){
    int i;
    for(i = 0; i < NUM_VEC; i++){
        idt[i].val[0] = 0;
        idt[i].val[1] = 0;
    }
    // init exceptions
    for(i = 0; i < NUM_EXC; i++){
        idt[i].dpl = 0;
        idt[i].present = 1;
        idt[i].size = 1;
        idt[i].reserved1 = 1;
        idt[i].reserved2 = 1;
        idt[i].reserved3 = 0;
        idt[i].seg_selector = KERNEL_CS;
    }
    // init interrupts
    for(i = 0; i < NUM_INT; i++){
        idt[INT_BASE + i].dpl = 0;
        idt[INT_BASE + i].present = 1;
        idt[INT_BASE + i].size = 1;
        idt[INT_BASE + i].reserved1 = 1;
        idt[INT_BASE + i].reserved2 = 1;
        idt[INT_BASE + i].reserved3 = 0;
        idt[INT_BASE + i].seg_selector = KERNEL_CS;
    }
    // init system call
    for(i = 0; i < NUM_SYS; i++){
        idt[SYS_BASE + i].dpl = 3;
        idt[SYS_BASE + i].present = 1;
        idt[SYS_BASE + i].size = 1;
        idt[SYS_BASE + i].reserved1 = 1;
        idt[SYS_BASE + i].reserved2 = 1;
        idt[SYS_BASE + i].reserved3 = 1;
        idt[SYS_BASE + i].seg_selector = KERNEL_CS;
    }
    
    // set exception handlers
    // all index of idt is given by Intel Document
    SET_IDT_ENTRY(idt[0x00], divide_error);
    SET_IDT_ENTRY(idt[0x01], intel_reserved);
    SET_IDT_ENTRY(idt[0x02], nmi_int);
    SET_IDT_ENTRY(idt[0x03], breakpoint);
    SET_IDT_ENTRY(idt[0x04], overflow);
    SET_IDT_ENTRY(idt[0x05], bound_range);
    SET_IDT_ENTRY(idt[0x06], inval_opcode);
    SET_IDT_ENTRY(idt[0x07], device_na);
    SET_IDT_ENTRY(idt[0x08], double_fault);
    SET_IDT_ENTRY(idt[0x09], coprocessor);
    SET_IDT_ENTRY(idt[0x0A], inval_tss);
    SET_IDT_ENTRY(idt[0x0B], segment_na);
    SET_IDT_ENTRY(idt[0x0C], segment_fault);
    SET_IDT_ENTRY(idt[0x0D], general_protect);
    SET_IDT_ENTRY(idt[0x0E], page_fault);
    SET_IDT_ENTRY(idt[0x0F], intel_reserved_2);
    SET_IDT_ENTRY(idt[0x10], FPU_FP);
    SET_IDT_ENTRY(idt[0x11], alignment);
    SET_IDT_ENTRY(idt[0x12], machine);
    SET_IDT_ENTRY(idt[0x13], SIMD_FP);
    // set interrupt handlers
    SET_IDT_ENTRY(idt[0x20], pit_lnk);
    SET_IDT_ENTRY(idt[0x21], kb_lnk);
    SET_IDT_ENTRY(idt[0x28], rtc_lnk);
    SET_IDT_ENTRY(idt[0x2C], ms_lnk);
    // set system call handlers
    SET_IDT_ENTRY(idt[0x80], sys_lnk);
}

/* exception_handler
 * Description: Exception common handler for 0x00-0x13
 * Inputs: Exception index, EFLAGS, Registers
 * Outputs: Print exception type and register infos to screen
 * Side Effects: None
 * Return Value: None
 */
void exception_handler(uint32_t index, uint32_t EFLAG, struct x86_registers regs){
    // squash exception
    pcb_t* pcb = get_cur_pcb();
    pcb->exception = 1;
    #ifdef DEBUG
        int i;
        for( i = 0; i < 6; i++ ){
            pcb_t* cur_pcb = get_pcb(i);
            printf("Process %d PCB info:\n",i);
            printf("pid=%d\n",cur_pcb->cur_pid);
            printf("ebp=%d\n",cur_pcb->ebp);
            printf("esp=%d\n",cur_pcb->esp);
            printf("parent_pid=%d\n\n",cur_pcb->parent_pid);
        }
        asm volatile("movl %%cr2, %%eax \n"
                    : /* no output */
                    : /* no input */
                    : "eax");
        register uint32_t eax asm("eax");
        printf("cr2=%u\n", eax);
    #endif
    halt(index);
    // blue();
    // clear(); //clear the screen
    // printf("\n\n\n\n"); //adding spacing to make it appear at the top
    // printf("you died skill issue\n");
    // printf("\nreasoning: %s\n", exceptions[index]);
    // printf("\nprinting out regs\n");
    // printf("EAX: %u\n", regs.eax);
    // printf("EBX: %u\n", regs.ebx);
    // printf("ECX: %u\n", regs.ecx);
    // printf("EDX: %u\n", regs.edx);
    // printf("ESI: %u\n", regs.esi);
    // printf("EDI: %u\n", regs.edi);
    // printf("EBP: %u\n", regs.ebp);
    // printf("ESP: %u\n", regs.esp);
    // printf("\nprinting out EFLAGS\n");
    // printf("EFLAGS: %u", EFLAG);
    // while(1);
}
