#include "schedule.h"
#include "lib.h"

int32_t cur_sched = 0; // TA
uint8_t cur_term = 0; // TS

int32_t sched_arr[SCHED_NUM] = {NO_TERM, NO_TERM, NO_TERM};
extern mouse_t mouse_info;

/* init_cur_sched
 * Description: initialize scheduled pid for the first call of scheduler
                make sure term 0-2 and pid 0-2 have exactly 1 to 1 mapping
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void init_cur_sched(){ cur_sched = -1; }

/* terminal_switch
 * Description: switch visible terminal
 * Inputs: tar_term - next terminal to jump to
 * Outputs: Switch screen
 * Side Effects: Change VRAM and VBUF
 * Return Value: 0 if success, -1 if fail
 */
int32_t terminal_switch(int tar_term){
    uint32_t cur_vidmap;

    // sanity check
    if( tar_term > 2 || tar_term < 0 ){ return -1; }

    // switch VRAM
    cur_vidmap = get_vidmap();
    set_vidmap(VRAM_ADDR);
    reset_attrib(mouse_info.cur_x, mouse_info.cur_y);
    memcpy((uint8_t*)(VBUF_ADDR+cur_term*FOUR_KB) ,(uint8_t*)VRAM_ADDR, FOUR_KB);
    memcpy((uint8_t*)VRAM_ADDR, (uint8_t*)(VBUF_ADDR+tar_term*FOUR_KB), FOUR_KB);
    // check TS & TA, reset vidmap
    if(tar_term != cur_sched){ set_vidmap(cur_vidmap); }

    // set parameters
    cur_term = tar_term;
    return 0;
}

/* scheduler
 * Description: run each process in turn, invisible to user
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 if success, -1 if fail
 */
int32_t scheduler() {
    pcb_t *cur_pcb, *next_pcb;
    int32_t pre_sched = cur_sched;

    cur_sched = (pre_sched + 1) % SCHED_NUM;
    cur_pcb = get_cur_pcb();
    next_pcb = get_pcb(sched_arr[cur_sched]);

    register uint32_t saved_ebp asm("ebp");
    register uint32_t saved_esp asm("esp");
    cur_pcb->ebp = saved_ebp;
    cur_pcb->esp = saved_esp;
    
    // set vidmap
    if(cur_term == cur_sched){ set_vidmap(VRAM_ADDR); } else { set_vidmap(VBUF_ADDR + cur_sched*FOUR_KB); }

    // launch base shells
    if (sched_arr[cur_sched] == NO_TERM) {
        sched_arr[cur_sched] = (cur_sched - 1) % SCHED_NUM;
        execute((uint8_t*)"shell");
    }

    // do context switch
    tss.ss0 = KERNEL_DS;
    tss.esp0 = EIGHT_MB - sched_arr[cur_sched] * EIGHT_KB - 4;

    set_user_page(EIGHT_MB + sched_arr[cur_sched] * FOUR_MB);
    
    // Jump to nex return 
    asm volatile("movl %0, %%ebp \n\
                  movl %1, %%esp \n\
                  leave          \n\
                  ret            \n"
                : /* no output */
                : "r" (next_pcb->ebp),\
                  "r" (next_pcb->esp)
                : "ebp", "esp");

    return 0;

}
