#include "pit.h"
#include "syscall.h"
#include "schedule.h"
#include "lib.h"

extern int32_t cur_sched;
extern uint8_t cur_term;
extern mouse_t mouse_info;

/* pit_init
 * Description: initialize PIT hardware
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void pit_init() {
    outb(PIT_MODE, PIT_CMD);
    outb(FREQ_DIV & 0xFF, CHANEL_0);
    outb((FREQ_DIV >> 8) & 0xFF, CHANEL_0);
    enable_irq(PIT_IRQ);
}

/* pit_handler
 * Description: handler for PIT interrupt, used for scheduling
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void pit_handler() {
    send_eoi(PIT_IRQ);
    if( cur_sched == cur_term ){
        update_term_cursor();
        read_rtc();
        update_status_bar();
        if( mouse_info.update_flag ){ update_mouse(); }
    }
    scheduler();
}
