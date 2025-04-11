/* rtc.c - Functions to interact with the real timing clock */

#include "rtc.h"
#include "schedule.h"

extern int32_t cur_sched;
// global variable
rtc_t rtc_info[NUM_TERM];
rtc_t timer_speaker;

// RTC Interrupt Interface
/* rtc_init
 * Description: Initialize RTC, enable IRQ8 and enable PIC port
 * Inputs: None
 * Outputs: None
 * Side Effects: Modify RTC registers, PIC mask Register
 * Return Value: None
 */
void rtc_init(){
    unsigned char prev;
    // enable irq 8
    outb(REG_B, REG_PORT);  // select reg B, disable NMI
    prev = inb(REG_DATA);   // save current reg B value
    outb(REG_B, REG_PORT);
    outb(prev | MASK_BIT_6, REG_DATA);  // enable 6th bit
    // set frequency to 1024Hz
    outb(REG_A, REG_PORT);  // select reg A, disable NMI
    prev = inb(REG_DATA);
    outb(REG_A, REG_PORT);	
    outb((prev & MAST_LOW_4) | RTC_RATE, REG_DATA); // rate is the bottom 4 bits, base freq = 1024Hz
    // enable pic ports
    enable_irq(RTC_IRQ); 
}

/* rtc_hdl
 * Description: RTC handler, to be wrapped in linkage and inserted in IDT
 * Inputs: None
 * Outputs: Screen flash with random chars
 * Side Effects: Writing VRAM
 * Return Value: None
 */
void rtc_hdl(){
    // for rtc virtualization
    // update all opening rtc counters
    int i;
    for(i = 0; i < NUM_TERM; i++){
        if(rtc_info[i].rtc_en){
            rtc_info[i].rtc_cnt--;
            if(rtc_info[i].rtc_cnt <= 0){
                    rtc_info[i].rtc_int = 1;
                    rtc_info[i].rtc_cnt = rtc_info[i].rtc_set_cnt;
                }
        }
    }
    // update speaker timer counter
    if(timer_speaker.rtc_en){
        timer_speaker.rtc_cnt--; 
        if(timer_speaker.rtc_cnt <= 0){
            timer_speaker.rtc_int = 1;
        }
    }
    // make sure keep receiving interrupts
    outb(REG_C, REG_PORT);  // select reg C
    inb(REG_DATA);          // throw content
    // end of interrupt, enable interrupts
    send_eoi(RTC_IRQ);
}


// RTC Driver Interface
/* rtc_open
 * Description: Open RTC for specific process, set RTC freq to 2Hz
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: Always 0
 */
int32_t rtc_open(const uint8_t* filename){
    rtc_set_freq(2); // set to 2Hz by default
    rtc_info[cur_sched].rtc_int = 0;
    rtc_info[cur_sched].rtc_en = 1;
    return 0;
}

/* rtc_close
 * Description: Close RTC for specific process
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: Always 0
 */
int32_t rtc_close(int32_t fd){
    rtc_info[cur_sched].rtc_en = 0;
    return 0;
}

/* rtc_read
 * Description: Wait for interrupt and return when interrupt occurs
 * Inputs: None
 * Outputs: None
 * Side Effects: Reset rtc_int back to 0
 * Return Value: Always 0
 */
int32_t rtc_read(int32_t fd, void* buf, int32_t nbytes){
    while(!rtc_info[cur_sched].rtc_int);   // wait for interrupt
    rtc_info[cur_sched].rtc_int = 0;
    return 0;
}

/* rtc_write
 * Description: for cp2, freq must be power of 2
 * Inputs: fd - file descriptor
 *         buf - buffer containing a 4-byte integer frequency
 *         nbytes - number of bytes inside buffer
 * Outputs: None
 * Side Effects: None
 * Return Value: 0 is success, 1 if fail
 */
int32_t rtc_write(int32_t fd, const void* buf, int32_t nbytes){
    // parameter sanity check
    if( (buf==NULL) || (nbytes!=4) ){ return -1; }  // freq should be a 4-byte integer
    // extract freqeuncy
    int32_t freq = *(int32_t*)buf;
    // freq sanity check
    if( freq<RTC_MIN_FREQ || freq>RTC_MAX_FREQ ){ return -1; }
    if ( freq & (freq-1) ){ return -1; } // check if freq is power of 2, only for cp2
    // set frequency
    rtc_set_freq(freq);
    return 0;
}

/* rtc_set_freq
 * Description: set user programmable rtc frequency, in range of 0-1024Hz
 * Inputs: freq - user customized frequency, with unit Hz
 * Outputs: None
 * Side Effects: Set rtc_set_cnt
 * Return Value: None
 */
void rtc_set_freq(int32_t freq){
    rtc_info[cur_sched].rtc_set_cnt = RTC_MAX_FREQ / freq;
    rtc_info[cur_sched].rtc_cnt = rtc_info[cur_sched].rtc_set_cnt;
}

void rtc_speaker(){
    timer_speaker.rtc_en = 1;
    timer_speaker.rtc_int = 0;
    timer_speaker.rtc_cnt = 10; // approximately 10ms
    while(!timer_speaker.rtc_int);
    timer_speaker.rtc_en = 0;
}
