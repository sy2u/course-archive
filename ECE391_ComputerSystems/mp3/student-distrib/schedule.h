#ifndef _SCHEDULE_H
#define _SCHEDULE_H

#include "syscall.h"
#include "types.h"
#include "pit.h"
#include "terminal.h"

#define NUM_TERM    3
#define SCHED_NUM   3
#define NO_TERM     -2

void init_cur_sched();
void get_screen_pos_buf(uint16_t* x, uint16_t* y);
void set_screen_pos_buf(uint16_t x, uint16_t y);
int32_t terminal_switch(int tar_term);
int32_t scheduler();


#endif // _SCHEDULE_H
