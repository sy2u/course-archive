#ifndef _CMOS_H
#define _CMOS_H

#include "types.h"
#include "lib.h"

#define CURRENT_YEAR    2024
#define CMOS_ADDR       0x70
#define CMOS_DATA       0x71

int get_update_in_progress_flag();
unsigned char get_rtc_reg(int reg);
void read_rtc();

#endif //_CMOS_H
