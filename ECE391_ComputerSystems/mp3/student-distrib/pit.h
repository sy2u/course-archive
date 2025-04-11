#ifndef _PIT_H
#define _PIT_H

#include "lib.h"
#include "i8259.h"
#include "cmos.h"
#include "mouse.h"

#define PIT_IRQ 0
// divide base frequency of 1.193182 MHz to roughly 10 Hz
// allowing ~10 ms for each time slice
#define FREQ_DIV 11930
#define PIT_CMD 0x43
#define CHANEL_0 0x40
// 0011 0110
// Channel 0 
// Access mode: lobyte/hibyte
// Operating mode : square wave generator
// 16-bit binary
#define PIT_MODE 0x36
// PIT Interrupt Interface
/* Initialize RTC */
extern void pit_init();
/* PIT Handler in IDT */
extern void pit_handler();
#endif 
