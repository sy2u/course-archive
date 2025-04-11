#ifndef _RTC_H
#define _RTC_H

#include "lib.h"
#include "i8259.h"

// RTC Interrupt Interface
typedef struct {
    uint32_t rtc_cnt;
    uint32_t rtc_set_cnt;
    uint8_t  rtc_int;
    uint8_t  rtc_en;
} rtc_t;

/* RTC Handler in IDT */
extern void rtc_hdl();
/* Initialize RTC */
extern void rtc_init();

// RTC Driver Interface
void rtc_set_freq(int32_t freq);
extern int32_t rtc_open(const uint8_t* filename);
extern int32_t rtc_close(int32_t fd);
extern int32_t rtc_read(int32_t fd, void* buf, int32_t nbytes);
extern int32_t rtc_write(int32_t fd, const void* buf, int32_t nbytes);

// speaker timer
void rtc_speaker();

/* Control Registers and Ports */
#define REG_A           0x8A
#define REG_B           0x8B
#define REG_C           0x0C
#define REG_PORT        0x70
#define REG_DATA        0x71
/* Masks */
#define MASK_BIT_6      0x40
#define MAST_LOW_4      0xF0
/* RTC Parameter */
#define RTC_IRQ         0x08
#define RTC_RATE        0x06    // FIXED, set to maximum for virtualization - 1024Hz
#define RTC_MAX_FREQ    1024
#define RTC_MIN_FREQ    1

#endif /* _RTC_H */
