/* cmos.c - driver for CMOS to get cuurent time */
#include "cmos.h"
 
unsigned char second, minute, hour, day, month;
unsigned int year;

/* get_update_in_progress_flag
 * Description: Get flag indicating if RTC is updating time now
 * Reference: osdev CMOS (https://wiki.osdev.org/CMOS)
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
int get_update_in_progress_flag() {
    outb(0x0A, CMOS_ADDR);
    return (inb(CMOS_DATA) & 0x0080); // update flag at bit 7 of Status Register A
}

/* get_rtc_reg
 * Description: get value from the target register
 * Reference: osdev CMOS (https://wiki.osdev.org/CMOS)
 * Inputs: reg - register to be read
 * Outputs: None
 * Side Effects: None
 * Return Value: the content of the target register
 */
unsigned char get_rtc_reg(int reg) {
    outb(reg, CMOS_ADDR);
    return inb(CMOS_DATA); // only reserve lower eight bytes
}

/* read_rtc
 * Description: read current date and time
 * Reference: osdev CMOS (https://wiki.osdev.org/CMOS)
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void read_rtc() {
    // Didn't find how to enable "update interrupt", so using the "read registers until you get the same values twice in a row" technique
    unsigned char last_second, last_minute, last_hour, last_day, last_month, last_year, regB;

    while (get_update_in_progress_flag()); // Make sure an update isn't in progress
    // all register address below are specified by osdev
    second = get_rtc_reg(0x00);
    minute = get_rtc_reg(0x02);
    hour = get_rtc_reg(0x04);
    day = get_rtc_reg(0x07);
    month = get_rtc_reg(0x08);
    year = get_rtc_reg(0x09);
    do { // check consistency between consecutive two reads
        last_second = second; last_minute = minute; last_hour = hour; last_day = day; last_month = month; last_year = year;
        while (get_update_in_progress_flag()); 
        // all register address below are specified by osdev
        second = get_rtc_reg(0x00);
        minute = get_rtc_reg(0x02);
        hour = get_rtc_reg(0x04);
        day = get_rtc_reg(0x07);
        month = get_rtc_reg(0x08);
        year = get_rtc_reg(0x09);
    } while( (last_second != second) || (last_minute != minute) || (last_hour != hour) || (last_day != day) || (last_month != month) || (last_year != year) );

    // control the format of bytes
    regB = get_rtc_reg(0x0B);

    // Status Register B, Bit 2 (value = 4): Enables Binary mode if set
    if (!(regB & 0x04)) {
        second = (second & 0x0F) + ((second / 16) * 10);
        minute = (minute & 0x0F) + ((minute / 16) * 10);
        hour = ( (hour & 0x0F) + (((hour & 0x70) / 16) * 10) ) | (hour & 0x80) ;
        day = (day & 0x0F) + ((day / 16) * 10);
        month = (month & 0x0F) + ((month / 16) * 10);
        year = (year & 0x0F) + ((year / 16) * 10);
    }

    // Status Register B, Bit 1 (value = 2): Enables 24 hour format if set
    if (!(regB & 0x02) && (hour & 0x80)) { hour = ((hour & 0x7F) + 12) % 24; }

    // Calculate the full (4-digit) year
    year += (CURRENT_YEAR / 100) * 100;
    if(year < CURRENT_YEAR) year += 100;

    // adjust time zone to UTC-5
    if( hour < 5 ){
        hour += 19; 
        if( day!= 1 ){
            day -= 1;
        } else {
            switch (month) {
            case 1: month = 12; day=31; break;
            case 2: month = 1; day=31; break;
            case 3: month = 2; if( year % 4 == 0 ){ day = 29; }else{ day = 28; }    break;
            case 4: month = 3; day=31; break;
            case 5: month = 4; day=30; break;
            case 6: month = 5; day=31; break;
            case 7: month = 6; day=30; break;
            case 8: month = 7; day=31; break;
            case 9: month = 8; day=31; break;
            case 10: month = 9; day=30; break;
            case 11: month = 10; day=31; break;
            case 12: month = 11; day=30; break;
            default: break;
            }
        }
    } else { hour -= 5; }
}
