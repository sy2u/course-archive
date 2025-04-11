#include "power.h"

/* reboot
 * Description: Reboot the system
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void reboot(){
    // Use the 8042 keyboard controller to pulse the CPU's RESET pin
    uint8_t good = 0x02;
    while (good & 0x02)
        good = inb(0x64);
    outb(0xFE, 0x64);
    asm volatile ("hlt");
}

/* shutdown
 * Description: shutdown the terminal
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void shutdown(){
    outw(0x2000, 0xB004);
}
