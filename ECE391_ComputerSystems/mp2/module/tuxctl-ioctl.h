// All necessary declarations for the Tux Controller driver must be in this file

#ifndef TUXCTL_H
#define TUXCTL_H

#define TUX_SET_LED _IOR('E', 0x10, unsigned long)
#define TUX_READ_LED _IOW('E', 0x11, unsigned long*)
#define TUX_BUTTONS _IOW('E', 0x12, unsigned long*)
#define TUX_INIT _IO('E', 0x13)
#define TUX_LED_REQUEST _IO('E', 0x14)
#define TUX_LED_ACK _IO('E', 0x15)

// checkpoint 2
#define BUTTON_NUM      8
#define ACK_NONE        0
#define ACK_GET         1
#define LED_OFF			0x00
#define LED_CMD_LENGTH	6
#define LED_NUM         4
#define BITMASK         0x01
#define LED_DP_MASK     0x10
#define LED_SEG_BASE    2
#define bit_MASK_3_0    0x0F

int tux_set_led(struct tty_struct* tty, unsigned long arg);
int tux_buttons(struct tty_struct* tty, unsigned long arg);
int tux_init(struct tty_struct* tty);
void tux_reset(struct tty_struct* tty);

#endif

