/* tuxctl-ioctl.c
 *
 * Driver (skeleton) for the mp2 tuxcontrollers for ECE391 at UIUC.
 *
 * Mark Murphy 2006
 * Andrew Ofisher 2007
 * Steve Lumetta 12-13 Sep 2009
 * Puskar Naha 2013
 */

#include <asm/current.h>
#include <asm/uaccess.h>

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/file.h>
#include <linux/miscdevice.h>
#include <linux/kdev_t.h>
#include <linux/tty.h>
#include <linux/spinlock.h>

#include "tuxctl-ld.h"
#include "tuxctl-ioctl.h"
#include "mtcp.h"

#define debug(str, ...) \
	//printk(KERN_DEBUG "%s: " str, __FUNCTION__, ## __VA_ARGS__)

// checkpoint 2
int buttons;
int tux_ack;
unsigned char led_cmd[LED_CMD_LENGTH]; // global led status, reserved for reset
static spinlock_t tuxctl_ldisc_lock = SPIN_LOCK_UNLOCKED;


/************************ Protocol Implementation *************************/

/* tuxctl_handle_packet()
 * IMPORTANT : Read the header for tuxctl_ldisc_data_callback() in 
 * tuxctl-ld.c. It calls this function, so all warnings there apply 
 * here as well.
 */
void tuxctl_handle_packet (struct tty_struct* tty, unsigned char* packet)
{
    unsigned a, b, c;
	unsigned long flags;

    a = packet[0]; /* Avoid printk() sign extending the 8-bit */
    b = packet[1]; /* values when printing them. */
    c = packet[2];


	switch(a) {
		case MTCP_RESET: return tux_reset(tty);
		case MTCP_BIOC_EVENT:
		// Button  +----7--+--6---+--5---+-4--+-3-+-2-+-1-+---0---+
		//		   | right | left | down | up | c | b | a | start |
		//		   +----7--+--6---+--5---+-4--+-3-+-2-+-1-+---0---+
		// byte 1  +-7-----4-+-3-+-2-+-1-+---0---+
		//		   | 1 X X X | C | B | A | START |
		//		   +---------+---+---+---+-------+
		// byte 2  +-7-----4-+---3---+--2---+--1---+-0--+
		//		   | 1 X X X | right | down | left | up |
		// 		   +---------+-------+------+------+----+
			spin_lock_irqsave(&tuxctl_ldisc_lock, flags);
			buttons = 0;
			buttons |= b & bit_MASK_3_0;	// c, b, a, start
			buttons |= (c & 0x09) << 4;		// right, up
			buttons |= (c & 0x04) << 3;		// down
			buttons |= (c & 0x02) << 5;		// left
			buttons = ~buttons;				// buttons are active low
			spin_unlock_irqrestore(&tuxctl_ldisc_lock, flags);
			break;
		case MTCP_ACK: tux_ack = ACK_GET; break;
		default: return;
	}
}

/******** IMPORTANT NOTE: READ THIS BEFORE IMPLEMENTING THE IOCTLS ************
 *                                                                            *
 * The ioctls should not spend any time waiting for responses to the commands *
 * they send to the controller. The data is sent over the serial line at      *
 * 9600 BAUD. At this rate, a byte takes approximately 1 millisecond to       *
 * transmit; this means that there will be about 9 milliseconds between       *
 * the time you request that the low-level serial driver send the             *
 * 6-byte SET_LEDS packet and the time the 3-byte ACK packet finishes         *
 * arriving. This is far too long a time for a system call to take. The       *
 * ioctls should return immediately with success if their parameters are      *
 * valid.                                                                     *
 *                                                                            *
 ******************************************************************************/
/* tuxctl_ioctl
 * DESCRIPTION: Wrapper function for tux ioctl functions calling
 * INPUTS: tty - line descriptor
 * 		   cmd - specific ioctl function to be called
 * 		   arg - argument for BUTTONS and SET_LED
 * OUTPUTS: none
 * RETURN VALUE: 0 for success, -EINVAL for fail
 */
int 
tuxctl_ioctl (struct tty_struct* tty, struct file* file, 
	      unsigned cmd, unsigned long arg)
{
    switch (cmd) {
	case TUX_INIT:		return tux_init(tty);
	case TUX_BUTTONS:	return tux_buttons(tty, arg);
	case TUX_SET_LED:	
		if(tux_ack == ACK_GET){		// call set_led only when last call is finished
			tux_ack = ACK_NONE;
			return tux_set_led(tty, arg);
		} else { return 0; }
	case TUX_LED_ACK:
	case TUX_LED_REQUEST:
	case TUX_READ_LED:
	default:
	    return -EINVAL;
    }
}

/* tux_init
 *   DESCRIPTION: Helper function to initialize TUX hardware & driver
 *   INPUTS: tty - line descriptor
 *   OUTPUTS: Send command package to tux driver
 *   RETURN VALUE: 0 if initialized successfully
 */
int tux_init(struct tty_struct* tty)
{
	unsigned char cmd[2] = {MTCP_BIOC_ON, MTCP_LED_USR};
	tuxctl_ldisc_lock = SPIN_LOCK_UNLOCKED;
	tux_ack = ACK_NONE;
	tuxctl_ldisc_put(tty, cmd, 2);	// 2 byte to be sent
	return 0;
};

/* tux_buttons
 *   DESCRIPTION: Helper function to send button data from kernel to user level
 *   INPUTS: tty - line descriptor
 * 			 arg - a int pointer in user space
 *   OUTPUTS: None
 *   RETURN VALUE: 0 if arg is valid and copy is successful
 */
int tux_buttons(struct tty_struct* tty, unsigned long arg){
	unsigned long flags;
	if( arg == 0 ){ return -EINVAL; }	// check NULL pointer
	spin_lock_irqsave(&tuxctl_ldisc_lock, flags);
	copy_to_user((int*)arg, &buttons, sizeof(buttons));
	spin_unlock_irqrestore(&tuxctl_ldisc_lock, flags);	// no need to check success as indicated in document
	return 0;
	
}

/* 	 tux_set_led
 *   DESCRIPTION: Helper function to convert target number into segment format and send the data to tux
 *   INPUTS: tty - line descriptor
 * 			 arg - a 32-bit int containing LED parameters
 *   OUTPUTS: None
 *   RETURN VALUE: 0 when finish
 */
int tux_set_led(struct tty_struct* tty, unsigned long arg){
	int i;
	int bitmask = BITMASK;
	unsigned long flags;

	unsigned char LED_pattern[16] = {0xE7, 0x06, 0xCB, 0x8F, 0x2E, 0xAD, 0xED, 0xA6, 
									 0xEF, 0xAF, 0xEE, 0x6D, 0xE1, 0x4F, 0xE9, 0xE8};	// LED segments for number 0-16 in hex
	
	unsigned char led_enable = (arg>>16) & bit_MASK_3_0;	// LED enablt at 16:19
	unsigned char dp_idx = (arg>>24) & bit_MASK_3_0;		// decimal points enable at 24:27

	spin_lock_irqsave(&tuxctl_ldisc_lock, flags);
	led_cmd[0] = MTCP_LED_SET;
	led_cmd[1] = bit_MASK_3_0;
	spin_unlock_irqrestore(&tuxctl_ldisc_lock, flags);

	for(i = 0; i < LED_NUM; i++){
		unsigned char led_seg = LED_OFF;	// turn off all leds if not enabled
		if(led_enable & bitmask){ led_seg = LED_pattern[(arg>>(i*4)) & bit_MASK_3_0]; }	// 4bit per LED
		if(dp_idx & bitmask){ led_seg |= LED_DP_MASK; } // check and set dp 
		spin_lock_irqsave(&tuxctl_ldisc_lock, flags);
		led_cmd[LED_SEG_BASE + i] = led_seg;
		spin_unlock_irqrestore(&tuxctl_ldisc_lock, flags);
		bitmask = bitmask << 1;				
	}

	tuxctl_ldisc_put(tty, led_cmd, LED_CMD_LENGTH);
	
	return 0;
}

/* 	 tux_reset
 *   DESCRIPTION: Helper function to reset tux (init again while keep the led value),
				  called by tuxctl_handle_package
 *   INPUTS: tty - line descriptor
 *   OUTPUTS: None
 *   RETURN VALUE: none
 */
void tux_reset(struct tty_struct* tty){
	unsigned long flags;
	unsigned char cmd[2] = {MTCP_LED_USR, MTCP_BIOC_ON};
	tuxctl_ldisc_put(tty, cmd, 2);	// 2 bytes to be sent
	if( tux_ack == ACK_GET ){ 
		tux_ack = ACK_NONE;
		spin_lock_irqsave(&tuxctl_ldisc_lock, flags);
		tuxctl_ldisc_put(tty, led_cmd, LED_CMD_LENGTH);
		spin_unlock_irqrestore(&tuxctl_ldisc_lock, flags);
	}
	return;
}
