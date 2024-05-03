#include "mouse.h"
#include "history.h"
#include "keyboard.h"

#define NUM_COLS    80
#define NUM_ROWS    25

int cnt = 0; // use FSM to handle 3 consecutive package
int8_t mouse_pkts[3];
mouse_t mouse_info;

/* mouse_wait
 * Inputs: type - read or write, 0 for write 1 for read
 * Return Value: none
 * Function: wait until read for ports */
void mouse_wait(char type) {
    // 0 for output to port 0x60 or 0x64
    int poll_time = 10000;
    if (type == 0) {
        while (poll_time--)
            if ((inb(0x64) & 2) == 0) return;
    }
    
    // 1 for read from port 0x60
    else if (type == 1) {
        while (poll_time--)
            if (inb(0x64) & 1) return;
    }
}

/* mouse_read
 * Inputs: none
 * Return Value: read from mouse
 * Function: read from mouse */
int8_t mouse_read(void) {
    mouse_wait(1);
    return inb(PS2_DATA);
}

/* mouse_write
 * Inputs: out - out byte to mouse
 * Return Value: none
 * Function: write to mouse */
void mouse_write(uint8_t out) {
    mouse_wait(0);
    outb(CMD_BYTE, 0x64);
    mouse_wait(0);
    outb(out, PS2_DATA);
}

/* mouse_init
 * Inputs: none
 * Return Value: none
 * Function: initialize the mouse */
void mouse_init(void) {
    char status;

    // init struct
    mouse_info.cur_x = 0;
    mouse_info.cur_y = 0;
    mouse_info.pre_x = 0;
    mouse_info.pre_y = 0;
    mouse_info.update_flag = 0;

    // read compaq status from port 0x60 and modify
    outb(GET_CPQ, 0x64);
    status = mouse_read();
    status |= 2;
    status &= BIT5_MASK;
    outb(SET_CPQ, 0x64);
    outb(status, PS2_DATA);
    mouse_read();

    // enable auxiliary device
    outb(AUX_ENABLE, 0x64);
    mouse_read();

    // set defaults
    mouse_write(SET_DEFAULT);
    mouse_read();

    // enable packet streaming
    mouse_write(PKT_ENABLE);
    mouse_read();

    // enable irq 12
    enable_irq(MOUSE_IRQ);
}

/* mouse_handler
 * Inputs: none
 * Return Value: none
 * Function: handle mouse interrupts */
void mouse_handler(void) {

    send_eoi(MOUSE_IRQ);
    
    switch (cnt) {
    case 0: 
        mouse_pkts[0] =  mouse_read();
        // invalid mouse read
        if( (!(mouse_pkts[0]&0x08)) || (mouse_pkts[0] & 0xC0)) return; 
        cnt++;
        return;
    case 1: mouse_pkts[1] = mouse_read(); cnt++; return;
    case 2: mouse_pkts[2] = mouse_read();
        // handle mouse movement
        mouse_info.cur_x += mouse_pkts[1] / 10;
        mouse_info.cur_y -= mouse_pkts[2] / 10;
        if (mouse_info.cur_x < 0) mouse_info.cur_x = 0;
        if (mouse_info.cur_x >= NUM_COLS) mouse_info.cur_x = NUM_COLS - 1;
        if (mouse_info.cur_y < 0) mouse_info.cur_y = 0;
        if (mouse_info.cur_y >= NUM_ROWS - 1) mouse_info.cur_y = NUM_ROWS - 2; // not going into status bar
        cnt = 0;
        mouse_info.update_flag = 1; // need update on screen
        // left click to scroll up
        if (mouse_pkts[0] & 1) restore_history_up();
        // right click to scroll down
        if (mouse_pkts[0] & (1<<1)) restore_history_down();
        // mid button to clear screen
        if (mouse_pkts[0] & (1<<2)) clear_sreen();
        // update_mouse();
        return;
    default: cnt = 0; return;
    }

}

/* update_mouse
 * Inputs: none
 * Return Value: none
 * Function: update mouse position */
void update_mouse(){
    reset_attrib(mouse_info.pre_x, mouse_info.pre_y);
    mouse_info.pre_x = mouse_info.cur_x;
    mouse_info.pre_y = mouse_info.cur_y;
    set_attrib(mouse_info.cur_x, mouse_info.cur_y);
    mouse_info.update_flag = 0; // update finished
}
