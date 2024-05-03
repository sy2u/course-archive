#ifndef MOUSE_H
#define MOUSE_H

#include "i8259.h"
#include "types.h"
#include "lib.h"
#include "keyboard.h"

#define MOUSE_IRQ   (12)
#define PS2_DATA    (0x60)
#define GET_CPQ     (0x20)
#define AUX_ENABLE  (0xA8)
#define AUX_PORT    (0x64)
#define SET_DEFAULT (0xF6)
#define PKT_ENABLE  (0xF4)
#define CMD_BYTE    (0xD4)
#define BIT5_MASK   (0xDF)
#define SET_CPQ     (0x60)

typedef struct mouse{
    int8_t cur_x;
    int8_t cur_y;
    int8_t pre_x;
    int8_t pre_y;
    int8_t update_flag;
} mouse_t;

void mouse_init(void);
void mouse_handler(void);
void update_mouse();
int8_t mouse_read(void);
void mouse_write(unsigned char out);
void mouse_wait(char type);

#endif
