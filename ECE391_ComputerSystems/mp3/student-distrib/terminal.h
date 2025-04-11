#ifndef _TERMINAL_H
#define _TERMINAL_H

#include "lib.h"
#include "keyboard.h"

#define BUF_SIZE    128
#define HIST_DEPTH  5

typedef struct history{
    char buf[HIST_DEPTH][BUF_SIZE];
    int len[HIST_DEPTH];
    uint8_t hpos; // history buffer head position
    int8_t entry_cnt; // history total num count
} history_t;

typedef struct terminal{
    char buf[BUF_SIZE];
    history_t hist;
    int8_t read_idx;
    int len;
    int flag;
    int screen_x;
    int screen_y;
} terminal;

int32_t terminal_open (const uint8_t* filename);
int32_t terminal_close (int32_t fd);
int32_t terminal_read (int32_t fd, void* buf, int32_t nbytes);
int32_t terminal_write (int32_t fd, const void* buf, int32_t nbytes);

int32_t illegal_open (const uint8_t* filename);
int32_t illegal_close (int32_t fd);
int32_t illegal_read (int32_t fd, void* buf, int32_t nbytes);
int32_t illegal_write (int32_t fd, const void* buf, int32_t nbytes);

#endif
