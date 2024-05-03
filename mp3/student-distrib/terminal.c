#include "terminal.h"
#include "schedule.h"

#define min(a, b) (a<b?a:b)

terminal term_buf[NUM_TERM];
extern int32_t cur_sched;

/* terminal_open
 * Description: Initialize variables for terminal
 * Inputs: filename - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
int32_t terminal_open (const uint8_t* filename) {
    int i;
    term_buf[cur_sched].flag = 0;
    memset(term_buf[cur_sched].buf, 0, sizeof(term_buf[cur_sched].buf));
    term_buf[cur_sched].len = 0;
    term_buf[cur_sched].hist.hpos = 0;
    term_buf[cur_sched].hist.entry_cnt = 0;
    for( i = 0; i < HIST_DEPTH; i++ ){
        term_buf[cur_sched].hist.len[i] = 0;
        memset(term_buf[cur_sched].hist.buf, 0, BUF_SIZE*sizeof(char)); 
    }
    term_buf[cur_sched].read_idx = -1; // points to most recent read entry
    return 0;
}

/* terminal_close
 * Description: Nothing
 * Inputs: fd - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: -1
 */
int32_t terminal_close (int32_t fd) {
    return -1;
}

/* terminal_read
 * Description: read from keyboard buffer to a specific bufffer
 * Inputs: fd - ignored
 *         buf - place to hold reading
 *         nbytes - number of bytes read
 * Outputs: None
 * Side Effects: None
 * Return Value: the actual length read from keyboard buffer
 */
int32_t terminal_read (int32_t fd, void* buf, int32_t nbytes) {
    // sanity check
    if( buf==NULL ){ return 0; }
    // read from kb buffer
    term_buf[cur_sched].flag = 1;
    while (term_buf[cur_sched].flag==1);
    int len = min(nbytes, term_buf[cur_sched].len);
    memcpy(buf, term_buf[cur_sched].buf, len);
    if( ((uint8_t*)buf)[len-1]!='\n'){ ((uint8_t*)buf)[len-1]='\n'; }
    // reset kb buffer
    term_buf[cur_sched].flag = 0;
    memset(term_buf[cur_sched].buf, 0, sizeof(term_buf[cur_sched].buf));
    term_buf[cur_sched].len = 0;
    return len;
}

/* terminal_write
 * Description: write from buffer to screen
 * Inputs: fd - ignored
 *         buf - place to hold reading
 *         nbytes - number of bytes read
 * Outputs: None
 * Side Effects: None
 * Return Value: the actual length written to screen
 */
int32_t terminal_write (int32_t fd, const void* buf, int32_t nbytes) {
    int i, j;
    // sanity check
    if( buf==NULL ){ return 0; }
    // write to VRAM
    for (i = 0; i < nbytes; i++){
        uint8_t text = ((uint8_t*)buf)[i];
        if( text != '\0' ){ // not print NUL char
            if( text == TAB ){ for( j = 0; j < 4; j++) putc(' '); }
            else{ putc(text); }
        }
    }
    return nbytes;
}

/* illegal_open
 * Description: do nothing for stdin/stdout
 * Inputs: filename - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
int32_t illegal_open (const uint8_t* filename) {
    return -1;
}

/* illegal_close
 * Description: do nothing for stdin/stdout
 * Inputs: fd - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: -1
 */
int32_t illegal_close (int32_t fd) {
    return -1;
}

/* illegal_read
 * Description: do nothing for stdout
 * Inputs: fd - ignored
 *         buf - ignored
 *         nbytes - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: -1
 */
int32_t illegal_read (int32_t fd, void* buf, int32_t nbytes) {
    return -1;
}

/* illegal_write
 * Description: do nothing for stdin
 * Inputs: fd - ignored
 *         buf - ignored
 *         nbytes - ignored
 * Outputs: None
 * Side Effects: None
 * Return Value: -1
 */
int32_t illegal_write (int32_t fd, const void* buf, int32_t nbytes) {
    return -1;
}
