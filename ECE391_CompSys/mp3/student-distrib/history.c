#include "history.h"
#include "schedule.h"

extern terminal term_buf[NUM_TERM];
extern uint8_t cur_term;

/* save_history
 * Description: save current input to history buffer
 * Reference: ECE391 Lecture 6 - Ring Buffer
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void save_history(){
    uint8_t tpos = (term_buf[cur_term].hist.hpos + term_buf[cur_term].hist.entry_cnt) % HIST_DEPTH; // next position to be inserted
    if(term_buf[cur_term].hist.entry_cnt == HIST_DEPTH){
        term_buf[cur_term].hist.hpos = (term_buf[cur_term].hist.hpos + 1) % HIST_DEPTH;
    } else {
        term_buf[cur_term].hist.entry_cnt++;
    }
    term_buf[cur_term].hist.len[tpos] = term_buf[cur_term].len;
    memcpy(term_buf[cur_term].hist.buf[tpos], term_buf[cur_term].buf, BUF_SIZE);
}

/* restore_history_up
 * Description: restore previous input with "up" pressed
 * Inputs: None
 * Outputs: write previous input to terminal
 * Side Effects: None
 * Return Value: None
 */
void restore_history_up(){
    int i;
    uint8_t tpos;
    uint32_t cur_vidmap = get_vidmap();
    // sanity check
    if( term_buf[cur_term].read_idx == term_buf[cur_term].hist.entry_cnt - 1 ){ return; } // no more history
    // update read_count
    term_buf[cur_term].read_idx++;
    // clear all original outputs
    set_vidmap(VRAM_ADDR);
    set_kb_flag();
    while(term_buf[cur_term].len > 0){
        if (term_buf[cur_term].buf[--term_buf[cur_term].len] == TAB){ for (i = 0; i < 3; i++){ backspace(); } }
        backspace();
    }
    // put into current buffer
    tpos = (term_buf[cur_term].hist.hpos + term_buf[cur_term].hist.entry_cnt - term_buf[cur_term].read_idx - 1) % HIST_DEPTH;
    memcpy(term_buf[cur_term].buf, term_buf[cur_term].hist.buf[tpos], BUF_SIZE);
    term_buf[cur_term].len = term_buf[cur_term].hist.len[tpos];
    // output to current terminal
    for( i = 0; i < term_buf[cur_term].len; i++ ){ putc(term_buf[cur_term].buf[i]); }
    set_vidmap(cur_vidmap); 
    clear_kb_flag();
}

/* void restore_history_down
 * Description: restore previous input with "down" pressed
 * Inputs: None
 * Outputs: write previous history to terminal
 * Side Effects: None
 * Return Value: None
 */
void restore_history_down(){
    int i;
    uint8_t tpos;
    uint32_t cur_vidmap = get_vidmap();
    // sanity check
    if( term_buf[cur_term].read_idx < 0 ){ return; } // no more history
    // update read_count
    term_buf[cur_term].read_idx--;
    // clear all original outputs
    set_vidmap(VRAM_ADDR);
    set_kb_flag();
    while(term_buf[cur_term].len > 0){
        if (term_buf[cur_term].buf[--term_buf[cur_term].len] == TAB){ for (i = 0; i < 3; i++){ backspace(); } }
        backspace();
    }
    if( term_buf[cur_term].read_idx == -1 ){ set_vidmap(cur_vidmap); clear_kb_flag();    return; }
    // put into current buffer
    tpos = (term_buf[cur_term].hist.hpos + term_buf[cur_term].hist.entry_cnt - term_buf[cur_term].read_idx - 1) % HIST_DEPTH;
    memcpy(term_buf[cur_term].buf, term_buf[cur_term].hist.buf[tpos], BUF_SIZE);
    term_buf[cur_term].len = term_buf[cur_term].hist.len[tpos];
    // output to current terminal
    for( i = 0; i < term_buf[cur_term].len; i++ ){ putc(term_buf[cur_term].buf[i]); }
    set_vidmap(cur_vidmap); 
    clear_kb_flag();
}
