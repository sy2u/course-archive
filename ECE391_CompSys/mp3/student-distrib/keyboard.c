#include "keyboard.h"
#include "terminal.h"
#include "schedule.h"

extern terminal term_buf[NUM_TERM];
extern uint8_t cur_term;
extern int32_t cur_sched;

uint8_t cur_kernel_mode[NUM_TERM] = {SHELL_MODE, SHELL_MODE, SHELL_MODE};
uint8_t per_kernel_mode[NUM_TERM] = {SHELL_MODE, SHELL_MODE, SHELL_MODE};

// scan code table for look up
char scan_code_tab[NUM_SCAN][2] = { {0x0, 0x0}, {0x0, 0x0}, {'1', '!'}, {'2', '@'}, 
                                    {'3', '#'}, {'4', '$'}, {'5', '%'}, {'6', '^'},
                                    {'7', '&'}, {'8', '*'}, {'9', '('}, {'0', ')'},
                                    {'-', '_'}, {'=', '+'}, {BKS, BKS}, {TAB, TAB}, 
                                    {'q', 'Q'}, {'w', 'W'}, {'e', 'E'}, {'r', 'R'}, 
                                    {'t', 'T'}, {'y', 'Y'}, {'u', 'U'}, {'i', 'I'},
                                    {'o', 'O'}, {'p', 'P'}, {'[', '{'}, {']', '}'}, 
                                    {'\n', '\n'}, {0, 0},   {'a', 'A'}, {'s', 'S'}, 
                                    {'d', 'D'}, {'f', 'F'}, {'g', 'G'}, {'h', 'H'},
                                    {'j', 'J'}, {'k', 'K'}, {'l', 'L'}, {';', ':'}, 
                                    {'\'', '\"'}, {'`', '~'}, {0, 0},  {'\\', '|'}, 
                                    {'z', 'Z'}, {'x', 'X'}, {'c', 'C'}, {'v', 'V'}, 
                                    {'b', 'B'}, {'n', 'N'}, {'m', 'M'}, {',', '<'}, 
                                    {'.', '>'}, {'/', '?'}, {0x0, 0x0}, {0x0, 0x0}, 
                                    {0x0, 0x0}, {' ', ' '}};

int shift = 0, ctrl = 0, alt = 0, caps = 0, input_len = 0, f1[NUM_TERM] = {0,0,0};
/* keyboard_init
 * Description: Initialize keyboard, enable IRQ1
 * Inputs: None
 * Outputs: None
 * Side Effects: Modify PIC mask
 * Return Value: None
 */     
void keyboard_init(void) {
    input_len = 0;
    enable_irq(KB_IRQ);
}

/* check_special
 * Description: check special input
 * Inputs: input - input from keyboard
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
int check_special(uint32_t input) {
    switch (input) {
    case L_ALT:     alt = 1;                return 1;
    case L_ALT_R:   alt = 0;                return 1;
    case L_CTRL:    ctrl = 1;               return 1;
    case L_CTRL_R:  ctrl = 0;               return 1;
    case L_SHIFT:   shift = 1;              return 1;
    case L_SHIFT_R: shift = 0;              return 1;
    case R_SHIFT:   shift = 1;              return 1;
    case R_SHIFT_R: shift = 0;              return 1;
    case CAPS:      caps ^= 1;              return 1;
    case F1:        switch_speaker_mode();  return 1;
    default:                                return 0;
    } 
}

void clear_sreen() {
    uint32_t cur_vidmap = get_vidmap();
    set_vidmap(VRAM_ADDR);
    set_kb_flag();
    clear();
    input_len = 0;
    term_buf[cur_term].len = 0;
    memset(term_buf[cur_term].buf, 0, sizeof(term_buf[cur_term].buf));
    update_cursor(0, 0);
    set_vidmap(cur_vidmap); 
    clear_kb_flag();
}

/* keyboard_handler
 * Description: Handle keyboard interrupts and print out character. 
 *              Currently support all alphabet and numbers
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void keyboard_handler(void) {
    uint32_t cur_vidmap = get_vidmap();
    uint32_t input = inb(PS2_DATA), tmp = 0;

    // switch terminal
    if( alt && (input == F1) ){ terminal_switch(0);   send_eoi(KB_IRQ);   return; }
    if( alt && (input == F2) ){ terminal_switch(1);   send_eoi(KB_IRQ);   return; }
    if( alt && (input == F3) ){ terminal_switch(2);   send_eoi(KB_IRQ);   return; }

    // check functional keys
    if (check_special(input)){
        send_eoi(KB_IRQ);
        return;
    }

    // input history: cursor up & down key support
    if( input == UP || input == DOWN ){
        if( !f1[cur_term] ){
            if( input == UP ){ restore_history_up(); }
            if( input == DOWN ){ restore_history_down(); }
        }
        send_eoi(KB_IRQ);
        return;
    }

    // other required function keys
    if (input < 0 || input > NUM_SCAN || scan_code_tab[input][0] == 0) {
        send_eoi(KB_IRQ);
        return;
    }
    if (ctrl && scan_code_tab[input][0] == 'l') {
        clear_sreen();
        send_eoi(KB_IRQ);
        return;
    }
    if (ctrl || alt) {
        send_eoi(KB_IRQ);
        return;
    }

    // speaker demo: keyboard piano
    if (f1[cur_term]) {
        send_eoi(KB_IRQ);
        keyboard_piano(input);
        return;
    }

    // common keys
    if( cur_sched != cur_term ){ // limit printf to current terminal
        set_vidmap(VRAM_ADDR);
        set_kb_flag();
    }
    if( term_buf[cur_term].flag ){ // terminal read is waiting
        if (scan_code_tab[input][shift] == '\n'){ // end of terminal read
            // only save non-empty inputs
            if( term_buf[cur_term].len != 0 ){ save_history(); }
            term_buf[cur_term].read_idx = -1; // reset read count
            putc(scan_code_tab[input][shift]);
            term_buf[cur_term].buf[term_buf[cur_term].len++] = '\n';
            term_buf[cur_term].flag = 0;
        } else {
            if (scan_code_tab[input][0] == BKS) {
                if (term_buf[cur_term].len > 0) {
                    //if (term_buf[cur_term].buf[--term_buf[cur_term].len] == TAB){ for (i = 0; i < 3; i++){ backspace(); } } // delete extra 3 spaces for TAB
                    term_buf[cur_term].len--;
                    backspace();
                }
            } else if( term_buf[cur_term].len<127 ){ // input 127 chars for the max length, 1 reserved for '\n'
                if (scan_code_tab[input][0] >= 'a' && scan_code_tab[input][0] <= 'z'){
                    tmp ^= shift;
                    tmp ^= caps;
                    putc(scan_code_tab[input][tmp]);
                    term_buf[cur_term].buf[term_buf[cur_term].len++] = scan_code_tab[input][tmp];
                } else {
                    if (scan_code_tab[input][0] == TAB){ auto_complete(); } 
                    else { putc(scan_code_tab[input][shift]);
                        term_buf[cur_term].buf[term_buf[cur_term].len++] = scan_code_tab[input][shift];
                    }
                }
            }
        }
    } else { // DON"T allow keyboard input when no terminal read is issued
        // if (scan_code_tab[input][0] == BKS){
        //     if (input_len > 0) {
        //         backspace();
        //         input_len--;
        //     }
        // } else {
        //     if (scan_code_tab[input][0] >= 'a' && scan_code_tab[input][0] <= 'z'){
        //         tmp ^= shift;
        //         tmp ^= caps;
        //         putc(scan_code_tab[input][tmp]);
        //     } else if (scan_code_tab[input][0] == TAB){
        //         for (i = 0; i < 4; i++){ putc(' '); } 
        //         input_len = input_len + 3;
        //     } else { 
        //         putc(scan_code_tab[input][shift]);
        //     }
        //     input_len++;
        //     if (scan_code_tab[input][0] == '\n'){ input_len = 0; }
        // }
    }
    if( cur_sched != cur_term ){ 
        set_vidmap(cur_vidmap); 
        clear_kb_flag();
    }
    send_eoi(KB_IRQ);
}

/* switch_speaker_mode
 * Description: switch between speaker mode and original mode
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void switch_speaker_mode(){
    uint32_t cur_vidmap = get_vidmap();
    f1[cur_term] ^= 1;    
    if( cur_sched != cur_term ){ set_vidmap(VRAM_ADDR);  set_kb_flag();  }
    if(f1[cur_term]){
        per_kernel_mode[cur_term] = cur_kernel_mode[cur_term];
        cur_kernel_mode[cur_term] = SPEAKER_MODE;
        printf("Entering Keyboard Piano Mode.\n"); }
    else { 
        cur_kernel_mode[cur_term] = per_kernel_mode[cur_term];
        printf("Exit Keyboard Piano Mode.\n"); }
    if( cur_sched != cur_term ){ set_vidmap(cur_vidmap); clear_kb_flag();}
}
