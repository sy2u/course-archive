#include "tab.h"
#include "schedule.h"

extern terminal term_buf[NUM_TERM];
extern uint8_t cur_term;
extern bootBlock_t* boot_ptr;

uint8_t* match_list[MAX_COMPLETIONS];

void auto_complete(){
    int i, j = 0;
    uint8_t tab_buf[MAX_ARGLEN];
    uint8_t* completion = NULL;

    memset(tab_buf, '\0', MAX_ARGLEN);

    for( i = 0; i < term_buf[cur_term].len; i++ ){
        if(term_buf[cur_term].buf[i] == ' '){
            memset(tab_buf, '\0', MAX_ARGLEN);
            j = 0;
        } else {
            tab_buf[j++] = term_buf[cur_term].buf[i];
        }
    }

    completion = find_completion(tab_buf);
    if (completion != NULL) {
        fill_term(completion, j);
    } else {
        for (i = 0; i < 4; i++){ putc(' '); } 
        term_buf[cur_term].len += 4;
    }
    
}

// Function to find the best match for the given input
uint8_t* find_completion(const uint8_t* input) {
    int i, num = 0, tar_idx;
    int min_length = MAX_ARGLEN;

    for(i = 0; i < boot_ptr->num_entries; i ++){
        // find all match dentries
        uint8_t* dname_ptr = boot_ptr->dentries[i].file_name;
        if( strlen((int8_t*)input) > strlen((int8_t*)dname_ptr) ){ continue; }
        if( strncmp((int8_t*)input, (int8_t*)dname_ptr, strlen((int8_t*)input)) == 0 ){
            match_list[num++] = dname_ptr;
        }
    }
    if( num == 0 ){ return NULL; }
    if( num == 1 ){ return match_list[0]; }

    // find the name with minimum length
    for( i = 0; i < num; i++ ){
        if( strlen((int8_t*)match_list[i]) < min_length ){
            min_length = strlen((int8_t*)match_list[i]);
            tar_idx = i;
        }
    }

    return match_list[tar_idx];
}

// prints string to terminal and adds it to the buffer
void fill_term(uint8_t* string, int input_len){
    int i, j, len;

    len = (strlen((int8_t*)string) < MAX_ARGLEN) ? strlen((int8_t*)string) : MAX_ARGLEN;

    // Erase the current input
    for (i = 0; i < input_len; i++) {
        backspace();
        term_buf[cur_term].len--;
    }

    for (j = 0; j < len; j++) {
        putc(string[j]);  // Output the character to the terminal
        term_buf[cur_term].buf[term_buf[cur_term].len++] = string[j];
    }
}
