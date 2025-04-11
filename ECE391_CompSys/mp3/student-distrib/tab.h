#ifndef _TAB_H
#define _TAB_H

#include "types.h"
#include "terminal.h"

#define MAX_COMPLETIONS 16

uint8_t* find_completion(const uint8_t* input);
void fill_term(uint8_t* string, int input_len);
void auto_complete();

#endif
