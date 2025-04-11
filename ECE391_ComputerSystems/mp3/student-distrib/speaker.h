#ifndef _SPEAKER_H
#define _SPEAKER_H

#include "types.h"
#include "rtc.h"

#define NUM_1       (0x02)
#define NUM_0       (0x0B)

void play_sound(uint32_t freq);
void nosound();
void beep(uint32_t freq);
void keyboard_piano(uint16_t input);

#endif //_SPEAKER_H
