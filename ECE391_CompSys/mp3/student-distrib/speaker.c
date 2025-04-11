/* speaker.c - Functions to play sound using PC speaker. */
#include "speaker.h"

/* middle C freqs:         do   re   mi   fa   so   la   si   do   re   mi */
uint32_t note_freq[10] = {262, 293, 330, 349, 392, 440, 494, 523, 587, 659};
extern uint8_t cur_term;
extern int32_t cur_sched;

/* keyboard_piano
 * Description: play the corresponding sound of the pressed keyboard
 * Inputs: input - keyboard input
 * Outputs: Speaker beeping sound
 * Side Effects: None
 * Return Value: None
 */
void keyboard_piano(uint16_t input){
	uint32_t cur_vidmap = get_vidmap();
	if( cur_sched != cur_term ){ set_vidmap(VRAM_ADDR);  set_kb_flag();  } // limit printf to current terminal
	// sanity check
	if( input < NUM_1 || input > NUM_0 ){
		printf("Not valid input. Please press numbers from 0 to 9!\n");
		if( cur_sched != cur_term ){ set_vidmap(cur_vidmap); clear_kb_flag();} // restore original paging
		return;
	}
	// play sound
	printf("Playing note: %d\n", input-1); // input "1" begins from 0x02
	if( cur_sched != cur_term ){ set_vidmap(cur_vidmap); clear_kb_flag();} // restore original paging
	// allow PIT interrupt for beeping
    sti();
    beep(note_freq[input-2]); // freq list start from 0
}

/* play_sound
 * Description: keep playing a sound using PC speaker.
 * Reference: osdev PC speaker (https://wiki.osdev.org/PC_Speaker)
 * Inputs: freq - the desired sound frequency
 * Outputs: Speaker beeping sound
 * Side Effects: None
 * Return Value: None
 */
void play_sound(uint32_t freq) {
 	uint32_t Div;
 	uint8_t tmp;
 
    //Set the PIT to the desired frequency
 	Div = 1193180 / freq;
 	outb(0xb6, 0x43);
 	outb((uint8_t) (Div), 0x42);
 	outb((uint8_t) (Div >> 8), 0x42);
 
    //And play the sound using the PC speaker
 	tmp = inb(0x61);
  	if (tmp != (tmp | 3)) { outb(tmp | 3, 0x61);    }
}

/* nosound
 * Description: shut PC speaker up
* Reference: osdev PC speaker (https://wiki.osdev.org/PC_Speaker)
 * Inputs: None
 * Outputs: None
 * Side Effects: None
 * Return Value: None
 */
void nosound() {
 	uint8_t tmp = inb(0x61) & 0xFC;
 	outb(tmp, 0x61);
}

/* beep
 * Description: make a single beep
 * Reference: osdev PC speaker (https://wiki.osdev.org/PC_Speaker)
 * Inputs: freq - the desired sound frequency
 * Outputs: Speaker beeping sound
 * Side Effects: None
 * Return Value: None
 */
void beep(uint32_t freq){
    play_sound(freq);
 	rtc_speaker(); // waiting time set to approximately 10ms
 	nosound();
}
