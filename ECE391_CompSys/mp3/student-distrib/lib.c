/* lib.c - Some basic library functions (printf, strlen, etc.)
 * vim:ts=4 noexpandtab */

#include "lib.h"
#include "schedule.h"

extern terminal term_buf[NUM_TERM];
extern unsigned char second, minute, hour, day, month;
extern unsigned int year;
extern uint8_t cur_kernel_mode[NUM_TERM];
extern int32_t cur_sched;
extern uint8_t cur_term;
extern mouse_t mouse_info;

int keyboard_flag = 0;
int status_bar_flag = 0;
static int back_color = 0;
static int front_color = 0x07;

#define VIDEO       0xB8000
#define NUM_COLS    80
#define NUM_ROWS    (25-1) // reserve one row for status bar
#define ATTRIB      (front_color + back_color)
#define ATTRIB_REV  (0x70) // high light remove step

#define VGA_PORT_1  0x3D4
#define VGA_PORT_2  0x3D5

#define ST_SCAN     0x0A
#define ED_SCAN     0x0B
#define DISAB       0x20
#define LOWER       0x0F
#define UPPER       0x0E

static char* video_mem = (char *)VIDEO;

/* reset_attrib
 * Inputs: mouse_x, mouse_y - current mouse location
 * Return Value: none
 * Function: reset current mouse location to normal attribute */
void reset_attrib(int mouse_x, int mouse_y) {
    *(uint8_t *)(VIDEO + ((NUM_COLS * mouse_y + mouse_x) << 1) + 1) = ATTRIB;
}

/* set_attrib
 * Inputs: mouse_x, mouse_y - current mouse location
 * Return Value: none
 * Function: reset current mouse location to reverse attribute */
void set_attrib(int mouse_x, int mouse_y) {
    *(uint8_t *)(VIDEO + ((NUM_COLS * mouse_y + mouse_x) << 1) + 1) = ATTRIB_REV;
}

/* update_status_bar
 * Inputs: void
 * Return Value: none
 * Function: Output current status to the status bar */
void update_status_bar(){
    int i;
    char *month_msg = "", *suffix = "", *mode_msg = "";
    // save original setting
    cli();
    int front_color_save = front_color;
    int back_color_save = back_color;
    int orig_x = term_buf[cur_term].screen_x;
    int orig_y = term_buf[cur_term].screen_y;
    // set flag
    status_bar_flag = 1;
    // set white background and black text
    front_color = 0x00;
    back_color = 0x70;
    // make sure printing to current terminal
    switch (month) {
        case 1:     month_msg = "January";      break;
        case 2:     month_msg = "February";     break;
        case 3:     month_msg = "March";        break;
        case 4:     month_msg = "April";        break;
        case 5:     month_msg = "May";          break;
        case 6:     month_msg = "June";         break;
        case 7:     month_msg = "July";         break;
        case 8:     month_msg = "August";       break;
        case 9:     month_msg = "September";    break;
        case 10:    month_msg = "October";      break;
        case 11:    month_msg = "November";     break;
        case 12:    month_msg = "December";     break;
        default:    month_msg = "Unknown";      break;
    }
    switch (day%10) {
    case 1:     suffix = "st";      break;
    case 2:     suffix = "nd";      break;
    case 3:     suffix = "rd";      break;
    default:    suffix = "th";      break;
    }
    for (i = NUM_COLS * NUM_ROWS; i < NUM_COLS * (NUM_ROWS + 1 ); i++) {
        *(uint8_t *)(video_mem + (i << 1)) = ' ';
        *(uint8_t *)(video_mem + (i << 1) + 1) = ATTRIB;
    }
    // set cursor to the last line, print from left for date
    term_buf[cur_term].screen_x = 1;
    term_buf[cur_term].screen_y = NUM_ROWS;
    printf("%s %d%s ", month_msg, day, suffix);
    if( hour < 10 ){ printf("0"); } printf("%d:",hour);
    if( minute < 10 ){ printf("0"); } printf("%d:",minute);
    if( second < 10 ){ printf("0"); } printf("%d ",second);
    printf("%d UTC-5",year);
    // print from right for current mode
    switch (cur_kernel_mode[cur_term]) {
    case SHELL_MODE:    mode_msg = "Shell Mode | Terminal 1";        break;
    case SPEAKER_MODE:  mode_msg = "Keyboard Piano Mode | Terminal 1";      break;
    default:            mode_msg = "Unknown Mode | Terminal 1";      break;
    }
    switch (cur_term) {
    case 0: mode_msg[strlen(mode_msg)-1] = '1';   break;
    case 1: mode_msg[strlen(mode_msg)-1] = '2';   break;
    case 2: mode_msg[strlen(mode_msg)-1] = '3';   break;
    default: mode_msg[strlen(mode_msg)-1] = 'x';  break;
    }
    term_buf[cur_term].screen_x = NUM_COLS-strlen(mode_msg)-1;
    term_buf[cur_term].screen_y = NUM_ROWS;
    printf("%s",mode_msg);
    // restore settings
    term_buf[cur_term].screen_x = orig_x;
    term_buf[cur_term].screen_y = orig_y;
    front_color = front_color_save;
    back_color = back_color_save;
    status_bar_flag = 0;
    sti();
}


/* set_kb_flag
 * Inputs: void
 * Return Value: none
 * Function: Indicate current output is given by keyboard */
void set_kb_flag(){ keyboard_flag = 1; }

/* clear_kb_flag
 * Inputs: void
 * Return Value: none
 * Function: Clear keyboard input flag */
void clear_kb_flag(){ keyboard_flag = 0; }

/* void set_cursor_top(void);
 * Inputs: void
 * Return Value: none
 * Function: Set cursor to the top */
void set_text_top(void) {
    int32_t cur_out = keyboard_flag ? cur_term : cur_sched;
    term_buf[cur_out].screen_x = 0;
    term_buf[cur_out].screen_y = 0;
}

/* void enable_cursor(void);
 * Inputs: cursor_start, cursor_end
 * Return Value: none
 * Function: enable cursor with cursor_start, cursor_end */
void enable_cursor(uint8_t cursor_start, uint8_t cursor_end)
{
	outb(ST_SCAN, VGA_PORT_1);
	outb((inb(VGA_PORT_2) & 0xC0) | cursor_start, VGA_PORT_2);
 
	outb(ED_SCAN, VGA_PORT_1);
	outb((inb(VGA_PORT_2) & 0xE0) | cursor_end, VGA_PORT_2);
}

/* void disable_cursor(void);
 * Inputs: none
 * Return Value: none
 * Function: disable cursor */
void disable_cursor()
{
	outb(ST_SCAN, VGA_PORT_1);
	outb(DISAB, VGA_PORT_2);
}

/* void update_cursor(int x, int y);
 * Inputs: x, y -- position of cursor
 * Return Value: none
 * Function: update cursor to the given position*/
void update_cursor(int x, int y)
{
	uint16_t pos = y * NUM_COLS + x;
 
	outb(LOWER, VGA_PORT_1);
	outb((uint8_t) (pos & 0xFF), VGA_PORT_2);
	outb(UPPER, VGA_PORT_1);
	outb((uint8_t) ((pos >> 8) & 0xFF), VGA_PORT_2);
}

/* void update_term_cursor()
 * Inputs: None
 * Return Value: None
 * Function: update cursor for current shown terminal*/
void update_term_cursor(){
    uint16_t pos = term_buf[cur_term].screen_y * NUM_COLS + term_buf[cur_term].screen_x;
    outb(LOWER, VGA_PORT_1);
	outb((uint8_t) (pos & 0xFF), VGA_PORT_2);
	outb(UPPER, VGA_PORT_1);
	outb((uint8_t) ((pos >> 8) & 0xFF), VGA_PORT_2);
}

/* void get_cursor_position(void);
 * Inputs: none
 * Return Value: position in form of y * NUM_COLS + x
 */
uint16_t get_cursor_position(void)
{
    uint16_t pos = 0;
    outb(LOWER, VGA_PORT_1);
    pos |= inb(VGA_PORT_2);
    outb(UPPER, VGA_PORT_1);
    pos |= ((uint16_t)inb(VGA_PORT_2)) << 8;
    return pos;
}

/* void clear(void);
 * Inputs: void
 * Return Value: none
 * Function: Clears video memory */
void clear(void) {
    int32_t i;
    for (i = 0; i < NUM_ROWS * NUM_COLS; i++) {
        *(uint8_t *)(video_mem + (i << 1)) = ' ';
        *(uint8_t *)(video_mem + (i << 1) + 1) = ATTRIB;
    }
    set_text_top();
}

/* void blue(void);
 * Inputs: void
 * Return Value: none
 * Function: blues video memory by offsetting ATTRIB */
void blue(void) {
    back_color = 0x10;  // set high 4 bit in ATTRIB byte for background color, 0x1 stands for blue
}

/* Standard printf().
 * Only supports the following format strings:
 * %%  - print a literal '%' character
 * %x  - print a number in hexadecimal
 * %u  - print a number as an unsigned integer
 * %d  - print a number as a signed integer
 * %c  - print a character
 * %s  - print a string
 * %#x - print a number in 32-bit aligned hexadecimal, i.e.
 *       print 8 hexadecimal digits, zero-padded on the left.
 *       For example, the hex number "E" would be printed as
 *       "0000000E".
 *       Note: This is slightly different than the libc specification
 *       for the "#" modifier (this implementation doesn't add a "0x" at
 *       the beginning), but I think it's more flexible this way.
 *       Also note: %x is the only conversion specifier that can use
 *       the "#" modifier to alter output. */
int32_t printf(int8_t *format, ...) {

    /* Pointer to the format string */
    int8_t* buf = format;

    /* Stack pointer for the other parameters */
    int32_t* esp = (void *)&format;
    esp++;

    while (*buf != '\0') {
        switch (*buf) {
            case '%':
                {
                    int32_t alternate = 0;
                    buf++;

format_char_switch:
                    /* Conversion specifiers */
                    switch (*buf) {
                        /* Print a literal '%' character */
                        case '%':
                            putc('%');
                            break;

                        /* Use alternate formatting */
                        case '#':
                            alternate = 1;
                            buf++;
                            /* Yes, I know gotos are bad.  This is the
                             * most elegant and general way to do this,
                             * IMHO. */
                            goto format_char_switch;

                        /* Print a number in hexadecimal form */
                        case 'x':
                            {
                                int8_t conv_buf[64];
                                if (alternate == 0) {
                                    itoa(*((uint32_t *)esp), conv_buf, 16);
                                    puts(conv_buf);
                                } else {
                                    int32_t starting_index;
                                    int32_t i;
                                    itoa(*((uint32_t *)esp), &conv_buf[8], 16);
                                    i = starting_index = strlen(&conv_buf[8]);
                                    while(i < 8) {
                                        conv_buf[i] = '0';
                                        i++;
                                    }
                                    puts(&conv_buf[starting_index]);
                                }
                                esp++;
                            }
                            break;

                        /* Print a number in unsigned int form */
                        case 'u':
                            {
                                int8_t conv_buf[36];
                                itoa(*((uint32_t *)esp), conv_buf, 10);
                                puts(conv_buf);
                                esp++;
                            }
                            break;

                        /* Print a number in signed int form */
                        case 'd':
                            {
                                int8_t conv_buf[36];
                                int32_t value = *((int32_t *)esp);
                                if(value < 0) {
                                    conv_buf[0] = '-';
                                    itoa(-value, &conv_buf[1], 10);
                                } else {
                                    itoa(value, conv_buf, 10);
                                }
                                puts(conv_buf);
                                esp++;
                            }
                            break;

                        /* Print a single character */
                        case 'c':
                            putc((uint8_t) *((int32_t *)esp));
                            esp++;
                            break;

                        /* Print a NULL-terminated string */
                        case 's':
                            puts(*((int8_t **)esp));
                            esp++;
                            break;

                        default:
                            break;
                    }

                }
                break;

            default:
                putc(*buf);
                break;
        }
        buf++;
    }
    return (buf - format);
}

/* int32_t puts(int8_t* s);
 *   Inputs: int_8* s = pointer to a string of characters
 *   Return Value: Number of bytes written
 *    Function: Output a string to the console */
int32_t puts(int8_t* s) {
    register int32_t index = 0;
    while (s[index] != '\0') {
        putc(s[index]);
        index++;
    }
    return index;
}

/* void scroll() 
 *   Inputs: none
 *   Return Value: none
 *   Function: scroll the screen for one row */
void scroll() {
    int32_t cur_out = keyboard_flag ? cur_term : cur_sched;
	int i;
    term_buf[cur_out].screen_y--;
	for (i = 0; i < NUM_ROWS - 1; i++) {
		memcpy((uint8_t *)(video_mem + (i * NUM_COLS * 2)),
			   (uint8_t *)(video_mem + ((i + 1) * NUM_COLS * 2)),
			   NUM_COLS * 2);
	}
    for (i = 0; i < NUM_COLS; i++) {
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + i) << 1)) = ' ';
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1) + 1) = ATTRIB;
    }
    if (mouse_info.cur_y > 0) 
        reset_attrib(mouse_info.cur_x, mouse_info.cur_y - 1);
    set_attrib(mouse_info.cur_x, mouse_info.cur_y);
}

/* void putc(uint8_t c);
 * Inputs: uint_8* c = character to print
 * Return Value: void
 *  Function: Output a character to the console */
void putc(uint8_t c) {
    int32_t cur_out = (keyboard_flag||status_bar_flag) ? cur_term : cur_sched;
    if(c == '\n' || c == '\r') {
        term_buf[cur_out].screen_y++;
        term_buf[cur_out].screen_x = 0;
        term_buf[cur_out].screen_y = (term_buf[cur_out].screen_y + (term_buf[cur_out].screen_x / NUM_COLS));
        if(!status_bar_flag){ if (term_buf[cur_out].screen_y == NUM_ROWS) scroll(); }
    } else {
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1)) = c;
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1) + 1) = ATTRIB;
        term_buf[cur_out].screen_x++;
        term_buf[cur_out].screen_y = (term_buf[cur_out].screen_y + (term_buf[cur_out].screen_x / NUM_COLS));
        if(!status_bar_flag){ if (term_buf[cur_out].screen_y == NUM_ROWS) scroll(); }
        term_buf[cur_out].screen_x %= NUM_COLS;
    }
    if(keyboard_flag || ((!status_bar_flag)&&(cur_term==cur_sched))){ update_cursor(term_buf[cur_out].screen_x,term_buf[cur_out].screen_y); }
}


/* int8_t* itoa(uint32_t value, int8_t* buf, int32_t radix);
 * Inputs: uint32_t value = number to convert
 *            int8_t* buf = allocated buffer to place string in
 *          int32_t radix = base system. hex, oct, dec, etc.
 * Return Value: number of bytes written
 * Function: Convert a number to its ASCII representation, with base "radix" */
int8_t* itoa(uint32_t value, int8_t* buf, int32_t radix) {
    static int8_t lookup[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int8_t *newbuf = buf;
    int32_t i;
    uint32_t newval = value;

    /* Special case for zero */
    if (value == 0) {
        buf[0] = '0';
        buf[1] = '\0';
        return buf;
    }

    /* Go through the number one place value at a time, and add the
     * correct digit to "newbuf".  We actually add characters to the
     * ASCII string from lowest place value to highest, which is the
     * opposite of how the number should be printed.  We'll reverse the
     * characters later. */
    while (newval > 0) {
        i = newval % radix;
        *newbuf = lookup[i];
        newbuf++;
        newval /= radix;
    }

    /* Add a terminating NULL */
    *newbuf = '\0';

    /* Reverse the string and return */
    return strrev(buf);
}

/* int8_t* strrev(int8_t* s);
 * Inputs: int8_t* s = string to reverse
 * Return Value: reversed string
 * Function: reverses a string s */
int8_t* strrev(int8_t* s) {
    register int8_t tmp;
    register int32_t beg = 0;
    register int32_t end = strlen(s) - 1;

    while (beg < end) {
        tmp = s[end];
        s[end] = s[beg];
        s[beg] = tmp;
        beg++;
        end--;
    }
    return s;
}

/* uint32_t strlen(const int8_t* s);
 * Inputs: const int8_t* s = string to take length of
 * Return Value: length of string s
 * Function: return length of string s */
uint32_t strlen(const int8_t* s) {
    register uint32_t len = 0;
    while (s[len] != '\0')
        len++;
    return len;
}

/* void* memset(void* s, int32_t c, uint32_t n);
 * Inputs:    void* s = pointer to memory
 *          int32_t c = value to set memory to
 *         uint32_t n = number of bytes to set
 * Return Value: new string
 * Function: set n consecutive bytes of pointer s to value c */
void* memset(void* s, int32_t c, uint32_t n) {
    c &= 0xFF;
    asm volatile ("                 \n\
            .memset_top:            \n\
            testl   %%ecx, %%ecx    \n\
            jz      .memset_done    \n\
            testl   $0x3, %%edi     \n\
            jz      .memset_aligned \n\
            movb    %%al, (%%edi)   \n\
            addl    $1, %%edi       \n\
            subl    $1, %%ecx       \n\
            jmp     .memset_top     \n\
            .memset_aligned:        \n\
            movw    %%ds, %%dx      \n\
            movw    %%dx, %%es      \n\
            movl    %%ecx, %%edx    \n\
            shrl    $2, %%ecx       \n\
            andl    $0x3, %%edx     \n\
            cld                     \n\
            rep     stosl           \n\
            .memset_bottom:         \n\
            testl   %%edx, %%edx    \n\
            jz      .memset_done    \n\
            movb    %%al, (%%edi)   \n\
            addl    $1, %%edi       \n\
            subl    $1, %%edx       \n\
            jmp     .memset_bottom  \n\
            .memset_done:           \n\
            "
            :
            : "a"(c << 24 | c << 16 | c << 8 | c), "D"(s), "c"(n)
            : "edx", "memory", "cc"
    );
    return s;
}

/* void* memset_word(void* s, int32_t c, uint32_t n);
 * Description: Optimized memset_word
 * Inputs:    void* s = pointer to memory
 *          int32_t c = value to set memory to
 *         uint32_t n = number of bytes to set
 * Return Value: new string
 * Function: set lower 16 bits of n consecutive memory locations of pointer s to value c */
void* memset_word(void* s, int32_t c, uint32_t n) {
    asm volatile ("                 \n\
            movw    %%ds, %%dx      \n\
            movw    %%dx, %%es      \n\
            cld                     \n\
            rep     stosw           \n\
            "
            :
            : "a"(c), "D"(s), "c"(n)
            : "edx", "memory", "cc"
    );
    return s;
}

/* void* memset_dword(void* s, int32_t c, uint32_t n);
 * Inputs:    void* s = pointer to memory
 *          int32_t c = value to set memory to
 *         uint32_t n = number of bytes to set
 * Return Value: new string
 * Function: set n consecutive memory locations of pointer s to value c */
void* memset_dword(void* s, int32_t c, uint32_t n) {
    asm volatile ("                 \n\
            movw    %%ds, %%dx      \n\
            movw    %%dx, %%es      \n\
            cld                     \n\
            rep     stosl           \n\
            "
            :
            : "a"(c), "D"(s), "c"(n)
            : "edx", "memory", "cc"
    );
    return s;
}

/* void* memcpy(void* dest, const void* src, uint32_t n);
 * Inputs:      void* dest = destination of copy
 *         const void* src = source of copy
 *              uint32_t n = number of byets to copy
 * Return Value: pointer to dest
 * Function: copy n bytes of src to dest */
void* memcpy(void* dest, const void* src, uint32_t n) {
    asm volatile ("                 \n\
            .memcpy_top:            \n\
            testl   %%ecx, %%ecx    \n\
            jz      .memcpy_done    \n\
            testl   $0x3, %%edi     \n\
            jz      .memcpy_aligned \n\
            movb    (%%esi), %%al   \n\
            movb    %%al, (%%edi)   \n\
            addl    $1, %%edi       \n\
            addl    $1, %%esi       \n\
            subl    $1, %%ecx       \n\
            jmp     .memcpy_top     \n\
            .memcpy_aligned:        \n\
            movw    %%ds, %%dx      \n\
            movw    %%dx, %%es      \n\
            movl    %%ecx, %%edx    \n\
            shrl    $2, %%ecx       \n\
            andl    $0x3, %%edx     \n\
            cld                     \n\
            rep     movsl           \n\
            .memcpy_bottom:         \n\
            testl   %%edx, %%edx    \n\
            jz      .memcpy_done    \n\
            movb    (%%esi), %%al   \n\
            movb    %%al, (%%edi)   \n\
            addl    $1, %%edi       \n\
            addl    $1, %%esi       \n\
            subl    $1, %%edx       \n\
            jmp     .memcpy_bottom  \n\
            .memcpy_done:           \n\
            "
            :
            : "S"(src), "D"(dest), "c"(n)
            : "eax", "edx", "memory", "cc"
    );
    return dest;
}

/* void* memmove(void* dest, const void* src, uint32_t n);
 * Description: Optimized memmove (used for overlapping memory areas)
 * Inputs:      void* dest = destination of move
 *         const void* src = source of move
 *              uint32_t n = number of byets to move
 * Return Value: pointer to dest
 * Function: move n bytes of src to dest */
void* memmove(void* dest, const void* src, uint32_t n) {
    asm volatile ("                             \n\
            movw    %%ds, %%dx                  \n\
            movw    %%dx, %%es                  \n\
            cld                                 \n\
            cmp     %%edi, %%esi                \n\
            jae     .memmove_go                 \n\
            leal    -1(%%esi, %%ecx), %%esi     \n\
            leal    -1(%%edi, %%ecx), %%edi     \n\
            std                                 \n\
            .memmove_go:                        \n\
            rep     movsb                       \n\
            "
            :
            : "D"(dest), "S"(src), "c"(n)
            : "edx", "memory", "cc"
    );
    return dest;
}

/* int32_t strncmp(const int8_t* s1, const int8_t* s2, uint32_t n)
 * Inputs: const int8_t* s1 = first string to compare
 *         const int8_t* s2 = second string to compare
 *               uint32_t n = number of bytes to compare
 * Return Value: A zero value indicates that the characters compared
 *               in both strings form the same string.
 *               A value greater than zero indicates that the first
 *               character that does not match has a greater value
 *               in str1 than in str2; And a value less than zero
 *               indicates the opposite.
 * Function: compares string 1 and string 2 for equality */
int32_t strncmp(const int8_t* s1, const int8_t* s2, uint32_t n) {
    int32_t i;
    for (i = 0; i < n; i++) {
        if ((s1[i] != s2[i]) || (s1[i] == '\0') /* || s2[i] == '\0' */) {

            /* The s2[i] == '\0' is unnecessary because of the short-circuit
             * semantics of 'if' expressions in C.  If the first expression
             * (s1[i] != s2[i]) evaluates to false, that is, if s1[i] ==
             * s2[i], then we only need to test either s1[i] or s2[i] for
             * '\0', since we know they are equal. */
            return s1[i] - s2[i];
        }
    }
    return 0;
}

/* int8_t* strcpy(int8_t* dest, const int8_t* src)
 * Inputs:      int8_t* dest = destination string of copy
 *         const int8_t* src = source string of copy
 * Return Value: pointer to dest
 * Function: copy the source string into the destination string */
int8_t* strcpy(int8_t* dest, const int8_t* src) {
    int32_t i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
    return dest;
}

/* int8_t* strcpy(int8_t* dest, const int8_t* src, uint32_t n)
 * Inputs:      int8_t* dest = destination string of copy
 *         const int8_t* src = source string of copy
 *                uint32_t n = number of bytes to copy
 * Return Value: pointer to dest
 * Function: copy n bytes of the source string into the destination string */
int8_t* strncpy(int8_t* dest, const int8_t* src, uint32_t n) {
    int32_t i = 0;
    while (src[i] != '\0' && i < n) {
        dest[i] = src[i];
        i++;
    }
    while (i < n) {
        dest[i] = '\0';
        i++;
    }
    return dest;
}

/* void test_interrupts(void)
 * Inputs: void
 * Return Value: void
 * Function: increments video memory. To be used to test rtc */
void test_interrupts(void) {
    int32_t i;
    for (i = 0; i < NUM_ROWS * NUM_COLS; i++) {
        video_mem[i << 1]++;
    }
}

/* void backspace()
 * Inputs: none
 * Return Value: none
 * Function: erase a character on the screen*/
void backspace() {
    int32_t cur_out = keyboard_flag ? cur_term : cur_sched;
    if (term_buf[cur_out].screen_x == 0 && term_buf[cur_out].screen_y > 0){
        term_buf[cur_out].screen_y--;
        term_buf[cur_out].screen_x = NUM_COLS - 1;
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1)) = ' ';
        *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1) + 1) = ATTRIB;
        if(keyboard_flag || (cur_term==cur_sched)){ update_cursor(term_buf[cur_out].screen_x,term_buf[cur_out].screen_y); }
        return ;
    }
    term_buf[cur_out].screen_x--;
    *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1)) = ' ';
    *(uint8_t *)(video_mem + ((NUM_COLS * term_buf[cur_out].screen_y + term_buf[cur_out].screen_x) << 1) + 1) = ATTRIB;
    if(keyboard_flag || (cur_term==cur_sched)){ update_cursor(term_buf[cur_out].screen_x,term_buf[cur_out].screen_y); }
}
